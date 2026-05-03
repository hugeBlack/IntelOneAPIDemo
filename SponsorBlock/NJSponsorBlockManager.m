//
//  NJSponsorBlockManager.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockManager.h"
#import "NJSponsorBlockSegment.h"
#import "NJSponsorBlockService.h"
#import "NJSponsorBlockSettings.h"
#import "NJSponsorBlockCacheStats.h"
#import "NJCommonDefine.h"
#import "NJSettingCache.h"
#import <math.h>
#import <objc/runtime.h>

NSNotificationName const NJSponsorBlockStateDidChangeNotification = @"NJSponsorBlockStateDidChangeNotification";
NSNotificationName const NJSponsorBlockManualSkipRequestNotification = @"NJSponsorBlockManualSkipRequestNotification";
NSNotificationName const NJSponsorBlockSeekRequestNotification = @"NJSponsorBlockSeekRequestNotification";

static NSString * const NJSponsorBlockCachePrefix = @"NJSponsorBlockSegments";
static NSTimeInterval const NJSponsorBlockCacheTTL = 24 * 60 * 60;
static NSTimeInterval const NJSponsorBlockCooldown = 1.0;

@interface NJSponsorBlockCacheItem : NSObject <NSSecureCoding>

@property (nonatomic, copy) NSArray<NJSponsorBlockSegment *> *segments;
@property (nonatomic, strong) NSDate *date;

@end

@implementation NJSponsorBlockCacheItem

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder {
    [coder encodeObject:self.segments forKey:@"segments"];
    [coder encodeObject:self.date forKey:@"date"];
}

- (instancetype)initWithCoder:(NSCoder *)coder {
    self = [super init];
    if (self) {
        NSSet *classes = [NSSet setWithObjects:[NSArray class], [NJSponsorBlockSegment class], nil];
        self.segments = [coder decodeObjectOfClasses:classes forKey:@"segments"] ?: @[];
        self.date = [coder decodeObjectOfClass:[NSDate class] forKey:@"date"] ?: [NSDate distantPast];
    }
    return self;
}

@end

@interface NJSponsorBlockManager ()

@property (nonatomic, copy) NSString *videoID;
@property (nonatomic, assign) NSInteger cid;
@property (nonatomic, strong) NSArray<NJSponsorBlockSegment *> *segments;
@property (nonatomic, strong) NSMutableSet<NSString *> *skippedUUIDs;
@property (nonatomic, strong) NSMutableSet<NSString *> *actualSkippedUUIDs;
@property (nonatomic, strong) NJSponsorBlockService *service;
@property (nonatomic, strong) NSDate *cooldownUntil;
@property (nonatomic, strong) NJSponsorBlockSegment *lastSkippedSegment;
@property (nonatomic, assign) NSTimeInterval lastProbeLogTime;
@property (nonatomic, assign) NSTimeInterval currentPlaybackTime;
@property (nonatomic, assign) NSTimeInterval nativeVideoDuration;
@property (nonatomic, copy) NSString *loadedServerBaseURLString;
@property (nonatomic, strong) NSMutableSet<NSString *> *trackedCacheKeys;

- (void)updateNativeVideoDuration:(NSTimeInterval)duration;
- (NSTimeInterval)videoDurationInObject:(id)object;
- (NSTimeInterval)durationInDictionary:(NSDictionary *)dictionary;
- (NSTimeInterval)durationFromKey:(NSString *)key value:(id)value;
- (NSTimeInterval)durationFromCandidateAccessorsOfObject:(id)object;
- (void)invalidateCachedSegmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid;
- (NSUInteger)estimatedSizeForSegments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (NSError *)submissionErrorWithCode:(NSInteger)code message:(NSString *)message;

@end

@implementation NJSponsorBlockManager

+ (instancetype)sharedInstance {
    static NJSponsorBlockManager *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[NJSponsorBlockManager alloc] init];
    });
    return instance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        self.videoID = @"";
        self.segments = @[];
        self.loadedServerBaseURLString = @"";
        self.skippedUUIDs = [NSMutableSet set];
        self.actualSkippedUUIDs = [NSMutableSet set];
        self.service = [[NJSponsorBlockService alloc] init];
        self.cooldownUntil = [NSDate distantPast];
        self.lastProbeLogTime = -100;
        self.trackedCacheKeys = [NSMutableSet set];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(handleSettingsDidChange)
                                                     name:NJSponsorBlockSettingsDidChangeNotification
                                                   object:nil];
    }
    return self;
}

- (void)updateVideoID:(NSString *)videoID cid:(NSInteger)cid {
    if (videoID.length == 0 || cid <= 0) {
        return;
    }
    if ([self.videoID isEqualToString:videoID] && self.cid == cid) {
        return;
    }
    
    self.videoID = videoID;
    self.cid = cid;
    self.segments = @[];
    self.loadedServerBaseURLString = @"";
    self.nativeVideoDuration = 0;
    self.lastSkippedSegment = nil;
    [self.skippedUUIDs removeAllObjects];
    [self.actualSkippedUUIDs removeAllObjects];
    [self postStateChangedNotification];
    NSLog(@"[NJSponsorBlock] update video %@:%ld", videoID, (long)cid);
    [self loadSegmentsForCurrentVideoIfNeeded];
}

- (void)inspectResponseData:(NSData *)data response:(NSURLResponse *)response {
    if (![NJSponsorBlockSettings enabled] || data.length == 0) {
        return;
    }
    
    if (![self shouldInspectResponse:response data:data]) {
        return;
    }
    
    NSError *error = nil;
    id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (error || !json) {
        return;
    }
    
    NSDictionary *identity = [self videoIdentityInObject:json];
    NSString *videoID = identity[@"videoID"];
    NSNumber *cid = identity[@"cid"];
    if (videoID.length == 0 || cid.integerValue <= 0) {
        return;
    }
    NSLog(@"[NJSponsorBlock] found video identity %@:%ld from %@", videoID, (long)cid.integerValue, response.URL.absoluteString);
    [self updateVideoID:videoID cid:cid.integerValue];
    [self updateNativeVideoDuration:[self videoDurationInObject:json]];
}

- (void)inspectModelObject:(id)object source:(NSString *)source {
    if (![NJSponsorBlockSettings enabled] || !object) {
        return;
    }
    
    NSMutableSet<NSValue *> *visited = [NSMutableSet set];
    NSString *__block videoID = nil;
    NSNumber *__block cid = nil;
    NSTimeInterval __block duration = 0;
    [self collectVideoIdentityFromObject:object
                                   depth:0
                                 visited:visited
                                 videoID:&videoID
                                     cid:&cid
                                duration:&duration];
    if (videoID.length == 0 || cid.integerValue <= 0) {
        NSLog(@"[NJSponsorBlock] model identity not found from %@ %@", source ?: @"unknown", NSStringFromClass([object class]));
        return;
    }
    
    NSLog(@"[NJSponsorBlock] found video identity %@:%ld from model %@ %@",
          videoID,
          (long)cid.integerValue,
          source ?: @"unknown",
          NSStringFromClass([object class]));
    [self updateVideoID:videoID cid:cid.integerValue];
    [self updateNativeVideoDuration:duration];
}

- (BOOL)shouldInspectResponse:(NSURLResponse *)response data:(NSData *)data {
    NSString *url = response.URL.absoluteString.lowercaseString ?: @"";
    NSString *mimeType = response.MIMEType.lowercaseString ?: @"";
    if ([url containsString:@"view"] ||
        [url containsString:@"detail"] ||
        [url containsString:@"player"] ||
        [url containsString:@"playurl"] ||
        [url containsString:@"archive"]) {
        return YES;
    }
    
    if (![mimeType containsString:@"json"] || data.length > 2 * 1024 * 1024) {
        return NO;
    }
    
    NSData *bvidData = [@"bvid" dataUsingEncoding:NSUTF8StringEncoding];
    NSData *cidData = [@"cid" dataUsingEncoding:NSUTF8StringEncoding];
    NSRange range = NSMakeRange(0, data.length);
    return [data rangeOfData:bvidData options:0 range:range].location != NSNotFound &&
           [data rangeOfData:cidData options:0 range:range].location != NSNotFound;
}

- (NSArray<NJSponsorBlockSegment *> *)allSegments {
    if (![NJSponsorBlockSettings enabled] || self.segments.count == 0) {
        return @[];
    }
    return self.segments;
}

- (NSArray<NJSponsorBlockSegment *> *)displaySegments {
    NSArray<NJSponsorBlockSegment *> *allSegments = [self allSegments];
    if (allSegments.count == 0) {
        return @[];
    }

    NSMutableArray<NJSponsorBlockSegment *> *segments = [NSMutableArray array];
    for (NJSponsorBlockSegment *segment in allSegments) {
        if ([NJSponsorBlockSettings shouldShowSegment:segment]) {
            [segments addObject:segment];
        }
    }
    return [segments copy];
}

- (NJSponsorBlockSegment *)activeSegmentAtPlaybackTime:(NSTimeInterval)time {
    for (NJSponsorBlockSegment *segment in [self displaySegments]) {
        if ([segment containsPlaybackTime:time]) {
            return segment;
        }
    }
    return nil;
}

- (NJSponsorBlockSegment *)autoSkipSegmentAtPlaybackTime:(NSTimeInterval)time {
    return [self autoSkipSegmentsAtPlaybackTime:time].lastObject;
}

- (NSArray<NJSponsorBlockSegment *> *)autoSkipSegmentsAtPlaybackTime:(NSTimeInterval)time {
    NSMutableArray<NJSponsorBlockSegment *> *targetSegments = [NSMutableArray array];
    NSTimeInterval targetEndTime = 0;
    for (NJSponsorBlockSegment *segment in [self displaySegments]) {
        if ([self hasSkippedSegment:segment] || ![NJSponsorBlockSettings shouldAutoSkipSegment:segment]) {
            continue;
        }
        if (targetSegments.count == 0) {
            if ([segment containsPlaybackTime:time]) {
                [targetSegments addObject:segment];
                targetEndTime = segment.endTime;
            }
            continue;
        }
        if (segment.startTime <= targetEndTime) {
            [targetSegments addObject:segment];
            targetEndTime = MAX(targetEndTime, segment.endTime);
        }
    }
    return [targetSegments copy];
}

- (NJSponsorBlockSegment *)manualSkipSegmentAtPlaybackTime:(NSTimeInterval)time {
    for (NJSponsorBlockSegment *segment in [self displaySegments]) {
        if ([self hasSkippedSegment:segment] || ![NJSponsorBlockSettings shouldManualSkipSegment:segment]) {
            continue;
        }
        if ([segment containsPlaybackTime:time]) {
            return segment;
        }
    }
    return nil;
}

- (NJSponsorBlockSegment *)upcomingAutoSkipSegmentAtPlaybackTime:(NSTimeInterval)time withinSeconds:(NSTimeInterval)seconds {
    if (seconds <= 0) {
        return nil;
    }
    for (NJSponsorBlockSegment *segment in [self displaySegments]) {
        if ([self hasSkippedSegment:segment] || ![NJSponsorBlockSettings shouldAutoSkipSegment:segment]) {
            continue;
        }
        NSTimeInterval remaining = segment.startTime - time;
        if (remaining > 0 && remaining <= seconds) {
            return segment;
        }
    }
    return nil;
}

- (NSTimeInterval)skippedDurationBeforePlaybackTime:(NSTimeInterval)time {
    NSTimeInterval skippedDuration = 0;
    for (NJSponsorBlockSegment *segment in [self displaySegments]) {
        if (![self.actualSkippedUUIDs containsObject:segment.uuid] || segment.endTime <= segment.startTime) {
            continue;
        }
        NSTimeInterval effectiveEndTime = MIN(segment.endTime, time);
        if (effectiveEndTime > segment.startTime) {
            skippedDuration += effectiveEndTime - segment.startTime;
        }
    }
    return MAX(0, skippedDuration);
}

- (NSTimeInterval)playbackTimeWithoutSkippedSegments:(NSTimeInterval)time {
    return MAX(0, time - [self skippedDurationBeforePlaybackTime:time]);
}

- (BOOL)skipOnSeekToSegment {
    return [NJSponsorBlockSettings skipOnSeekToSegment];
}

- (void)handlePlaybackTimeForProbe:(NSTimeInterval)time {
    if (![NJSponsorBlockSettings enabled]) {
        return;
    }
    
    BOOL shouldNotify = fabs(time - self.currentPlaybackTime) >= 0.5 || time < self.currentPlaybackTime;
    self.currentPlaybackTime = time;
    if (shouldNotify) {
        [self postStateChangedNotification];
    }
    
    NJSponsorBlockSegment *segment = [self activeSegmentAtPlaybackTime:time];
    if (segment) {
        NSLog(@"[NJSponsorBlock] active segment %@ %.2f-%.2f current=%.2f", segment.uuid, segment.startTime, segment.endTime, time);
        return;
    }
    
    if (time - self.lastProbeLogTime >= 5.0 || time < self.lastProbeLogTime) {
        self.lastProbeLogTime = time;
        NSLog(@"[NJSponsorBlock] playback time %.2f video=%@ cid=%ld segments=%lu", time, self.videoID, (long)self.cid, (unsigned long)self.segments.count);
    }
}

- (void)markSegmentSkipped:(NJSponsorBlockSegment *)segment {
    if (segment.uuid.length == 0) {
        return;
    }
    [self.skippedUUIDs addObject:segment.uuid];
}

- (void)clearSkippedSegment:(NJSponsorBlockSegment *)segment {
    if (segment.uuid.length == 0) {
        return;
    }
    [self.skippedUUIDs removeObject:segment.uuid];
    [self.actualSkippedUUIDs removeObject:segment.uuid];
    if ([self.lastSkippedSegment.uuid isEqualToString:segment.uuid]) {
        self.lastSkippedSegment = nil;
    }
    [self postStateChangedNotification];
}

- (void)recordLastSkippedSegment:(NJSponsorBlockSegment *)segment {
    if (segment.uuid.length == 0) {
        return;
    }
    [self.actualSkippedUUIDs addObject:segment.uuid];
    self.lastSkippedSegment = segment;
    [self postStateChangedNotification];
}

- (void)reportSegmentSkipped:(NJSponsorBlockSegment *)segment {
    if (segment.uuid.length == 0 || ![NJSponsorBlockSettings skipTrackingEnabled]) {
        return;
    }
    if (segment.actionType.length == 0 || [segment.actionType isEqualToString:@"skip"]) {
        [self.service reportViewedSegmentWithUUID:segment.uuid];
    }
}

- (void)submitSegmentWithCategory:(NSString *)category
                       actionType:(NSString *)actionType
                          segment:(NSArray<NSNumber *> *)segment
                       completion:(NJSponsorBlockSubmitCompletion)completion {
    if (self.videoID.length == 0 || self.cid <= 0) {
        if (completion) {
            completion(NO, [self submissionErrorWithCode:-1 message:@"尚未识别当前视频"]);
        }
        return;
    }

    NSTimeInterval duration = self.estimatedVideoDuration;
    if (duration <= 0 || isnan(duration) || isinf(duration)) {
        if (completion) {
            completion(NO, [self submissionErrorWithCode:-2 message:@"暂未获取视频时长，稍后再试"]);
        }
        return;
    }

    NSString *videoID = self.videoID;
    NSInteger cid = self.cid;
    __weak typeof(self) weakSelf = self;
    [self.service submitSegmentWithVideoID:videoID
                                       cid:cid
                                  category:category
                                actionType:actionType
                                   segment:segment
                             videoDuration:duration
                                completion:^(BOOL success, NSError *error) {
        if (!success) {
            if (completion) {
                completion(NO, error);
            }
            return;
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf || ![strongSelf.videoID isEqualToString:videoID] || strongSelf.cid != cid) {
                if (completion) {
                    completion(YES, nil);
                }
                return;
            }
            [strongSelf invalidateCachedSegmentsForVideoID:videoID cid:cid];
            strongSelf.segments = @[];
            [strongSelf postStateChangedNotification];
            [strongSelf loadSegmentsForCurrentVideoIfNeeded];
            if (completion) {
                completion(YES, nil);
            }
        });
    }];
}

- (BOOL)hasSkippedSegment:(NJSponsorBlockSegment *)segment {
    if (segment.uuid.length == 0) {
        return NO;
    }
    return [self.skippedUUIDs containsObject:segment.uuid];
}

- (BOOL)hasActuallySkippedSegment:(NJSponsorBlockSegment *)segment {
    if (segment.uuid.length == 0) {
        return NO;
    }
    return [self.actualSkippedUUIDs containsObject:segment.uuid];
}

- (BOOL)isInCooldown {
    return [[NSDate date] compare:self.cooldownUntil] == NSOrderedAscending;
}

- (void)enterCooldown {
    self.cooldownUntil = [NSDate dateWithTimeIntervalSinceNow:NJSponsorBlockCooldown];
}

- (void)updateNativeVideoDuration:(NSTimeInterval)duration {
    if (duration <= 0 || !isfinite(duration)) {
        return;
    }
    if (fabs(self.nativeVideoDuration - duration) < 0.5) {
        return;
    }
    self.nativeVideoDuration = duration;
    [self postStateChangedNotification];
}

- (NSTimeInterval)estimatedVideoDuration {
    NSTimeInterval duration = self.nativeVideoDuration;
    for (NJSponsorBlockSegment *segment in self.segments) {
        duration = MAX(duration, segment.videoDuration);
        duration = MAX(duration, segment.endTime);
    }
    return duration;
}

- (void)loadSegmentsForCurrentVideoIfNeeded {
    if (![NJSponsorBlockSettings enabled] || self.videoID.length == 0 || self.cid <= 0) {
        return;
    }

    NSString *serverBaseURLString = [NJSponsorBlockSettings serverBaseURLString];
    NSArray<NJSponsorBlockSegment *> *cachedSegments = [self cachedSegmentsForVideoID:self.videoID cid:self.cid];
    if (cachedSegments) {
        self.segments = cachedSegments;
        self.loadedServerBaseURLString = serverBaseURLString;
        [self postStateChangedNotification];
        return;
    }

    NSString *videoID = self.videoID;
    NSInteger cid = self.cid;
    __weak typeof(self) weakSelf = self;
    [self.service fetchSegmentsWithVideoID:videoID cid:cid categories:@[] completion:^(NSArray<NJSponsorBlockSegment *> *segments, NSError *error) {
        if (error) {
            NSLog(@"[NJSponsorBlock] fetch segments failed: %@", error);
            return;
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf || ![strongSelf.videoID isEqualToString:videoID] || strongSelf.cid != cid || ![serverBaseURLString isEqualToString:[NJSponsorBlockSettings serverBaseURLString]]) {
                return;
            }
            strongSelf.segments = segments;
            strongSelf.loadedServerBaseURLString = serverBaseURLString;
            [strongSelf saveSegments:segments videoID:videoID cid:cid];
            [strongSelf postStateChangedNotification];
            NSLog(@"[NJSponsorBlock] loaded %lu segments for %@:%ld", (unsigned long)segments.count, videoID, (long)cid);
        });
    }];
}

- (NSArray<NJSponsorBlockSegment *> *)cachedSegmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid {
    if (![NJSponsorBlockSettings cacheEnabled]) {
        return nil;
    }
    NJSponsorBlockCacheItem *item = (NJSponsorBlockCacheItem *)[NJ_SETTING_CACHE objectForKey:[self cacheKeyWithVideoID:videoID cid:cid]];
    if (![item isKindOfClass:[NJSponsorBlockCacheItem class]]) {
        return nil;
    }
    if (fabs([item.date timeIntervalSinceNow]) > NJSponsorBlockCacheTTL) {
        return nil;
    }
    NSUInteger size = [self estimatedSizeForSegments:item.segments];
    [[NJSponsorBlockCacheStats sharedInstance] recordHitWithSize:size];
    return item.segments;
}

- (void)saveSegments:(NSArray<NJSponsorBlockSegment *> *)segments videoID:(NSString *)videoID cid:(NSInteger)cid {
    if (![NJSponsorBlockSettings cacheEnabled]) {
        return;
    }
    NSString *key = [self cacheKeyWithVideoID:videoID cid:cid];
    NJSponsorBlockCacheItem *item = [[NJSponsorBlockCacheItem alloc] init];
    item.segments = segments ?: @[];
    item.date = [NSDate date];
    [NJ_SETTING_CACHE setObject:item forKey:key withBlock:nil];
    [self.trackedCacheKeys addObject:key];
    NSUInteger size = [self estimatedSizeForSegments:item.segments];
    [[NJSponsorBlockCacheStats sharedInstance] recordSaveWithSize:size];
}

- (NSString *)cacheKeyWithVideoID:(NSString *)videoID cid:(NSInteger)cid {
    NSString *server = [[NJSponsorBlockSettings serverBaseURLString] stringByReplacingOccurrencesOfString:@"|" withString:@"_"];
    server = [server stringByReplacingOccurrencesOfString:@":" withString:@"-"];
    server = [server stringByReplacingOccurrencesOfString:@"/" withString:@"_"];
    return [NSString stringWithFormat:@"%@_%@_%ld_%@", NJSponsorBlockCachePrefix, videoID, (long)cid, server];
}

- (void)invalidateCachedSegmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid {
    NSString *key = [self cacheKeyWithVideoID:videoID cid:cid];
    [NJ_SETTING_CACHE removeObjectForKey:key];
    [self.trackedCacheKeys removeObject:key];
    [[NJSponsorBlockCacheStats sharedInstance] recordRemoval];
}

- (void)clearAllCachedSegments {
    for (NSString *key in [self.trackedCacheKeys copy]) {
        [NJ_SETTING_CACHE removeObjectForKey:key];
    }
    [self.trackedCacheKeys removeAllObjects];
    [[NJSponsorBlockCacheStats sharedInstance] clearAll];
}

- (NSUInteger)estimatedSizeForSegments:(NSArray<NJSponsorBlockSegment *> *)segments {
    if (segments.count == 0) {
        return 0;
    }
    NSUInteger size = 0;
    for (NJSponsorBlockSegment *seg in segments) {
        size += seg.uuid.length * 2;
        size += seg.videoID.length * 2;
        size += seg.category.length * 2;
        size += seg.actionType.length * 2;
        size += sizeof(NSTimeInterval) * 3;
        size += sizeof(NSInteger);
    }
    return size;
}

- (NSError *)submissionErrorWithCode:(NSInteger)code message:(NSString *)message {
    return [NSError errorWithDomain:@"NJSponsorBlockManager"
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: message ?: @"SponsorBlock submission failed"}];
}

- (void)handleSettingsDidChange {
    NSString *serverBaseURLString = [NJSponsorBlockSettings serverBaseURLString];
    BOOL shouldReload = self.loadedServerBaseURLString.length == 0 || ![self.loadedServerBaseURLString isEqualToString:serverBaseURLString];
    if (shouldReload) {
        self.segments = @[];
        self.loadedServerBaseURLString = @"";
        self.lastSkippedSegment = nil;
        [self.skippedUUIDs removeAllObjects];
        [self.actualSkippedUUIDs removeAllObjects];
    }
    [self postStateChangedNotification];
    if (shouldReload) {
        [self loadSegmentsForCurrentVideoIfNeeded];
    }
}

- (void)postStateChangedNotification {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockStateDidChangeNotification object:self];
    });
}

- (NSDictionary *)videoIdentityInObject:(id)object {
    if ([object isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = (NSDictionary *)object;
        NSString *videoID = [self videoIDInDictionary:dictionary];
        NSNumber *cid = [self cidInDictionary:dictionary];
        if (videoID.length > 0 && cid.integerValue > 0) {
            return @{@"videoID": videoID, @"cid": cid};
        }
        
        for (id value in dictionary.allValues) {
            NSDictionary *identity = [self videoIdentityInObject:value];
            if (identity) {
                return identity;
            }
        }
        return nil;
    }
    
    if ([object isKindOfClass:[NSArray class]]) {
        for (id value in (NSArray *)object) {
            NSDictionary *identity = [self videoIdentityInObject:value];
            if (identity) {
                return identity;
            }
        }
    }
    return nil;
}

- (NSString *)videoIDInDictionary:(NSDictionary *)dictionary {
    for (NSString *key in dictionary) {
        if (![key isKindOfClass:[NSString class]]) {
            continue;
        }
        NSString *lowerKey = key.lowercaseString;
        if (![lowerKey isEqualToString:@"bvid"] &&
            ![lowerKey isEqualToString:@"bvidstr"] &&
            ![lowerKey isEqualToString:@"bvid_str"] &&
            ![lowerKey isEqualToString:@"bv_id"]) {
            continue;
        }
        id value = dictionary[key];
        if ([value isKindOfClass:[NSString class]] && [value hasPrefix:@"BV"]) {
            return value;
        }
    }
    return nil;
}

- (NSNumber *)cidInDictionary:(NSDictionary *)dictionary {
    for (NSString *key in dictionary) {
        if (![key isKindOfClass:[NSString class]] || ![key.lowercaseString isEqualToString:@"cid"]) {
            continue;
        }
        id value = dictionary[key];
        if ([value respondsToSelector:@selector(integerValue)] && [value integerValue] > 0) {
            return @([value integerValue]);
        }
    }
    return nil;
}

- (NSTimeInterval)videoDurationInObject:(id)object {
    if ([object isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = (NSDictionary *)object;
        NSTimeInterval duration = [self durationInDictionary:dictionary];
        if (duration > 0) {
            return duration;
        }
        for (id value in dictionary.allValues) {
            duration = [self videoDurationInObject:value];
            if (duration > 0) {
                return duration;
            }
        }
        return 0;
    }

    if ([object isKindOfClass:[NSArray class]]) {
        for (id value in (NSArray *)object) {
            NSTimeInterval duration = [self videoDurationInObject:value];
            if (duration > 0) {
                return duration;
            }
        }
    }
    return 0;
}

- (NSTimeInterval)durationInDictionary:(NSDictionary *)dictionary {
    for (NSString *key in dictionary) {
        if (![key isKindOfClass:[NSString class]]) {
            continue;
        }
        NSTimeInterval duration = [self durationFromKey:key value:dictionary[key]];
        if (duration > 0) {
            return duration;
        }
    }
    return 0;
}

- (NSTimeInterval)durationFromKey:(NSString *)key value:(id)value {
    if (![value respondsToSelector:@selector(doubleValue)]) {
        return 0;
    }
    NSString *lowerKey = key.lowercaseString;
    NSTimeInterval rawValue = [value doubleValue];
    if (rawValue <= 0 || !isfinite(rawValue)) {
        return 0;
    }
    if ([lowerKey isEqualToString:@"timelength"] ||
        [lowerKey isEqualToString:@"time_length"] ||
        [lowerKey isEqualToString:@"duration_ms"]) {
        return rawValue / 1000.0;
    }
    if ([lowerKey isEqualToString:@"duration"] ||
        [lowerKey isEqualToString:@"video_duration"] ||
        [lowerKey isEqualToString:@"videoduration"]) {
        return rawValue > 86400 ? rawValue / 1000.0 : rawValue;
    }
    return 0;
}

- (NSTimeInterval)durationFromCandidateAccessorsOfObject:(id)object {
    NSArray<NSString *> *selectors = @[@"duration", @"timelength", @"timeLength", @"videoDuration"];
    for (NSString *selectorName in selectors) {
        id value = [self safeValueForSelectorName:selectorName object:object];
        NSTimeInterval duration = [self durationFromKey:selectorName value:value];
        if (duration > 0) {
            return duration;
        }
    }
    return 0;
}

- (void)collectVideoIdentityFromObject:(id)object
                                 depth:(NSInteger)depth
                               visited:(NSMutableSet<NSValue *> *)visited
                               videoID:(NSString **)videoID
                                   cid:(NSNumber **)cid
                              duration:(NSTimeInterval *)duration {
    if (!object || depth > 5 || ((*videoID).length > 0 && (*cid).integerValue > 0 && *duration > 0)) {
        return;
    }
    
    if ([object isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = (NSDictionary *)object;
        NSString *foundVideoID = [self videoIDInDictionary:dictionary];
        NSNumber *foundCID = [self cidInDictionary:dictionary];
        if ((*videoID).length == 0 && foundVideoID.length > 0) {
            *videoID = foundVideoID;
        }
        if ((*cid).integerValue <= 0 && foundCID.integerValue > 0) {
            *cid = foundCID;
        }
        if (*duration <= 0) {
            *duration = [self durationInDictionary:dictionary];
        }
        for (id value in dictionary.allValues) {
            [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
        }
        return;
    }
    
    if ([object isKindOfClass:[NSArray class]] || [object isKindOfClass:[NSSet class]]) {
        for (id value in object) {
            [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
        }
        return;
    }
    
    if ([object isKindOfClass:[NSString class]] ||
        [object isKindOfClass:[NSNumber class]] ||
        [object isKindOfClass:[NSData class]] ||
        [object isKindOfClass:[NSDate class]]) {
        return;
    }
    
    NSValue *pointer = [NSValue valueWithNonretainedObject:object];
    if ([visited containsObject:pointer]) {
        return;
    }
    [visited addObject:pointer];
    
    NSString *className = NSStringFromClass([object class]);
    if (![className hasPrefix:@"BAPI"] && ![className hasPrefix:@"BBPlayer"] && ![className hasPrefix:@"BFCPlayer"]) {
        return;
    }
    
    [self collectVideoIdentityFromCandidateAccessorsOfObject:object videoID:videoID cid:cid];
    if (*duration <= 0) {
        *duration = [self durationFromCandidateAccessorsOfObject:object];
    }

    unsigned int propertyCount = 0;
    objc_property_t *properties = class_copyPropertyList([object class], &propertyCount);
    for (unsigned int i = 0; i < propertyCount; i++) {
        const char *name = property_getName(properties[i]);
        if (!name) {
            continue;
        }
        id value = [self safeValueForKey:[NSString stringWithUTF8String:name] object:object];
        [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
    }
    free(properties);
    
    unsigned int ivarCount = 0;
    Ivar *ivars = class_copyIvarList([object class], &ivarCount);
    for (unsigned int i = 0; i < ivarCount; i++) {
        Ivar ivar = ivars[i];
        const char *type = ivar_getTypeEncoding(ivar);
        const char *name = ivar_getName(ivar);
        if (!type || type[0] != '@' || !name) {
            continue;
        }
        id value = object_getIvar(object, ivar);
        [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
    }
    free(ivars);
}

- (void)collectVideoIdentityFromCandidateAccessorsOfObject:(id)object
                                                   videoID:(NSString **)videoID
                                                       cid:(NSNumber **)cid {
    NSArray<NSString *> *videoSelectors = @[@"bvid", @"bvidStr", @"bvidString", @"bvID", @"bvId"];
    for (NSString *selectorName in videoSelectors) {
        id value = [self safeValueForSelectorName:selectorName object:object];
        if ((*videoID).length == 0 && [value isKindOfClass:[NSString class]] && [value hasPrefix:@"BV"]) {
            *videoID = value;
        }
    }
    
    id cidValue = [self safeValueForSelectorName:@"cid" object:object];
    if ((*cid).integerValue <= 0 && [cidValue respondsToSelector:@selector(integerValue)] && [cidValue integerValue] > 0) {
        *cid = @([cidValue integerValue]);
    }
}

- (id)safeValueForSelectorName:(NSString *)selectorName object:(id)object {
    SEL selector = NSSelectorFromString(selectorName);
    if (![object respondsToSelector:selector]) {
        return nil;
    }
    
    NSMethodSignature *signature = [object methodSignatureForSelector:selector];
    if (!signature || signature.numberOfArguments != 2) {
        return nil;
    }
    
    return [self safeValueForKey:selectorName object:object];
}

- (id)safeValueForKey:(NSString *)key object:(id)object {
    @try {
        return [object valueForKey:key];
    } @catch (__unused NSException *exception) {
        if ([key hasPrefix:@"_"]) {
            return nil;
        }
        @try {
            return [object valueForKey:[@"_" stringByAppendingString:key]];
        } @catch (__unused NSException *innerException) {
            return nil;
        }
    }
}

@end
