//
//  NJSponsorBlockManager.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockManager.h"
#import "NJSponsorBlockSegment.h"
#import "NJSponsorBlockService.h"
#import "NJCommonDefine.h"
#import "NJSettingCache.h"
#import <objc/runtime.h>

NSNotificationName const NJSponsorBlockStateDidChangeNotification = @"NJSponsorBlockStateDidChangeNotification";

static NSString * const NJSponsorBlockCachePrefix = @"NJSponsorBlockSegments";
static NSTimeInterval const NJSponsorBlockCacheTTL = 24 * 60 * 60;
static NSTimeInterval const NJSponsorBlockCooldown = 1.0;

static NSArray<NSString *> *NJSponsorBlockDefaultCategories(void) {
    return @[@"sponsor",
             @"intro",
             @"outro",
             @"selfpromo",
             @"interaction",
             @"preview",
             @"poi_highlight",
             @"filler",
             @"music_offtopic"];
}

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
@property (nonatomic, strong) NJSponsorBlockService *service;
@property (nonatomic, strong) NSDate *cooldownUntil;
@property (nonatomic, assign) NSTimeInterval lastProbeLogTime;
@property (nonatomic, assign) NSTimeInterval currentPlaybackTime;

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
        self.skippedUUIDs = [NSMutableSet set];
        self.service = [[NJSponsorBlockService alloc] init];
        self.cooldownUntil = [NSDate distantPast];
        self.lastProbeLogTime = -100;
    }
    return self;
}

- (void)updateVideoID:(NSString *)videoID cid:(NSInteger)cid {
    if (videoID.length == 0 || cid <= 0) {
        return;
    }
    
    self.videoID = videoID;
    self.cid = cid;
    self.segments = @[];
    [self.skippedUUIDs removeAllObjects];
    [self postStateChangedNotification];
    NSLog(@"[NJSponsorBlock] update video %@:%ld", videoID, (long)cid);
    [self loadSegmentsForCurrentVideoIfNeeded];
}

- (void)inspectModelObject:(id)object source:(NSString *)source {
    if (!NJ_SPONSOR_BLOCK_VALUE || !object) {
        return;
    }
    
    NSMutableSet<NSValue *> *visited = [NSMutableSet set];
    NSString *__block videoID = nil;
    NSNumber *__block cid = nil;
    [self collectVideoIdentityFromObject:object
                                   depth:0
                                 visited:visited
                                 videoID:&videoID
                                     cid:&cid];
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
}

- (NJSponsorBlockSegment *)activeSegmentAtPlaybackTime:(NSTimeInterval)time {
    if (!NJ_SPONSOR_BLOCK_VALUE || self.segments.count == 0) {
        return nil;
    }
    
    for (NJSponsorBlockSegment *segment in self.segments) {
        if ([self hasSkippedSegment:segment]) {
            continue;
        }
        if ([segment containsPlaybackTime:time]) {
            return segment;
        }
    }
    return nil;
}

- (void)handlePlaybackTimeForProbe:(NSTimeInterval)time {
    if (!NJ_SPONSOR_BLOCK_VALUE) {
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

- (BOOL)hasSkippedSegment:(NJSponsorBlockSegment *)segment {
    if (segment.uuid.length == 0) {
        return NO;
    }
    return [self.skippedUUIDs containsObject:segment.uuid];
}

- (BOOL)isInCooldown {
    return [[NSDate date] compare:self.cooldownUntil] == NSOrderedAscending;
}

- (void)enterCooldown {
    self.cooldownUntil = [NSDate dateWithTimeIntervalSinceNow:NJSponsorBlockCooldown];
}

- (NSTimeInterval)estimatedVideoDuration {
    NSTimeInterval duration = 0;
    for (NJSponsorBlockSegment *segment in self.segments) {
        duration = MAX(duration, segment.videoDuration);
        duration = MAX(duration, segment.endTime);
    }
    return duration;
}

- (void)loadSegmentsForCurrentVideoIfNeeded {
    if (!NJ_SPONSOR_BLOCK_VALUE || self.videoID.length == 0 || self.cid <= 0) {
        return;
    }
    
    NSArray<NJSponsorBlockSegment *> *cachedSegments = [self cachedSegmentsForVideoID:self.videoID cid:self.cid];
    if (cachedSegments) {
        self.segments = cachedSegments;
        [self postStateChangedNotification];
        return;
    }
    
    NSString *videoID = self.videoID;
    NSInteger cid = self.cid;
    __weak typeof(self) weakSelf = self;
    [self.service fetchSegmentsWithVideoID:videoID cid:cid categories:NJSponsorBlockDefaultCategories() completion:^(NSArray<NJSponsorBlockSegment *> *segments, NSError *error) {
        if (error) {
            NSLog(@"[NJSponsorBlock] fetch segments failed: %@", error);
            return;
        }
        
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf || ![strongSelf.videoID isEqualToString:videoID] || strongSelf.cid != cid) {
                return;
            }
            strongSelf.segments = segments;
            [strongSelf saveSegments:segments videoID:videoID cid:cid];
            [strongSelf postStateChangedNotification];
            NSLog(@"[NJSponsorBlock] loaded %lu segments for %@:%ld", (unsigned long)segments.count, videoID, (long)cid);
        });
    }];
}

- (NSArray<NJSponsorBlockSegment *> *)cachedSegmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid {
    NJSponsorBlockCacheItem *item = nil;
//    NJSponsorBlockCacheItem *item = (NJSponsorBlockCacheItem *)[[NJSettingCache sharedInstance].cache objectForKey:[self cacheKeyWithVideoID:videoID cid:cid]];
    if (![item isKindOfClass:[NJSponsorBlockCacheItem class]]) {
        return nil;
    }
    if (fabs([item.date timeIntervalSinceNow]) > NJSponsorBlockCacheTTL) {
        return nil;
    }
    return item.segments;
}

- (void)saveSegments:(NSArray<NJSponsorBlockSegment *> *)segments videoID:(NSString *)videoID cid:(NSInteger)cid {
    NJSponsorBlockCacheItem *item = [[NJSponsorBlockCacheItem alloc] init];
    item.segments = segments ?: @[];
    item.date = [NSDate date];
//    [[NJSettingCache sharedInstance].cache setObject:item forKey:[self cacheKeyWithVideoID:videoID cid:cid] withBlock:nil];
}

- (NSString *)cacheKeyWithVideoID:(NSString *)videoID cid:(NSInteger)cid {
    return [NSString stringWithFormat:@"%@_%@_%ld_default", NJSponsorBlockCachePrefix, videoID, (long)cid];
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

- (void)collectVideoIdentityFromObject:(id)object
                                 depth:(NSInteger)depth
                               visited:(NSMutableSet<NSValue *> *)visited
                               videoID:(NSString **)videoID
                                   cid:(NSNumber **)cid {
    if (!object || depth > 5 || ((*videoID).length > 0 && (*cid).integerValue > 0)) {
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
        for (id value in dictionary.allValues) {
            [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid];
        }
        return;
    }
    
    if ([object isKindOfClass:[NSArray class]] || [object isKindOfClass:[NSSet class]]) {
        for (id value in object) {
            [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid];
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
    
    unsigned int propertyCount = 0;
    objc_property_t *properties = class_copyPropertyList([object class], &propertyCount);
    for (unsigned int i = 0; i < propertyCount; i++) {
        const char *name = property_getName(properties[i]);
        if (!name) {
            continue;
        }
        id value = [self safeValueForKey:[NSString stringWithUTF8String:name] object:object];
        [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid];
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
        [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid];
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
