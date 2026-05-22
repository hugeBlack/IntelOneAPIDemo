//
//  NJSponsorBlockManager.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockManager.h"
#import "../Models/NJSponsorBlockSegment.h"
#import "NJSponsorBlockService.h"
#import "../Settings/NJSponsorBlockSettings.h"
#import "../Models/NJSponsorBlockCacheStats.h"
#import "NJSponsorBlockUnsubmittedSegmentStore.h"
#import "../Settings/NJCommonDefine.h"
#import "../Tweaks/RuntimeClasses/SponsorBlockHintToast.h"
#import "../Tweaks/Tweaks.h"
#import <math.h>
#import <objc/runtime.h>
#import "NJSponsorBlockSubmissionController.h"

NSNotificationName const NJSponsorBlockStateDidChangeNotification = @"NJSponsorBlockStateDidChangeNotification";
NSNotificationName const NJSponsorBlockPlaybackTimeDidChangeNotification = @"NJSponsorBlockPlaybackTimeDidChangeNotification";
NSNotificationName const NJSponsorBlockManualSkipRequestNotification = @"NJSponsorBlockManualSkipRequestNotification";
NSNotificationName const NJSponsorBlockSeekRequestNotification = @"NJSponsorBlockSeekRequestNotification";
NSNotificationName const NJSponsorBlockVideoInfoRetrievedNotification = @"NJSponsorBlockVideoInfoRetrievedNotification";

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

@interface NJSponsorBlockManager () <NJSponsorBlockSubmissionDelegate>

@property (nonatomic, copy) NSString *videoID;
@property (nonatomic, assign) NSInteger cid;
@property (nonatomic, strong) NSArray<NJSponsorBlockSegment *> *segments;
@property (nonatomic, strong) NSMutableSet<NSString *> *skippedUUIDs;
@property (nonatomic, strong) NSMutableSet<NSString *> *actualSkippedUUIDs;
@property (nonatomic, strong) NJSponsorBlockService *service;
@property (nonatomic, strong) NSDate *cooldownUntil;
@property (nonatomic, strong) NJSponsorBlockSegment *lastSkippedSegment;
@property (nonatomic, assign) NSTimeInterval lastProbeLogTime;
@property (nonatomic, assign) NSTimeInterval nativeVideoDuration;
@property (nonatomic, copy) NSString *loadedServerBaseURLString;
@property (nonatomic, strong) NSMutableSet<NSString *> *trackedCacheKeys;

@property NSTimer* playbackPollTimer;
@property NSTimeInterval NJSponsorBlockLastPlaybackPosition;

@property (nonatomic, strong) NJSponsorBlockSubmissionController *submissionController;

// Advance-notice dedup: UUID of the segment for which we already showed the toast
@property (nonatomic, copy) NSString *advanceNoticeShownForSegmentUUID;

- (void)updateVideoID:(NSString *)videoID cid:(NSInteger)cid duration:(NSTimeInterval)duration;

- (void)invalidateCachedSegmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid;
- (NSUInteger)estimatedSizeForSegments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (void)clearSkipStateForSegments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (BOOL)hasSkippedSegment:(NJSponsorBlockSegment *)segment;
- (void)postStateChangedNotification;
- (void)postPlaybackTimeChangedNotification;
- (void)checkAndShowAdvanceNoticeAtPlaybackTime:(NSTimeInterval)time;
- (void)showAdvanceNoticeToastForSegment:(NJSponsorBlockSegment *)segment remaining:(NSTimeInterval)remaining;
- (void)showSkippedNoticeToastForSegment:(NJSponsorBlockSegment *)segment;

@end

@implementation NJSponsorBlockManager

- (instancetype)initWithContext:(BBPlayerContext*)playerContext {
    self = [super init];
    if (!self) {
        return nil;
    }
    self.playerContext = playerContext;
    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(refresh)
                                                 name:NJSponsorBlockSettingsDidChangeNotification
                                               object:nil];
    [self reset];
    return self;
}

- (void)startListeningForVideoInfoWithCID:(NSInteger)cid {
    self.cid = cid;

    NSDictionary* videoInfo = cachedCidVideoInfoDict[@(cid)];
    if(videoInfo) {
        [self handleVideoInfoRetrieved:nil];
    } else {
        [NSNotificationCenter.defaultCenter addObserver:self selector:@selector(handleVideoInfoRetrieved:) name:NJSponsorBlockVideoInfoRetrievedNotification object:nil];
    }
    
}

- (void)handleVideoInfoRetrieved:(NSNotification*)notification {
    @synchronized (cachedCidVideoInfoDict) {
        NSDictionary* videoInfo = cachedCidVideoInfoDict[@(self.cid)];
        if(videoInfo) {
            NSString *videoID = videoInfo[@"videoID"];
            NSNumber *cid = videoInfo[@"cid"];
            NSNumber *duration = videoInfo[@"duration"];
            [self updateVideoID:videoID cid:cid.integerValue duration:duration.longLongValue];
            [NSNotificationCenter.defaultCenter removeObserver:self name:NJSponsorBlockVideoInfoRetrievedNotification object:nil];
        }
    }
}

- (void)updateVideoID:(NSString *)videoID cid:(NSInteger)cid duration:(NSTimeInterval)duration {
    if (videoID.length == 0 || cid <= 0) {
        return;
    }
    if ([self.videoID isEqualToString:videoID] && self.cid == cid) {
        return;
    }
    
    if (duration <= 0 || !isfinite(duration)) {
        return;
    }
    if (fabs(self.nativeVideoDuration - duration) < 0.5) {
        return;
    }

    self.videoID = videoID;
    self.cid = cid;
    self.segments = @[];
    self.loadedServerBaseURLString = @"";
    self.nativeVideoDuration = duration;
    self.lastSkippedSegment = nil;
    self.advanceNoticeShownForSegmentUUID = nil;
    [self.skippedUUIDs removeAllObjects];
    [self.actualSkippedUUIDs removeAllObjects];
    [self postStateChangedNotification];
    NSLog(@"[NJSponsorBlock] update video %@:%ld", videoID, (long)cid);
    [self loadSegmentsForCurrentVideoIfNeeded];
}

- (void)startPollingPlaybackTime {
    if(_playbackPollTimer && [_playbackPollTimer isValid]) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        __weak typeof(self) weakSelf = self;
        self->_playbackPollTimer = [NSTimer scheduledTimerWithTimeInterval:0.5 repeats:YES block:^(__unused NSTimer *timer) {
            NSTimeInterval time = weakSelf.playerContext.playback.currentTime;
            [weakSelf handlePlaybackTime:time];
        }];
    });

}

- (NSArray<NJSponsorBlockSegment *> *)allSegments {
    NSArray<NJSponsorBlockSegment *> *localSegments = [self.submissionController segmentsForCurrentVideo];
    if (self.segments.count == 0) {
        return localSegments;
    }
    if (localSegments.count == 0) {
        return self.segments;
    }

    NSMutableArray<NJSponsorBlockSegment *> *segments = [self.segments mutableCopy];
    [segments addObjectsFromArray:localSegments];
    return [segments copy];
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

- (void)handlePlaybackTimeForProbe:(NSTimeInterval)time {
    [self postPlaybackTimeChangedNotification];

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
    if (segment.uuid.length == 0 || segment.isUnsubmitted || ![NJSponsorBlockSettings skipTrackingEnabled]) {
        return;
    }
    if (segment.actionType.length == 0 || [segment.actionType isEqualToString:@"skip"]) {
        [self.service reportViewedSegmentWithUUID:segment.uuid];
    }
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

- (NSTimeInterval)estimatedVideoDuration {
    NSTimeInterval duration = self.nativeVideoDuration;
    for (NJSponsorBlockSegment *segment in [self allSegments]) {
        duration = MAX(duration, segment.videoDuration);
        duration = MAX(duration, segment.endTime);
    }
    return duration;
}

- (void)loadSegmentsForCurrentVideoIfNeeded {
    if (self.videoID.length == 0 || self.cid <= 0) {
        return;
    }

    NSString *serverBaseURLString = [NJSponsorBlockSettings serverBaseURLString];
    NSArray<NJSponsorBlockSegment *> *cachedSegments = [self cachedSegmentsForVideoID:self.videoID cid:self.cid];
    if (cachedSegments) {
        self.segments = cachedSegments;
        self.loadedServerBaseURLString = serverBaseURLString;
        [self postStateChangedNotification];
        if([cachedSegments count] > 0) {
            [self startPollingPlaybackTime];
        }
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
            if([segments count] > 0) {
                [strongSelf startPollingPlaybackTime];
            }
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

- (void)clearSkipStateForSegments:(NSArray<NJSponsorBlockSegment *> *)segments {
    for (NJSponsorBlockSegment *segment in segments) {
        if (segment.uuid.length == 0) {
            continue;
        }
        [self.skippedUUIDs removeObject:segment.uuid];
        [self.actualSkippedUUIDs removeObject:segment.uuid];
        if ([self.lastSkippedSegment.uuid isEqualToString:segment.uuid]) {
            self.lastSkippedSegment = nil;
        }
    }
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

- (void)refresh {
    BOOL shouldReload = YES;
    if (shouldReload) {
        self.segments = @[];
        self.loadedServerBaseURLString = @"";
        self.lastSkippedSegment = nil;
        [self.skippedUUIDs removeAllObjects];
        [self.actualSkippedUUIDs removeAllObjects];
    }
    [self postStateChangedNotification];
    if (shouldReload) {
        [self invalidateCachedSegmentsForVideoID:_videoID cid:_cid];
        [self loadSegmentsForCurrentVideoIfNeeded];
    }
}

- (void)postStateChangedNotification {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockStateDidChangeNotification object:self];
    });
}

- (void)postPlaybackTimeChangedNotification {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockPlaybackTimeDidChangeNotification object:self];
    });
}

- (void)handlePlaybackTime:(NSTimeInterval) position {
    NJSponsorBlockManager *manager = self;

    BOOL movedBySeek = _NJSponsorBlockLastPlaybackPosition >= 0 && fabs(position - _NJSponsorBlockLastPlaybackPosition) > 2.0;
    _NJSponsorBlockLastPlaybackPosition = position;

    [manager handlePlaybackTimeForProbe:position];

    if ([NJSponsorBlockSettings showAutoSkipToast]) {
        // Show "即将自动跳过" toast for upcoming segments
        [self checkAndShowAdvanceNoticeAtPlaybackTime:position];
    }

    if ([manager isInCooldown]) {
        return;
    }

    NSArray<NJSponsorBlockSegment *> *segments = [manager autoSkipSegmentsAtPlaybackTime:position];
    NJSponsorBlockSegment *segment = segments.lastObject;
    if (!segment) {
        return;
    }

    if (movedBySeek && ![NJSponsorBlockSettings skipOnSeekToSegment]) {
        NSLog(@"[NJSponsorBlock] ignore seek into segment %@ %.2f-%.2f", segment.uuid, segment.startTime, segment.endTime);
        for (NJSponsorBlockSegment *skippedSegment in segments) {
            [manager markSegmentSkipped:skippedSegment];
        }
        return;
    }

    NSTimeInterval targetTime = segment.endTime;
    if (segment.videoDuration > 0 && targetTime > segment.videoDuration - 2.0) {
        targetTime -= 2.0;
    }
    
    [self skipSegment:segment];
}

- (void)skipSegment:(NJSponsorBlockSegment*)segment {
    if (!segment) {
        return;
    }

    NSTimeInterval targetTime = [segment.actionType isEqualToString:@"poi"] ? segment.startTime : segment.endTime;
    [_playerContext.playback seekTo:segment.endTime];
    [self markSegmentSkipped:segment];
    [self reportSegmentSkipped:segment];
    [self recordLastSkippedSegment:segment];
    [self enterCooldown];
    NSLog(@"[NJSponsorBlock] manually skipped %@ %.2f-%.2f target=%.2f", segment.uuid, segment.startTime, segment.endTime, targetTime);

    if ([NJSponsorBlockSettings showSkipUndoToast]) {
        // Show "已跳过片段 / 撤销" toast
        [self showSkippedNoticeToastForSegment:segment];
    }

}

- (void)seekTo:(NSTimeInterval)dest {
    [_playerContext.playback seekTo:dest];
}

- (NSTimeInterval)currentPlaybackTime {
    return self.playerContext.playback.currentTime;
}

// ── Toast trigger methods ─────────────────────────────────────────────────

- (void)checkAndShowAdvanceNoticeAtPlaybackTime:(NSTimeInterval)time {
    if (![NJSponsorBlockSettings enabled]) return;
    NSTimeInterval advanceSeconds = [NJSponsorBlockSettings advanceNoticeDuration];
    if (advanceSeconds <= 0) return;

    NJSponsorBlockSegment *upcoming = [self upcomingAutoSkipSegmentAtPlaybackTime:time withinSeconds:advanceSeconds];
    if (!upcoming) return;
    if ([self.advanceNoticeShownForSegmentUUID isEqualToString:upcoming.uuid]) return;

    self.advanceNoticeShownForSegmentUUID = upcoming.uuid;
    NSTimeInterval remaining = MAX(0.0, upcoming.startTime - time);
    [self showAdvanceNoticeToastForSegment:upcoming remaining:remaining];
}

- (void)showAdvanceNoticeToastForSegment:(NJSponsorBlockSegment *)segment remaining:(NSTimeInterval)remaining {
    if (!_playerContext) return;

    NSString *detail = [NSString stringWithFormat:@"%@ · 还有 %@",
                        [NJSponsorBlockSettings titleForCategory:segment.category],
                        NJSBFormatCompact(remaining)];

    __weak typeof(self) weakSelf = self;
    id toast = NJSponsorBlockCreateHintToast(
        _playerContext,
        @"即将自动跳过",
        detail,
        @"立即",      // primary – skip immediately
        @"本次不跳",  // secondary – mark as skipped (suppress this session)
        ^{            // primary handler
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) return;
            [strongSelf skipSegment:segment];
        },
        ^{            // secondary handler
            __strong typeof(weakSelf) weakSelf2 = weakSelf;
            __strong typeof(weakSelf) strongSelf = weakSelf2;
            if (!strongSelf) return;
            [strongSelf markSegmentSkipped:segment];
        },
        nil,          // close (×) – just dismiss
        remaining - 0.5,
        YES
    );
    if (toast) {
        [_playerContext.toastWidgetService presentCustomToast:toast];
    }
}

- (void)showSkippedNoticeToastForSegment:(NJSponsorBlockSegment *)segment {
    if (!_playerContext) return;

    NSString *detail = [NSString stringWithFormat:@"%@-%@ · %@",
                        NJSBFormatTime(segment.startTime),
                        NJSBFormatTime(segment.endTime),
                        [NJSponsorBlockSettings titleForCategory:segment.category]];

    __weak typeof(self) weakSelf = self;
    id toast = NJSponsorBlockCreateHintToast(
        _playerContext,
        @"已跳过片段",
        detail,
        @"撤销",  // primary – undo the skip
        nil,
        ^{        // primary handler
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) return;
            [strongSelf seekTo:segment.startTime];
        },
        nil,
        nil,
        5.0,
        YES
    );
    if (toast) {
        [_playerContext.toastWidgetService presentCustomToast:toast];
    }
}

// ── NJSponsorBlockSubmissionDelegate ────────────────────────────────────

- (void)submissionControllerDidChangeState:(NJSponsorBlockSubmissionController *)controller {
    [self postStateChangedNotification];
    if([self.segments count] > 0) {
        [self startPollingPlaybackTime];
    }
}

- (void)submissionController:(NJSponsorBlockSubmissionController *)controller
    didRemoveUnsubmittedSegments:(NSArray<NJSponsorBlockSegment *> *)segments {
    [self clearSkipStateForSegments:segments];
    [self postStateChangedNotification];
    if([self.segments count] > 0) {
        [self startPollingPlaybackTime];
    }
}

- (void)submissionController:(NJSponsorBlockSubmissionController *)controller
    didCompleteServerSubmissionForVideoID:(NSString *)videoID
                                      cid:(NSInteger)cid
                                 segments:(NSArray<NJSponsorBlockSegment *> *)segments {
    [self clearSkipStateForSegments:segments];
    [self invalidateCachedSegmentsForVideoID:videoID cid:cid];
    self.segments = @[];
    [self postStateChangedNotification];
    [self loadSegmentsForCurrentVideoIfNeeded];
}

- (void)dealloc {
    [NSNotificationCenter.defaultCenter removeObserver:self];
    [self.playbackPollTimer invalidate];
}

- (void)reset {
    self.videoID = @"";
    self.segments = @[];
    self.loadedServerBaseURLString = @"";
    self.skippedUUIDs = [NSMutableSet set];
    self.actualSkippedUUIDs = [NSMutableSet set];
    self.service = [[NJSponsorBlockService alloc] init];
    self.submissionController = [[NJSponsorBlockSubmissionController alloc] initWithService:self.service delegate:self];
    self.cooldownUntil = [NSDate distantPast];
    self.lastProbeLogTime = -100;
    self.trackedCacheKeys = [NSMutableSet set];
    [self postStateChangedNotification];
}

@end
