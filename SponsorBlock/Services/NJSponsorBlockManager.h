//
//  NJSponsorBlockManager.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>
#import "NJSponsorBlockService.h"
#import "../Tweaks/Tweaks.h"

@class NJSponsorBlockSegment;
@class NJSponsorBlockPanelView;

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockStateDidChangeNotification;
FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockPlaybackTimeDidChangeNotification;

FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockVideoInfoRetrievedNotification;

@interface NJSponsorBlockManager : NSObject

@property (nonatomic, copy, readonly) NSString *videoID;
@property (nonatomic, assign, readonly) NSInteger cid;
@property (nonatomic, strong, readonly) NSArray<NJSponsorBlockSegment *> *segments;
@property (nonatomic, assign, readonly) NSTimeInterval currentPlaybackTime;
@property (nonatomic, assign, readonly) NSTimeInterval estimatedVideoDuration;

@property (nonatomic, weak) BBPlayerContext* playerContext;
@property NJSponsorBlockPanelView* panelView;

- (instancetype)initWithContext:(BBPlayerContext*)playerContext;

- (void)updateVideoID:(NSString *)videoID cid:(NSInteger)cid;

- (NSArray<NJSponsorBlockSegment *> *)allSegments;
- (NSArray<NJSponsorBlockSegment *> *)displaySegments;
- (nullable NJSponsorBlockSegment *)activeSegmentAtPlaybackTime:(NSTimeInterval)time;
- (NSArray<NJSponsorBlockSegment *> *)autoSkipSegmentsAtPlaybackTime:(NSTimeInterval)time;
- (nullable NJSponsorBlockSegment *)manualSkipSegmentAtPlaybackTime:(NSTimeInterval)time;
- (nullable NJSponsorBlockSegment *)upcomingAutoSkipSegmentAtPlaybackTime:(NSTimeInterval)time withinSeconds:(NSTimeInterval)seconds;
- (NSTimeInterval)skippedDurationBeforePlaybackTime:(NSTimeInterval)time;
- (NSTimeInterval)playbackTimeWithoutSkippedSegments:(NSTimeInterval)time;
- (nullable NJSponsorBlockSegment *)lastSkippedSegment;
- (BOOL)skipOnSeekToSegment;
- (void)handlePlaybackTimeForProbe:(NSTimeInterval)time;
- (void)markSegmentSkipped:(NJSponsorBlockSegment *)segment;
- (void)clearSkippedSegment:(NJSponsorBlockSegment *)segment;
- (void)recordLastSkippedSegment:(NJSponsorBlockSegment *)segment;
- (void)reportSegmentSkipped:(NJSponsorBlockSegment *)segment;
- (void)submitUnsubmittedSegmentsForCurrentVideoWithCompletion:(nullable NJSponsorBlockSubmitCompletion)completion;
- (nullable NJSponsorBlockSegment *)addUnsubmittedSegmentWithCategory:(NSString *)category
                                                           actionType:(NSString *)actionType
                                                              segment:(NSArray<NSNumber *> *)segment;
- (NSArray<NJSponsorBlockSegment *> *)unsubmittedSegmentsForCurrentVideo;
- (void)clearUnsubmittedSegmentsForCurrentVideo;
- (void)clearAllUnsubmittedSegments;
- (BOOL)hasActuallySkippedSegment:(NJSponsorBlockSegment *)segment;
- (BOOL)isInCooldown;
- (void)enterCooldown;
- (void)clearAllCachedSegments;

- (void)skipSegment:(NJSponsorBlockSegment*)segment;
- (void)seekTo:(NSTimeInterval)dest;
- (void)reset;
@end

NS_ASSUME_NONNULL_END
