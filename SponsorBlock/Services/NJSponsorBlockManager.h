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

/// YES while the user is recording an end-time submission draft (toast is visible).
@property (nonatomic, assign, readonly) BOOL submissionDraftInProgress;

/// YES while a network submission request is in flight.
@property (nonatomic, assign, readonly) BOOL isSubmissionInFlight;

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

/// Begin recording an end-time submission draft for the given category.
/// Shows a persistent toast with "终点提交" / "取消" buttons.
- (void)beginSubmissionDraftWithCategory:(NSString *)category;

/// Cancel an active submission draft and dismiss its toast.
- (void)cancelSubmissionDraft;

/// Submit all unsubmitted segments for the current video, showing progress toasts.
- (void)submitCurrentVideoDraftsShowingToasts;

/// Show a brief auto-dismissing info toast with no close button.
- (void)showInfoToast:(NSString *)title detail:(NSString *)detail;

- (void)reset;
@end

NS_ASSUME_NONNULL_END
