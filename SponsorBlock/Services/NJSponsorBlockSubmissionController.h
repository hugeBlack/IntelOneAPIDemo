//
//  NJSponsorBlockSubmissionController.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>
#import "NJSponsorBlockService.h"
#import "../Tweaks/Tweaks.h"

NS_ASSUME_NONNULL_BEGIN

@class NJSponsorBlockSegment;
@class NJSponsorBlockSubmissionController;

/// Context and callback protocol required by NJSponsorBlockSubmissionController.
@protocol NJSponsorBlockSubmissionDelegate <NSObject>

/// Current video identifier.
@property (nonatomic, copy, readonly) NSString *videoID;
/// Current video CID.
@property (nonatomic, assign, readonly) NSInteger cid;
/// Current player playback position in seconds.
@property (nonatomic, assign, readonly) NSTimeInterval currentPlaybackTime;
/// Estimated total video duration in seconds.
@property (nonatomic, assign, readonly) NSTimeInterval estimatedVideoDuration;
/// The player context used for presenting toasts.
@property (nonatomic, weak, readonly) BBPlayerContext *playerContext;

/// Called when the controller's observable state changes (submissionInFlight, draftInProgress, or segments added).
/// Delegate should post a state-changed notification.
- (void)submissionControllerDidChangeState:(NJSponsorBlockSubmissionController *)controller;

/// Called when locally stored unsubmitted segments are removed.
/// Delegate should clear skip-tracking state for the removed segments and post a state-changed notification.
- (void)submissionController:(NJSponsorBlockSubmissionController *)controller
    didRemoveUnsubmittedSegments:(NSArray<NJSponsorBlockSegment *> *)segments;

/// Called after all pending segments for the specified video have been successfully submitted to the server.
/// Delegate should invalidate cached segments for that video and trigger a reload.
- (void)submissionController:(NJSponsorBlockSubmissionController *)controller
    didCompleteServerSubmissionForVideoID:(NSString *)videoID
                                      cid:(NSInteger)cid
                                 segments:(NSArray<NJSponsorBlockSegment *> *)segments;

@end

/// Handles recording segment submission drafts, storing them locally,
/// and submitting them to the SponsorBlock server.
@interface NJSponsorBlockSubmissionController : NSObject

/// YES while the user is recording an end-time submission draft (toast is visible).
@property (nonatomic, assign, readonly) BOOL draftInProgress;

/// YES while a network submission request is in flight.
@property (nonatomic, assign, readonly) BOOL submissionInFlight;

- (instancetype)initWithService:(NJSponsorBlockService *)service
                       delegate:(id<NJSponsorBlockSubmissionDelegate>)delegate;

/// Begin recording an end-time submission draft for the given category.
/// Shows a persistent toast with "终点提交" / "取消" buttons.
- (void)beginDraftWithCategory:(NSString *)category;

/// Cancel an active submission draft.
- (void)cancelDraft;

/// Submit all unsubmitted segments for the current video, showing progress toasts.
- (void)submitCurrentVideoSegmentsShowingToasts;

/// Submit all unsubmitted segments for the current video.
- (void)submitSegmentsForCurrentVideoWithCompletion:(nullable NJSponsorBlockSubmitCompletion)completion;

/// Add a new unsubmitted (draft) segment for the current video.
- (nullable NJSponsorBlockSegment *)addSegmentWithCategory:(NSString *)category
                                                actionType:(NSString *)actionType
                                                   segment:(NSArray<NSNumber *> *)segment;

/// Returns all locally stored unsubmitted segments for the current video.
- (NSArray<NJSponsorBlockSegment *> *)segmentsForCurrentVideo;

/// Remove all locally stored unsubmitted segments for the current video.
- (void)clearSegmentsForCurrentVideo;

/// Remove all locally stored unsubmitted segments for all videos.
- (void)clearAllSegments;

/// Show a brief auto-dismissing info toast.
- (void)showInfoToast:(NSString *)title detail:(NSString *)detail;

@end

NSString *NJSBFormatTime(NSTimeInterval time);
NSString *NJSBFormatCompact(NSTimeInterval time);

NS_ASSUME_NONNULL_END
