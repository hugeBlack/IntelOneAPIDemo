//
//  NJSponsorBlockManager.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>
#import "NJSponsorBlockService.h"

@class NJSponsorBlockSegment;

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockStateDidChangeNotification;
FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockManualSkipRequestNotification;
FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockSeekRequestNotification;

@interface NJSponsorBlockManager : NSObject

@property (nonatomic, copy, readonly) NSString *videoID;
@property (nonatomic, assign, readonly) NSInteger cid;
@property (nonatomic, strong, readonly) NSArray<NJSponsorBlockSegment *> *segments;
@property (nonatomic, assign, readonly) NSTimeInterval currentPlaybackTime;
@property (nonatomic, assign, readonly) NSTimeInterval estimatedVideoDuration;

+ (instancetype)sharedInstance;

- (void)updateVideoID:(NSString *)videoID cid:(NSInteger)cid;
- (void)inspectResponseData:(NSData *)data response:(NSURLResponse *)response;
- (void)inspectModelObject:(id)object source:(NSString *)source;
- (NSArray<NJSponsorBlockSegment *> *)allSegments;
- (NSArray<NJSponsorBlockSegment *> *)displaySegments;
- (nullable NJSponsorBlockSegment *)activeSegmentAtPlaybackTime:(NSTimeInterval)time;
- (nullable NJSponsorBlockSegment *)autoSkipSegmentAtPlaybackTime:(NSTimeInterval)time;
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
- (void)submitSegmentWithCategory:(NSString *)category
                       actionType:(NSString *)actionType
                          segment:(NSArray<NSNumber *> *)segment
                       completion:(nullable NJSponsorBlockSubmitCompletion)completion;
- (BOOL)hasSkippedSegment:(NJSponsorBlockSegment *)segment;
- (BOOL)hasActuallySkippedSegment:(NJSponsorBlockSegment *)segment;
- (BOOL)isInCooldown;
- (void)enterCooldown;

@end

NS_ASSUME_NONNULL_END
