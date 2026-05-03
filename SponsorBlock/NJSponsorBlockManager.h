//
//  NJSponsorBlockManager.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>

@class NJSponsorBlockSegment;

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockStateDidChangeNotification;

@interface NJSponsorBlockManager : NSObject

@property (nonatomic, copy, readonly) NSString *videoID;
@property (nonatomic, assign, readonly) NSInteger cid;
@property (nonatomic, strong, readonly) NSArray<NJSponsorBlockSegment *> *segments;
@property (nonatomic, assign, readonly) NSTimeInterval currentPlaybackTime;
@property (nonatomic, assign, readonly) NSTimeInterval estimatedVideoDuration;

+ (instancetype)sharedInstance;

- (void)updateVideoID:(NSString *)videoID cid:(NSInteger)cid;
- (void)inspectModelObject:(id)object source:(NSString *)source;
- (nullable NJSponsorBlockSegment *)activeSegmentAtPlaybackTime:(NSTimeInterval)time;
- (void)handlePlaybackTimeForProbe:(NSTimeInterval)time;
- (void)markSegmentSkipped:(NJSponsorBlockSegment *)segment;
- (BOOL)hasSkippedSegment:(NJSponsorBlockSegment *)segment;
- (BOOL)isInCooldown;
- (void)enterCooldown;

@end

NS_ASSUME_NONNULL_END
