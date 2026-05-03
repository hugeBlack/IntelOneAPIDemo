//
//  NJSponsorBlockService.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>

@class NJSponsorBlockSegment;

NS_ASSUME_NONNULL_BEGIN

typedef void(^NJSponsorBlockSegmentsCompletion)(NSArray<NJSponsorBlockSegment *> *segments, NSError *_Nullable error);
typedef void(^NJSponsorBlockVoteCompletion)(BOOL success, NSError *_Nullable error);
typedef void(^NJSponsorBlockSubmitCompletion)(BOOL success, NSError *_Nullable error);

@interface NJSponsorBlockService : NSObject

+ (nullable NSString *)hashPrefixForVideoID:(NSString *)videoID;

- (void)fetchSegmentsWithVideoID:(NSString *)videoID
                             cid:(NSInteger)cid
                      categories:(NSArray<NSString *> *)categories
                      completion:(NJSponsorBlockSegmentsCompletion)completion;
- (void)reportViewedSegmentWithUUID:(NSString *)uuid;
- (void)voteForSegmentWithUUID:(NSString *)uuid
                          type:(NSInteger)type
                    completion:(nullable NJSponsorBlockVoteCompletion)completion;
- (void)submitSegmentWithVideoID:(NSString *)videoID
                             cid:(NSInteger)cid
                        category:(NSString *)category
                      actionType:(NSString *)actionType
                         segment:(NSArray<NSNumber *> *)segment
                   videoDuration:(NSTimeInterval)videoDuration
                      completion:(nullable NJSponsorBlockSubmitCompletion)completion;

@end

NS_ASSUME_NONNULL_END
