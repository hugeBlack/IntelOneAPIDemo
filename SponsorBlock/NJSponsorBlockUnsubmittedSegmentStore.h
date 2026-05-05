//
//  NJSponsorBlockUnsubmittedSegmentStore.h
//  SponsorBlock
//

#import <Foundation/Foundation.h>

@class NJSponsorBlockSegment;

NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockUnsubmittedSegmentStore : NSObject

+ (instancetype)sharedStore;

- (NSString *)keyForVideoID:(NSString *)videoID cid:(NSInteger)cid;
- (NSArray<NSString *> *)videoKeys;
- (NSArray<NJSponsorBlockSegment *> *)segmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid;
- (NSArray<NJSponsorBlockSegment *> *)segmentsForVideoKey:(NSString *)videoKey;
- (NSArray<NJSponsorBlockSegment *> *)allSegments;
- (nullable NJSponsorBlockSegment *)addSegmentForVideoID:(NSString *)videoID
                                                     cid:(NSInteger)cid
                                                category:(NSString *)category
                                              actionType:(NSString *)actionType
                                                 segment:(NSArray<NSNumber *> *)segment
                                           videoDuration:(NSTimeInterval)videoDuration;
- (BOOL)updateSegmentWithUUID:(NSString *)uuid
                      videoID:(NSString *)videoID
                          cid:(NSInteger)cid
                    startTime:(NSTimeInterval)startTime
                      endTime:(NSTimeInterval)endTime;
- (BOOL)removeSegmentWithUUID:(NSString *)uuid videoID:(NSString *)videoID cid:(NSInteger)cid;
- (void)removeSegmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid;
- (void)clearAllSegments;
- (NSUInteger)totalSegmentCount;
- (NSUInteger)videoCount;

@end

NS_ASSUME_NONNULL_END
