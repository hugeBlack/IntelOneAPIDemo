//
//  NJSponsorBlockService.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>

@class NJSponsorBlockSegment;

NS_ASSUME_NONNULL_BEGIN

typedef void(^NJSponsorBlockSegmentsCompletion)(NSArray<NJSponsorBlockSegment *> *segments, NSError *_Nullable error);

@interface NJSponsorBlockService : NSObject

- (void)fetchSegmentsWithVideoID:(NSString *)videoID
                             cid:(NSInteger)cid
                      categories:(NSArray<NSString *> *)categories
                      completion:(NJSponsorBlockSegmentsCompletion)completion;

@end

NS_ASSUME_NONNULL_END
