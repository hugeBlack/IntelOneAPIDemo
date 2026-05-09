//
//  NJSponsorBlockThumbnailLabelService.h
//  SponsorBlock
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^NJSBThumbnailLabelCompletion)(NSString *_Nullable category, NSString *_Nullable text);

@interface NJSponsorBlockThumbnailLabelService : NSObject

+ (instancetype)sharedService;
- (void)fetchLabelForBVID:(NSString *)bvid completion:(NJSBThumbnailLabelCompletion)completion;

@end

NS_ASSUME_NONNULL_END
