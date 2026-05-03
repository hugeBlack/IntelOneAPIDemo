//
//  NJSponsorBlockCacheStats.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockCacheStats : NSObject

@property (nonatomic, assign) NSUInteger totalItems;
@property (nonatomic, assign) NSUInteger totalSizeBytes;
@property (nonatomic, assign) NSUInteger dailyHits;
@property (nonatomic, assign) NSUInteger dailySizeBytes;

+ (instancetype)sharedInstance;

- (void)recordHitWithSize:(NSUInteger)size;
- (void)recordSaveWithSize:(NSUInteger)size;
- (void)recordRemoval;
- (void)clearAll;
- (void)refreshFromCache;
- (NSDictionary<NSString *, id> *)exportStats;

@end

NS_ASSUME_NONNULL_END
