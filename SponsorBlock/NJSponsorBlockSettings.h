//
//  NJSponsorBlockSettings.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>

@class NJSponsorBlockSegment;

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, NJSponsorBlockCategoryAction) {
    NJSponsorBlockCategoryActionDisabled = -1,
    NJSponsorBlockCategoryActionShowOverlay = 0,
    NJSponsorBlockCategoryActionManualSkip = 1,
    NJSponsorBlockCategoryActionAutoSkip = 2,
};

@interface NJSponsorBlockCategoryOption : NSObject

@property (nonatomic, copy, readonly) NSString *category;
@property (nonatomic, copy, readonly) NSString *title;

- (instancetype)initWithCategory:(NSString *)category title:(NSString *)title;

@end

@interface NJSponsorBlockSettings : NSObject

+ (BOOL)enabled;
+ (void)setEnabled:(BOOL)enabled;
+ (BOOL)cacheEnabled;
+ (void)setCacheEnabled:(BOOL)enabled;
+ (BOOL)skipOnSeekToSegment;
+ (void)setSkipOnSeekToSegment:(BOOL)enabled;
+ (BOOL)testingServerEnabled;
+ (void)setTestingServerEnabled:(BOOL)enabled;
+ (NSTimeInterval)minDuration;
+ (void)setMinDuration:(NSTimeInterval)duration;
+ (NSTimeInterval)advanceNoticeDuration;
+ (void)setAdvanceNoticeDuration:(NSTimeInterval)duration;
+ (NSString *)serverBaseURLString;
+ (void)setServerBaseURLString:(NSString *)serverBaseURLString;

+ (NSArray<NJSponsorBlockCategoryOption *> *)categoryOptions;
+ (NSArray<NSString *> *)requestCategories;
+ (NJSponsorBlockCategoryAction)actionForCategory:(NSString *)category;
+ (void)setAction:(NJSponsorBlockCategoryAction)action forCategory:(NSString *)category;
+ (NSString *)titleForAction:(NJSponsorBlockCategoryAction)action;
+ (NSString *)titleForCategory:(NSString *)category;
+ (BOOL)shouldShowSegment:(NJSponsorBlockSegment *)segment;
+ (BOOL)shouldAutoSkipSegment:(NJSponsorBlockSegment *)segment;
+ (BOOL)shouldManualSkipSegment:(NJSponsorBlockSegment *)segment;
+ (NSString *)requestConfigurationIdentifier;
+ (void)postSettingsDidChangeNotification;

@end

FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockSettingsDidChangeNotification;

NS_ASSUME_NONNULL_END
