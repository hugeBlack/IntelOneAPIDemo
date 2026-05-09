//
//  NJSponsorBlockSettings.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

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

+ (BOOL)showSegmentsInSeekbarWidget;
+ (void)setShowSegmentsInSeekbarWidget:(BOOL)enabled;
+ (BOOL)showSegmentsInProgressWidget;
+ (void)setShowSegmentsInProgressWidget:(BOOL)enabled;
+ (BOOL)showSharedEntryButton;
+ (void)setShowSharedEntryButton:(BOOL)enabled;
+ (BOOL)showVideoLabels;
+ (void)setShowVideoLabels:(BOOL)enabled;

+ (NSArray<NJSponsorBlockCategoryOption *> *)categoryOptions;
+ (NSArray<NJSponsorBlockCategoryOption *> *)thumbnailBadgeLabelOptions;
+ (NSArray<NSString *> *)requestCategories;
+ (NJSponsorBlockCategoryAction)actionForCategory:(NSString *)category;
+ (void)setAction:(NJSponsorBlockCategoryAction)action forCategory:(NSString *)category;
+ (NSString *)titleForAction:(NJSponsorBlockCategoryAction)action;
+ (NSString *)titleForCategory:(NSString *)category;
+ (BOOL)shouldShowSegment:(NJSponsorBlockSegment *)segment;
+ (BOOL)shouldAutoSkipSegment:(NJSponsorBlockSegment *)segment;
+ (BOOL)shouldManualSkipSegment:(NJSponsorBlockSegment *)segment;
+ (UIColor *)colorForCategory:(NSString *)category;
+ (void)setColor:(UIColor *)color forCategory:(NSString *)category;
+ (UIColor *)defaultColorForCategory:(NSString *)category;
+ (void)resetColors;

+ (NSString *)sponsorBlockUserID;
+ (void)setSponsorBlockUserID:(NSString *)userID;
+ (BOOL)skipTrackingEnabled;
+ (void)setSkipTrackingEnabled:(BOOL)enabled;

+ (NSDictionary<NSString *, id> *)exportSettings;
+ (BOOL)importSettings:(NSDictionary<NSString *, id> *)settings;
+ (void)resetToDefaults;

+ (NSString *)requestConfigurationIdentifier;
+ (void)postSettingsDidChangeNotification;

#pragma mark - Thumbnail Badge Color

+ (UIColor *)thumbnailBadgeColorForLabel:(NSString *)label;
+ (UIColor *)defaultThumbnailBadgeColorForLabel:(NSString *)label;
+ (void)setThumbnailBadgeColor:(UIColor *)color forLabel:(NSString *)label;
+ (void)resetThumbnailBadgeColors;

@end

FOUNDATION_EXPORT NSNotificationName const NJSponsorBlockSettingsDidChangeNotification;

NS_ASSUME_NONNULL_END
