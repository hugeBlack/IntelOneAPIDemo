//
//  NJSponsorBlockSettings.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockSettings.h"
#import "../Models/NJSponsorBlockSegment.h"
#import "NJCommonDefine.h"

NSNotificationName const NJSponsorBlockSettingsDidChangeNotification = @"NJSponsorBlockSettingsDidChangeNotification";

static NSString * const NJSponsorBlockEnabledKey = @"NJSponsorBlockEnabledKey";
static NSString * const NJSponsorBlockCacheEnabledKey = @"NJSponsorBlockCacheEnabledKey";
static NSString * const NJSponsorBlockSkipOnSeekKey = @"NJSponsorBlockSkipOnSeekKey";
static NSString * const NJSponsorBlockTestingServerKey = @"NJSponsorBlockTestingServerKey";
static NSString * const NJSponsorBlockMinDurationKey = @"NJSponsorBlockMinDurationKey";
static NSString * const NJSponsorBlockAdvanceNoticeDurationKey = @"NJSponsorBlockAdvanceNoticeDurationKey";
static NSString * const NJSponsorBlockServerBaseURLKey = @"NJSponsorBlockServerBaseURLKey";
static NSString * const NJSponsorBlockCategoryActionsKey = @"NJSponsorBlockCategoryActionsKey";
static NSString * const NJSponsorBlockCategoryColorsKey = @"NJSponsorBlockCategoryColorsKey";
static NSString * const NJSponsorBlockUserIDKey = @"NJSponsorBlockVoteUserIDKey";
static NSString * const NJSponsorBlockSkipTrackingEnabledKey = @"NJSponsorBlockSkipTrackingEnabledKey";
static NSString * const NJSponsorBlockShowSegmentsInSeekbarWidgetKey = @"NJSponsorBlockShowSegmentsInSeekbarWidgetKey";
static NSString * const NJSponsorBlockShowSegmentsInProgressWidgetKey = @"NJSponsorBlockShowSegmentsInProgressWidgetKey";
static NSString * const NJSponsorBlockShowSharedEntryButtonKey = @"NJSponsorBlockShowSharedEntryButtonKey";
static NSString * const NJSponsorBlockShowVideoLabelsKey = @"NJSponsorBlockShowVideoLabelsKey";
static NSString * const NJSponsorBlockShowAutoSkipToastKey = @"NJSponsorBlockShowAutoSkipToastKey";
static NSString * const NJSponsorBlockShowSkipUndoToastKey = @"NJSponsorBlockShowSkipUndoToastKey";
static NSString * const NJSponsorBlockThumbnailBadgeColorsKey = @"NJSponsorBlockThumbnailBadgeColorsKey";

static NSString * const NJSponsorBlockDefaultServerBaseURLString = @"https://bsbsb.top";
static NSString * const NJSponsorBlockTestingServerBaseURLString = @"http://127.0.0.1:9876";

@interface NJSponsorBlockSettings ()

+ (NSDictionary<NSString *, NSNumber *> *)categoryActions;
+ (NJSponsorBlockCategoryAction)defaultActionForCategory:(NSString *)category;
+ (BOOL)segmentUsesSeekAction:(NJSponsorBlockSegment *)segment;
+ (BOOL)segmentPassesDurationFilter:(NJSponsorBlockSegment *)segment;

@end

@implementation NJSponsorBlockCategoryOption

- (instancetype)initWithCategory:(NSString *)category title:(NSString *)title {
    self = [super init];
    if (self) {
        _category = [category copy];
        _title = [title copy];
    }
    return self;
}

@end

@implementation NJSponsorBlockSettings

+ (BOOL)enabled {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockEnabledKey];
    if ([value respondsToSelector:@selector(boolValue)]) {
        return [value boolValue];
    }
    return NJ_SPONSOR_BLOCK_VALUE;
}

+ (void)setEnabled:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockEnabledKey];
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJ_SPONSOR_BLOCK_KEY];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)cacheEnabled {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockCacheEnabledKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setCacheEnabled:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockCacheEnabledKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)skipOnSeekToSegment {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockSkipOnSeekKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setSkipOnSeekToSegment:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockSkipOnSeekKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)testingServerEnabled {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockTestingServerKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : NO;
}

+ (void)setTestingServerEnabled:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockTestingServerKey];
    [self postSettingsDidChangeNotification];
}

+ (NSTimeInterval)minDuration {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockMinDurationKey];
    return [value respondsToSelector:@selector(doubleValue)] ? MAX(0, [value doubleValue]) : 0;
}

+ (void)setMinDuration:(NSTimeInterval)duration {
    [NJ_SETTING_CACHE setObject:@(MAX(0, duration)) forKey:NJSponsorBlockMinDurationKey];
    [self postSettingsDidChangeNotification];
}

+ (NSTimeInterval)advanceNoticeDuration {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockAdvanceNoticeDurationKey];
    return [value respondsToSelector:@selector(doubleValue)] ? MAX(0, [value doubleValue]) : 3;
}

+ (void)setAdvanceNoticeDuration:(NSTimeInterval)duration {
    [NJ_SETTING_CACHE setObject:@(MAX(0, duration)) forKey:NJSponsorBlockAdvanceNoticeDurationKey];
    [self postSettingsDidChangeNotification];
}

+ (NSString *)serverBaseURLString {
    if ([self testingServerEnabled]) {
        return NJSponsorBlockTestingServerBaseURLString;
    }
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockServerBaseURLKey];
    if ([value isKindOfClass:[NSString class]] && [(NSString *)value length] > 0) {
        return value;
    }
    return NJSponsorBlockDefaultServerBaseURLString;
}

+ (void)setServerBaseURLString:(NSString *)serverBaseURLString {
    NSString *value = [serverBaseURLString stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (value.length == 0) {
        [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockServerBaseURLKey];
    } else {
        [NJ_SETTING_CACHE setObject:value forKey:NJSponsorBlockServerBaseURLKey];
    }
    [self postSettingsDidChangeNotification];
}

+ (NSArray<NJSponsorBlockCategoryOption *> *)categoryOptions {
    return @[
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"sponsor" title:@"赞助/恰饭"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"selfpromo" title:@"自我推广"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"interaction" title:@"互动提醒"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"intro" title:@"开场动画"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"outro" title:@"结束片段"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"preview" title:@"前情/预览"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"filler" title:@"填充片段"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"music_offtopic" title:@"音乐/跑题"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"poi_highlight" title:@"精彩片段"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"padding" title:@"空白/填充"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"exclusive_access" title:@"会员专享"],
    ];
}

+ (NSArray<NJSponsorBlockCategoryOption *> *)thumbnailBadgeLabelOptions {
    return @[
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"sponsor" title:@"推广"],
        [[NJSponsorBlockCategoryOption alloc] initWithCategory:@"exclusive_access" title:@"独家"],
    ];
}

+ (NSArray<NSString *> *)requestCategories {
    NSMutableArray<NSString *> *categories = [NSMutableArray array];
    for (NJSponsorBlockCategoryOption *option in [self categoryOptions]) {
        if ([self actionForCategory:option.category] != NJSponsorBlockCategoryActionDisabled) {
            [categories addObject:option.category];
        }
    }
    return [categories copy];
}

+ (NJSponsorBlockCategoryAction)actionForCategory:(NSString *)category {
    NSNumber *storedAction = [self categoryActions][category];
    if ([storedAction respondsToSelector:@selector(integerValue)]) {
        return (NJSponsorBlockCategoryAction)[storedAction integerValue];
    }
    return [self defaultActionForCategory:category];
}

+ (void)setAction:(NJSponsorBlockCategoryAction)action forCategory:(NSString *)category {
    if (category.length == 0) {
        return;
    }
    NSMutableDictionary *actions = [[self categoryActions] mutableCopy];
    actions[category] = @(action);
    [NJ_SETTING_CACHE setObject:[actions copy] forKey:NJSponsorBlockCategoryActionsKey];
    [self postSettingsDidChangeNotification];
}

+ (NSString *)titleForAction:(NJSponsorBlockCategoryAction)action {
    switch (action) {
        case NJSponsorBlockCategoryActionDisabled:
            return @"禁用";
        case NJSponsorBlockCategoryActionShowOverlay:
            return @"仅显示";
        case NJSponsorBlockCategoryActionManualSkip:
            return @"手动跳过";
        case NJSponsorBlockCategoryActionAutoSkip:
            return @"自动跳过";
    }
}

+ (NSString *)titleForCategory:(NSString *)category {
    for (NJSponsorBlockCategoryOption *option in [self categoryOptions]) {
        if ([option.category isEqualToString:category]) {
            return option.title;
        }
    }
    return category.length > 0 ? category : @"未知片段";
}

+ (BOOL)shouldShowSegment:(NJSponsorBlockSegment *)segment {
    if (![self enabled] || ![self segmentPassesDurationFilter:segment]) {
        return NO;
    }
    return [self actionForCategory:segment.category] != NJSponsorBlockCategoryActionDisabled;
}

+ (BOOL)shouldAutoSkipSegment:(NJSponsorBlockSegment *)segment {
    return [self shouldShowSegment:segment] && [self segmentUsesSeekAction:segment] && [self actionForCategory:segment.category] == NJSponsorBlockCategoryActionAutoSkip;
}

+ (BOOL)shouldManualSkipSegment:(NJSponsorBlockSegment *)segment {
    return [self shouldShowSegment:segment] && ([self segmentUsesSeekAction:segment] || [segment.actionType isEqualToString:@"poi"]) && [self actionForCategory:segment.category] == NJSponsorBlockCategoryActionManualSkip;
}

+ (NSString *)sponsorBlockUserID {
    NSUserDefaults *defaults = NSUserDefaults.standardUserDefaults;
    NSString *userID = [defaults stringForKey:NJSponsorBlockUserIDKey];
    if (userID.length > 0) {
        return userID;
    }
    userID = NSUUID.UUID.UUIDString;
    [defaults setObject:userID forKey:NJSponsorBlockUserIDKey];
    return userID;
}

+ (void)setSponsorBlockUserID:(NSString *)userID {
    if (userID.length == 0) {
        return;
    }
    [NSUserDefaults.standardUserDefaults setObject:userID forKey:NJSponsorBlockUserIDKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)skipTrackingEnabled {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockSkipTrackingEnabledKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setSkipTrackingEnabled:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockSkipTrackingEnabledKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)showSegmentsInSeekbarWidget {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockShowSegmentsInSeekbarWidgetKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setShowSegmentsInSeekbarWidget:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockShowSegmentsInSeekbarWidgetKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)showSegmentsInProgressWidget {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockShowSegmentsInProgressWidgetKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setShowSegmentsInProgressWidget:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockShowSegmentsInProgressWidgetKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)showSharedEntryButton {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockShowSharedEntryButtonKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setShowSharedEntryButton:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockShowSharedEntryButtonKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)showVideoLabels {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockShowVideoLabelsKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setShowVideoLabels:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockShowVideoLabelsKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)showAutoSkipToast {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockShowAutoSkipToastKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setShowAutoSkipToast:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockShowAutoSkipToastKey];
    [self postSettingsDidChangeNotification];
}

+ (BOOL)showSkipUndoToast {
    id value = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockShowSkipUndoToastKey];
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : YES;
}

+ (void)setShowSkipUndoToast:(BOOL)enabled {
    [NJ_SETTING_CACHE setObject:@(enabled) forKey:NJSponsorBlockShowSkipUndoToastKey];
    [self postSettingsDidChangeNotification];
}

#pragma mark - Export / Import / Reset

+ (NSDictionary<NSString *, id> *)exportSettings {
    NSMutableDictionary<NSString *, id> *dict = [NSMutableDictionary dictionary];
    dict[@"enabled"] = @([self enabled]);
    dict[@"cacheEnabled"] = @([self cacheEnabled]);
    dict[@"skipOnSeekToSegment"] = @([self skipOnSeekToSegment]);
    dict[@"testingServerEnabled"] = @([self testingServerEnabled]);
    dict[@"skipTrackingEnabled"] = @([self skipTrackingEnabled]);
    dict[@"showVideoLabels"] = @([self showVideoLabels]);
    dict[@"minDuration"] = @([self minDuration]);
    dict[@"advanceNoticeDuration"] = @([self advanceNoticeDuration]);
    NSString *serverURL = (NSString *)[NJ_SETTING_CACHE objectForKey:NJSponsorBlockServerBaseURLKey];
    if (serverURL) {
        dict[@"serverBaseURL"] = serverURL;
    }
    dict[@"categoryActions"] = [self categoryActions];
    dict[@"categoryColors"] = [self categoryColors];
    NSDictionary *thumbnailBadgeColors = (NSDictionary *)[NJ_SETTING_CACHE objectForKey:NJSponsorBlockThumbnailBadgeColorsKey];
    if ([thumbnailBadgeColors isKindOfClass:[NSDictionary class]]) {
        dict[@"thumbnailBadgeColors"] = thumbnailBadgeColors;
    }
    dict[@"userID"] = [self sponsorBlockUserID];
    return [dict copy];
}

+ (BOOL)importSettings:(NSDictionary<NSString *, id> *)settings {
    if (!settings) {
        return NO;
    }
    if (settings[@"enabled"]) {
        [self setEnabled:[settings[@"enabled"] boolValue]];
    }
    if (settings[@"cacheEnabled"]) {
        [self setCacheEnabled:[settings[@"cacheEnabled"] boolValue]];
    }
    if (settings[@"skipOnSeekToSegment"]) {
        [self setSkipOnSeekToSegment:[settings[@"skipOnSeekToSegment"] boolValue]];
    }
    if (settings[@"testingServerEnabled"]) {
        [self setTestingServerEnabled:[settings[@"testingServerEnabled"] boolValue]];
    }
    if (settings[@"skipTrackingEnabled"]) {
        [self setSkipTrackingEnabled:[settings[@"skipTrackingEnabled"] boolValue]];
    }
    if (settings[@"showVideoLabels"]) {
        [self setShowVideoLabels:[settings[@"showVideoLabels"] boolValue]];
    }
    if (settings[@"minDuration"]) {
        [self setMinDuration:[settings[@"minDuration"] doubleValue]];
    }
    if (settings[@"advanceNoticeDuration"]) {
        [self setAdvanceNoticeDuration:[settings[@"advanceNoticeDuration"] doubleValue]];
    }
    if ([settings[@"serverBaseURL"] isKindOfClass:[NSString class]]) {
        [self setServerBaseURLString:settings[@"serverBaseURL"]];
    }
    if ([settings[@"categoryActions"] isKindOfClass:[NSDictionary class]]) {
        NSDictionary *actions = settings[@"categoryActions"];
        for (NSString *category in actions) {
            if ([actions[category] respondsToSelector:@selector(integerValue)]) {
                [self setAction:(NJSponsorBlockCategoryAction)[actions[category] integerValue] forCategory:category];
            }
        }
    }
    if ([settings[@"categoryColors"] isKindOfClass:[NSDictionary class]]) {
        [NJ_SETTING_CACHE setObject:settings[@"categoryColors"] forKey:NJSponsorBlockCategoryColorsKey];
    }
    if ([settings[@"thumbnailBadgeColors"] isKindOfClass:[NSDictionary class]]) {
        [NJ_SETTING_CACHE setObject:settings[@"thumbnailBadgeColors"] forKey:NJSponsorBlockThumbnailBadgeColorsKey];
    }
    if ([settings[@"userID"] isKindOfClass:[NSString class]] && [settings[@"userID"] length] > 0) {
        [self setSponsorBlockUserID:settings[@"userID"]];
    }
    [self postSettingsDidChangeNotification];
    return YES;
}

+ (void)resetToDefaults {
    NSString *userID = [self sponsorBlockUserID];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockEnabledKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockCacheEnabledKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockSkipOnSeekKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockTestingServerKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockSkipTrackingEnabledKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockMinDurationKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockAdvanceNoticeDurationKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockServerBaseURLKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockCategoryActionsKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockCategoryColorsKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockShowSegmentsInSeekbarWidgetKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockShowSegmentsInProgressWidgetKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockShowSharedEntryButtonKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockShowVideoLabelsKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockThumbnailBadgeColorsKey];
    [NJ_SETTING_CACHE removeObjectForKey:NJ_SPONSOR_BLOCK_KEY];
    [self setSponsorBlockUserID:userID];
    [self postSettingsDidChangeNotification];
}

+ (NSString *)requestConfigurationIdentifier {
    NSMutableArray<NSString *> *items = [NSMutableArray array];
    for (NSString *category in [self requestCategories]) {
        [items addObject:[NSString stringWithFormat:@"%@:%ld", category, (long)[self actionForCategory:category]]];
    }
    return [items componentsJoinedByString:@"|"];
}

+ (void)postSettingsDidChangeNotification {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockSettingsDidChangeNotification object:nil];
    });
}

+ (NSDictionary<NSString *, NSNumber *> *)categoryActions {
    id actions = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockCategoryActionsKey];
    return [actions isKindOfClass:[NSDictionary class]] ? actions : @{};
}

+ (NJSponsorBlockCategoryAction)defaultActionForCategory:(NSString *)category {
    if ([category isEqualToString:@"sponsor"] ||
        [category isEqualToString:@"music_offtopic"] ||
        [category isEqualToString:@"padding"]) {
        return NJSponsorBlockCategoryActionAutoSkip;
    }
    if ([category isEqualToString:@"preview"] ||
        [category isEqualToString:@"exclusive_access"]) {
        return NJSponsorBlockCategoryActionShowOverlay;
    }
    return NJSponsorBlockCategoryActionManualSkip;
}

+ (BOOL)segmentUsesSeekAction:(NJSponsorBlockSegment *)segment {
    return segment.actionType.length == 0 || [segment.actionType isEqualToString:@"skip"];
}

+ (BOOL)segmentPassesDurationFilter:(NJSponsorBlockSegment *)segment {
    if (!segment) {
        return NO;
    }
    NSTimeInterval minDuration = [self minDuration];
    return minDuration <= 0 || segment.endTime - segment.startTime >= minDuration;
}

#pragma mark - Category Colors

+ (NSString *)hexStringFromColor:(UIColor *)color {
    CGFloat r = 0, g = 0, b = 0, a = 0;
    [color getRed:&r green:&g blue:&b alpha:&a];
    return [NSString stringWithFormat:@"#%02X%02X%02X",
            (int)(r * 255), (int)(g * 255), (int)(b * 255)];
}

+ (UIColor *)colorFromHexString:(NSString *)hex {
    if (![hex hasPrefix:@"#"] || hex.length != 7) {
        return nil;
    }
    unsigned int rgb = 0;
    NSScanner *scanner = [NSScanner scannerWithString:[hex substringFromIndex:1]];
    if (![scanner scanHexInt:&rgb]) {
        return nil;
    }
    return [UIColor colorWithRed:((rgb >> 16) & 0xFF) / 255.0
                           green:((rgb >> 8) & 0xFF) / 255.0
                            blue:(rgb & 0xFF) / 255.0
                           alpha:1.0];
}

+ (UIColor *)defaultColorForCategory:(NSString *)category {
    if ([category isEqualToString:@"sponsor"]) {
        return [UIColor colorWithRed:0 green:0.90 blue:0.10 alpha:1.0];
    }
    if ([category isEqualToString:@"intro"]) {
        return [UIColor cyanColor];
    }
    if ([category isEqualToString:@"outro"]) {
        return [UIColor colorWithRed:0.92 green:0.40 blue:0.95 alpha:1.0];
    }
    if ([category isEqualToString:@"selfpromo"]) {
        return [UIColor colorWithRed:1.00 green:0.65 blue:0.10 alpha:1.0];
    }
    if ([category isEqualToString:@"preview"] || [category isEqualToString:@"poi_highlight"] || [category isEqualToString:@"exclusive_access"]) {
        return [UIColor colorWithRed:1.00 green:0.86 blue:0.18 alpha:1.0];
    }
    if ([category isEqualToString:@"filler"] || [category isEqualToString:@"music_offtopic"]) {
        return [UIColor colorWithRed:0.55 green:0.72 blue:1.00 alpha:1.0];
    }
    return [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:1.0];
}

+ (UIColor *)colorForCategory:(NSString *)category {
    NSDictionary *colors = [self categoryColors];
    NSString *hex = colors[category];
    if (hex) {
        UIColor *color = [self colorFromHexString:hex];
        if (color) {
            return color;
        }
    }
    return [self defaultColorForCategory:category];
}

+ (void)setColor:(UIColor *)color forCategory:(NSString *)category {
    if (category.length == 0 || !color) {
        return;
    }
    NSMutableDictionary *colors = [[self categoryColors] mutableCopy];
    colors[category] = [self hexStringFromColor:color];
    [NJ_SETTING_CACHE setObject:[colors copy] forKey:NJSponsorBlockCategoryColorsKey];
    [self postSettingsDidChangeNotification];
}

+ (void)resetColors {
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockCategoryColorsKey];
    [self postSettingsDidChangeNotification];
}

+ (NSDictionary<NSString *, NSString *> *)categoryColors {
    id colors = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockCategoryColorsKey];
    return [colors isKindOfClass:[NSDictionary class]] ? colors : @{};
}

#pragma mark - Thumbnail Badge Colors

+ (UIColor *)thumbnailBadgeColorForLabel:(NSString *)label {
    NSDictionary *colors = (NSDictionary *)[NJ_SETTING_CACHE objectForKey:NJSponsorBlockThumbnailBadgeColorsKey];
    if ([colors isKindOfClass:[NSDictionary class]]) {
        UIColor *color = [self colorFromHexString:colors[label]];
        if (color) {
            return color;
        }
    }
    return [self defaultThumbnailBadgeColorForLabel:label];
}

+ (UIColor *)defaultThumbnailBadgeColorForLabel:(NSString *)label {
    return [[UIColor blackColor] colorWithAlphaComponent:0.72];
}

+ (void)setThumbnailBadgeColor:(UIColor *)color forLabel:(NSString *)label {
    if (label.length == 0) {
        return;
    }
    NSMutableDictionary *colors = [(NSDictionary *)[NJ_SETTING_CACHE objectForKey:NJSponsorBlockThumbnailBadgeColorsKey] mutableCopy];
    if (!colors) {
        colors = [NSMutableDictionary dictionary];
    }
    if (color) {
        colors[label] = [self hexStringFromColor:color];
    } else {
        [colors removeObjectForKey:label];
    }
    [NJ_SETTING_CACHE setObject:[colors copy] forKey:NJSponsorBlockThumbnailBadgeColorsKey];
    [self postSettingsDidChangeNotification];
}

+ (void)resetThumbnailBadgeColors {
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockThumbnailBadgeColorsKey];
    [self postSettingsDidChangeNotification];
}

@end

YYCache* getSettingsCache(void) {
    static YYCache* cache;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        cache = ((NJSettingCache*)[objc_getClass("NJSettingCache") sharedInstance]).cache;
    });
    return cache;
}
