//
//  NJSponsorBlockSettings.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockSettings.h"
#import "NJSponsorBlockSegment.h"
#import "NJCommonDefine.h"
#import "NJSettingCache.h"

NSNotificationName const NJSponsorBlockSettingsDidChangeNotification = @"NJSponsorBlockSettingsDidChangeNotification";

static NSString * const NJSponsorBlockEnabledKey = @"NJSponsorBlockEnabledKey";
static NSString * const NJSponsorBlockCacheEnabledKey = @"NJSponsorBlockCacheEnabledKey";
static NSString * const NJSponsorBlockSkipOnSeekKey = @"NJSponsorBlockSkipOnSeekKey";
static NSString * const NJSponsorBlockTestingServerKey = @"NJSponsorBlockTestingServerKey";
static NSString * const NJSponsorBlockMinDurationKey = @"NJSponsorBlockMinDurationKey";
static NSString * const NJSponsorBlockAdvanceNoticeDurationKey = @"NJSponsorBlockAdvanceNoticeDurationKey";
static NSString * const NJSponsorBlockServerBaseURLKey = @"NJSponsorBlockServerBaseURLKey";
static NSString * const NJSponsorBlockCategoryActionsKey = @"NJSponsorBlockCategoryActionsKey";

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

@end
