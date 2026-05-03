//
//  NJSponsorBlockCacheStats.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockCacheStats.h"
#import "NJSettingCache.h"

static NSString * const NJSponsorBlockCacheStatsKey = @"NJSponsorBlockCacheStatsKey";
static NSString * const NJSponsorBlockCachePrefix = @"NJSponsorBlockSegments";

static NSString * const kTotalItems = @"totalItems";
static NSString * const kTotalSizeBytes = @"totalSizeBytes";
static NSString * const kDailyHits = @"dailyHits";
static NSString * const kDailySizeBytes = @"dailySizeBytes";
static NSString * const kDailyDate = @"dailyDate";

@implementation NJSponsorBlockCacheStats {
    NSMutableDictionary<NSString *, id> *_stats;
}

+ (instancetype)sharedInstance {
    static NJSponsorBlockCacheStats *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[NJSponsorBlockCacheStats alloc] init];
    });
    return instance;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        [self loadStats];
        [self checkDailyReset];
    }
    return self;
}

- (void)loadStats {
    NSDictionary *stored = [NSUserDefaults.standardUserDefaults objectForKey:NJSponsorBlockCacheStatsKey];
    if ([stored isKindOfClass:[NSDictionary class]]) {
        _stats = [stored mutableCopy];
    } else {
        _stats = [NSMutableDictionary dictionary];
        _stats[kTotalItems] = @(0);
        _stats[kTotalSizeBytes] = @(0);
        _stats[kDailyHits] = @(0);
        _stats[kDailySizeBytes] = @(0);
        _stats[kDailyDate] = [self todayDateString];
    }
}

- (void)saveStats {
    [NSUserDefaults.standardUserDefaults setObject:[_stats copy] forKey:NJSponsorBlockCacheStatsKey];
}

- (NSString *)todayDateString {
    NSDateFormatter *fmt = [[NSDateFormatter alloc] init];
    fmt.dateFormat = @"yyyy-MM-dd";
    return [fmt stringFromDate:[NSDate date]];
}

- (void)checkDailyReset {
    NSString *storedDate = _stats[kDailyDate];
    NSString *today = [self todayDateString];
    if (![storedDate isEqualToString:today]) {
        _stats[kDailyHits] = @(0);
        _stats[kDailySizeBytes] = @(0);
        _stats[kDailyDate] = today;
        [self saveStats];
    }
}

- (NSUInteger)totalItems {
    return [_stats[kTotalItems] unsignedIntegerValue];
}

- (void)setTotalItems:(NSUInteger)totalItems {
    _stats[kTotalItems] = @(totalItems);
    [self saveStats];
}

- (NSUInteger)totalSizeBytes {
    return [_stats[kTotalSizeBytes] unsignedIntegerValue];
}

- (void)setTotalSizeBytes:(NSUInteger)totalSizeBytes {
    _stats[kTotalSizeBytes] = @(totalSizeBytes);
    [self saveStats];
}

- (NSUInteger)dailyHits {
    [self checkDailyReset];
    return [_stats[kDailyHits] unsignedIntegerValue];
}

- (void)setDailyHits:(NSUInteger)dailyHits {
    _stats[kDailyHits] = @(dailyHits);
    [self saveStats];
}

- (NSUInteger)dailySizeBytes {
    [self checkDailyReset];
    return [_stats[kDailySizeBytes] unsignedIntegerValue];
}

- (void)setDailySizeBytes:(NSUInteger)dailySizeBytes {
    _stats[kDailySizeBytes] = @(dailySizeBytes);
    [self saveStats];
}

- (void)recordHitWithSize:(NSUInteger)size {
    [self checkDailyReset];
    NSUInteger hits = [_stats[kDailyHits] unsignedIntegerValue];
    NSUInteger dailySize = [_stats[kDailySizeBytes] unsignedIntegerValue];
    _stats[kDailyHits] = @(hits + 1);
    _stats[kDailySizeBytes] = @(dailySize + size);
    [self saveStats];
}

- (void)recordSaveWithSize:(NSUInteger)size {
    NSUInteger items = [_stats[kTotalItems] unsignedIntegerValue];
    NSUInteger totalSize = [_stats[kTotalSizeBytes] unsignedIntegerValue];
    _stats[kTotalItems] = @(items + 1);
    _stats[kTotalSizeBytes] = @(totalSize + size);
    [self saveStats];
}

- (void)recordRemoval {
    NSUInteger items = [_stats[kTotalItems] unsignedIntegerValue];
    if (items > 0) {
        _stats[kTotalItems] = @(items - 1);
    }
    [self saveStats];
}

- (void)clearAll {
    _stats[kTotalItems] = @(0);
    _stats[kTotalSizeBytes] = @(0);
    _stats[kDailyHits] = @(0);
    _stats[kDailySizeBytes] = @(0);
    _stats[kDailyDate] = [self todayDateString];
    [self saveStats];
}

- (void)refreshFromCache {
    // Recount items by scanning known cache keys
    // Since YYCache doesn't expose enumeration, we track via recordSave/recordRemoval
    // This method just ensures daily reset is applied
    [self checkDailyReset];
}

- (NSDictionary<NSString *, id> *)exportStats {
    [self checkDailyReset];
    return @{
        @"totalItems": _stats[kTotalItems] ?: @(0),
        @"totalSizeBytes": _stats[kTotalSizeBytes] ?: @(0),
        @"dailyHits": _stats[kDailyHits] ?: @(0),
        @"dailySizeBytes": _stats[kDailySizeBytes] ?: @(0),
        @"dailyDate": _stats[kDailyDate] ?: [self todayDateString],
    };
}

@end
