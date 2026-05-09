//
//  NJSettingCache.h
//  nmsl
//
//  Created by s s on 2026/5/2.
//
#include "../Vendor/YYCache.h"
#include <objc/objc.h>
#include <objc/runtime.h>

@interface NJSettingCache : NSObject

/// YYCache
@property (nonatomic, strong) YYCache *cache;

/// 单例
+ (instancetype)sharedInstance;
@end

YYCache* getSettingsCache(void);

#define NJ_SETTING_CACHE getSettingsCache()
