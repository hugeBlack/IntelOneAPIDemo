//
//  NJSettingCache.h
//  nmsl
//
//  Created by s s on 2026/5/2.
//
#include "YYCache.h"
@import ObjectiveC;
#define PrivClass(name) ((Class)objc_lookUpClass(#name))

@interface NJSettingCache : NSObject

/// YYCache
@property (nonatomic, strong) YYCache *cache;

/// 单例
+ (instancetype)sharedInstance;
@end

#define NJ_SETTING_CACHE ((NJSettingCache*)[PrivClass(NJSettingCache) sharedInstance]).cache
