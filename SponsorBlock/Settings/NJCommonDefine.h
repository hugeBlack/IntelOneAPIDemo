//
//  NJCommonDefine.h
//  SponsorBlock
//
//  Created by s s on 2026/5/2.
//
/// 总开关
#include "NJSettingCache.h"
#define NJ_MASTER_SWITCH_KEY @"NJ_MASTER_SWITCH_KEY"
/// 总开关的值
#define NJ_MASTER_SWITCH_VALUE (![NJ_SETTING_CACHE containsObjectForKey:NJ_MASTER_SWITCH_KEY] || \
[(NSNumber *)[NJ_SETTING_CACHE objectForKey:NJ_MASTER_SWITCH_KEY] boolValue])
/// SponsorBlock 跳过片段
#define NJ_SPONSOR_BLOCK_KEY @"NJ_SPONSOR_BLOCK_KEY"
/// SponsorBlock 跳过片段的值
#define NJ_SPONSOR_BLOCK_VALUE (![NJ_SETTING_CACHE containsObjectForKey:NJ_SPONSOR_BLOCK_KEY] || \
[(NSNumber *)[NJ_SETTING_CACHE objectForKey:NJ_SPONSOR_BLOCK_KEY] boolValue])



