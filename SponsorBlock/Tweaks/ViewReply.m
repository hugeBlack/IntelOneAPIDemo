//
//  ViewReply.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "Tweaks.h"
#include "../Settings/NJSettingDefine.h"
#include "../Services/NJSponsorBlockManager.h"

NSMutableDictionary* cachedCidVideoInfoDict = nil;

id (*orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    BAPIAppViewuniteV1ViewReply* ret = orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(self, sel, data, registry, error);
    if([ret hasArc]) {
        NSNumber* cid = @([[ret arc] cid]);
        NSDictionary* userInfo = @{
            @"videoID": [[ret arc] bvid],
            @"cid": cid,
            @"duration": @([[ret arc] duration])
        };
        cachedCidVideoInfoDict[cid] = userInfo;
        [NSNotificationCenter.defaultCenter postNotificationName:NJSponsorBlockVideoInfoRetrievedNotification object:self];
    }

    return ret;
}

id (*orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    BAPIAppViewV1ViewReply* ret = orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error(self, sel, data, registry, error);
    if([ret hasArc]) {
        NSNumber* cid = @([[ret arc] firstCid]);
        NSDictionary* userInfo = @{
            @"videoID": [ret bvid],
            @"cid": cid,
            @"duration": @([[ret arc] duration])
        };
        cachedCidVideoInfoDict[cid] = userInfo;
        [NSNotificationCenter.defaultCenter postNotificationName:NJSponsorBlockVideoInfoRetrievedNotification object:self];
    }
    return ret;
}

void initViewReplyHooks(void) {
    cachedCidVideoInfoDict = [NSMutableDictionary new];
    
    JRSwizzleInstanceMethod(objc_getClass("BAPIAppViewuniteV1ViewReply"), @selector(initWithData:extensionRegistry:error:),
                            (IMP)hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error,
                            (IMP*)&orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error);
    
    JRSwizzleInstanceMethod(objc_getClass("BAPIAppViewV1ViewReply"), @selector(initWithData:extensionRegistry:error:),
                            (IMP)hook_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error,
                            (IMP*)&orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error);
    
}
