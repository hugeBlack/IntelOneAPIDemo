//
//  ViewReply.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "Tweaks.h"
#include "../Settings/NJSettingDefine.h"
#include "../Services/NJSponsorBlockManager.h"

id (*orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    id ret = orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(self, sel, data, registry, error);
    [[NJSponsorBlockManager sharedInstance] inspectModelObject:ret source:@"BAPIAppViewuniteV1ViewReply"];
    return ret;
}

id (*orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    id ret = orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error(self, sel, data, registry, error);
    [[NJSponsorBlockManager sharedInstance] inspectModelObject:ret source:@"BAPIAppViewV1ViewReply"];
    return ret;
}

//@interface BBVDPlayerFlowConfig: NSObject
//@property NSInteger avid;
//@property (nonatomic, copy) NSString *bvid;
//@property NSInteger cid;
//@end
//
//void (*orig_BBVDPlayerVC_play)(id self, SEL sel, BBVDPlayerFlowConfig* config) = nil;
//void hook_BBVDPlayerVC_play(id self, SEL sel, BBVDPlayerFlowConfig* config) {
//    orig_BBVDPlayerVC_play(self, sel, config);
//
//    [[NJSponsorBlockManager sharedInstance] updateVideoID:config.bvid cid:config.cid];
//    [[NJSponsorBlockManager sharedInstance] updateNativeVideoDuration:123];
//}

void initViewReplyHooks(void) {
    //    JRSwizzleInstanceMethod(objc_getClass("BBVDPlayerVC"), @selector(play:),
    //                            (IMP)hook_BBVDPlayerVC_play,
    //                            (IMP*)&orig_BBVDPlayerVC_play);
        JRSwizzleInstanceMethod(objc_getClass("BAPIAppViewuniteV1ViewReply"), @selector(initWithData:extensionRegistry:error:),
                                (IMP)hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error,
                                (IMP*)&orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error);
        
        JRSwizzleInstanceMethod(objc_getClass("BAPIAppViewV1ViewReply"), @selector(initWithData:extensionRegistry:error:),
                                (IMP)hook_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error,
                                (IMP*)&orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error);
}
