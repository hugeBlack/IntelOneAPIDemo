//
//  PlayerContext.m
//  SponsorBlock
//
//  Created by s s on 2026/5/16.
//
#include "Tweaks.h"
#include "../Services/NJSponsorBlockManager.h"

void* sponsorBlockManagerKey = &sponsorBlockManagerKey;

BBPlayerContext* (*orig_BBPlayerContext_initWithConfiguration)(id self, SEL sel, id configuration) = nil;
static BBPlayerContext* hook_BBPlayerContext_initWithConfiguration(BBPlayerContext* self, SEL sel, id configuration) {
    BBPlayerContext* ans = orig_BBPlayerContext_initWithConfiguration(self, sel, configuration);
    NJSponsorBlockManager* manager = [[NJSponsorBlockManager alloc] initWithContext:ans];
    
    objc_setAssociatedObject(ans, sponsorBlockManagerKey, manager, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    
    return ans;
}


void initPlayerContextHooks(void) {
    JRSwizzleInstanceMethod(PrivClass(BBPlayerContext),
                            @selector(initWithConfiguration:),
                            (IMP)hook_BBPlayerContext_initWithConfiguration,
                            (IMP*)&orig_BBPlayerContext_initWithConfiguration);
}
