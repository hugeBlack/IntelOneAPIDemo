//
//  Seekbar.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "Tweaks.h"
#include "../NJSponsorBlockPanelView.h"

// hooks start
void (*orig_BBPlayerSeekbarContainerView_layoutSubviews)(id self, SEL sel) = nil;
void hook_BBPlayerSeekbarContainerView_layoutSubviews(id self, SEL sel) {
    [NJSponsorBlockPanelView installNativeTimelineInView:(UIView *)self];
    orig_BBPlayerSeekbarContainerView_layoutSubviews(self, sel);
}

void initSeekbarHooks(void) {

    JRSwizzleInstanceMethod(objc_getClass("BBPlayerSeekbarContainerView"), @selector(layoutSubviews),
                            (IMP)hook_BBPlayerSeekbarContainerView_layoutSubviews,
                            (IMP*)&orig_BBPlayerSeekbarContainerView_layoutSubviews);
}
