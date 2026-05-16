//
//  OpenPanelButtonWidget.m
//  SponsorBlock
//
//  Created by s s on 2026/5/15.
//
#include "../Tweaks.h"
#include "../../UI/NJSponsorBlockPanelView.h"
#include "../../Services/NJSponsorBlockManager.h"
@import ObjectiveC;
@import UIKit;

static UIView *OpenPanelButtonWidget_view(id self, SEL _cmd) {
    NJSponsorBlockManager* manager = objc_getAssociatedObject([self context], sponsorBlockManagerKey);
    NJSponsorBlockPanelView * panel = [manager panelView];
    if(!panel) {
        panel = [[NJSponsorBlockPanelView alloc] initWithManager:manager];
        manager.panelView = panel;
    }

    return panel;
}


void registerSponsorBlockPanelWidget(void) {
    Class BBPlayerFloatingWidgetClass = objc_allocateClassPair(PrivClass(BBPlayerFloatingWidget), "SponsorBlockPanelWidget", 0);

    class_addMethod(
                    BBPlayerFloatingWidgetClass,
        @selector(view),
        (IMP)OpenPanelButtonWidget_view,
        "@@:"
    );
    
    objc_registerClassPair(BBPlayerFloatingWidgetClass);
}
