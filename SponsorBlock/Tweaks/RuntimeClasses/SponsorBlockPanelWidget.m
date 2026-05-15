//
//  OpenPanelButtonWidget.m
//  SponsorBlock
//
//  Created by s s on 2026/5/15.
//
#include "../Tweaks.h"
#include "../../UI/NJSponsorBlockPanelView.h"
@import ObjectiveC;
@import UIKit;

static UIView *OpenPanelButtonWidget_view(id self, SEL _cmd) {
    NJSponsorBlockPanelView* panel = [NJSponsorBlockPanelView sharedPanel];
    [panel refreshContent];
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
