//
//  PlayerWidgetButton.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "Tweaks.h"
#include "../NJSponsorBlockPanelView.h"
#include "../NJSponsorBlockSettings.h"

void hook_BBPlayerFlexContainerWidget_viewWillDisappear(id self, SEL sel, bool animated) {
    [NJSponsorBlockPanelView removePanel];
}

void (*orig_BBPlayerHalfScreenTopWidget_setupSubWidgets)(BBPlayerWidget* self, SEL sel) = nil;
void (*orig_BBPlayerFullScreenTopWidget_setupSubWidgets)(BBPlayerWidget* self, SEL sel) = nil;
void (*orig_BBHD2MPPlayerHalfScreenTopWidget_setupSubWidgets)(BBPlayerWidget* self, SEL sel) = nil;
void (*orig_BBHD2MPPlayerFullScreenTopWidget_setupSubWidgets)(BBPlayerWidget* self, SEL sel) = nil;
void hook_BBPlayerHalfScreenTopWidget_setupSubWidgets(BBPlayerWidget* self, SEL sel) {
    static Ivar BBPlayerHalfScreenTopWidget_rightControlWidgetIvar;
    static Ivar BBPlayerFullScreenTopWidget_rightControlWidgetIvar;
    static Ivar BBHD2MPPlayerHalfScreenTopWidget_rightControlWidgetIvar;
    static Ivar BBHD2MPPlayerFullScreenTopWidget_rightControlWidgetIvar;
    static Ivar castButtonIvar;
    
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        BBPlayerHalfScreenTopWidget_rightControlWidgetIvar = class_getInstanceVariable(PrivClass(BBPlayerHalfScreenTopWidget), "_rightControlWidget");
        BBPlayerFullScreenTopWidget_rightControlWidgetIvar = class_getInstanceVariable(PrivClass(BBPlayerFullScreenTopWidget), "_rightControlWidget");
        BBHD2MPPlayerHalfScreenTopWidget_rightControlWidgetIvar = class_getInstanceVariable(PrivClass(BBHD2MPPlayerHalfScreenTopWidget), "_rightControlWidget");
        BBHD2MPPlayerFullScreenTopWidget_rightControlWidgetIvar = class_getInstanceVariable(PrivClass(BBHD2MPPlayerFullScreenTopWidget), "_rightControlWidget");
        castButtonIvar = class_getInstanceVariable(PrivClass(BBPlayerCastBtnWidget), "_castBtn");
    });
    
    Ivar rightControlWidgetIvar = 0;
    if([self isKindOfClass:PrivClass(BBPlayerHalfScreenTopWidget)]) {
        orig_BBPlayerHalfScreenTopWidget_setupSubWidgets(self, sel);
        rightControlWidgetIvar = BBPlayerHalfScreenTopWidget_rightControlWidgetIvar;
    } else if([self isKindOfClass:PrivClass(BBPlayerFullScreenTopWidget)]) {
        orig_BBPlayerFullScreenTopWidget_setupSubWidgets(self, sel);
        rightControlWidgetIvar = BBPlayerFullScreenTopWidget_rightControlWidgetIvar;
    } else if([self isKindOfClass:PrivClass(BBHD2MPPlayerHalfScreenTopWidget)]) {
        orig_BBHD2MPPlayerHalfScreenTopWidget_setupSubWidgets(self, sel);
        rightControlWidgetIvar = BBHD2MPPlayerHalfScreenTopWidget_rightControlWidgetIvar;
    } else {
        orig_BBHD2MPPlayerFullScreenTopWidget_setupSubWidgets(self, sel);
        rightControlWidgetIvar = BBHD2MPPlayerFullScreenTopWidget_rightControlWidgetIvar;
    }
    
    if(![NJSponsorBlockSettings enabled]) {
        return;
    }

    BBPlayerWidget* rightControlWidget = object_getIvar(self, rightControlWidgetIvar);
    if([[rightControlWidget subWidgets] count] == 0) {
        return;
    }
    
    BBPlayerCastBtnWidget* fakeWidget = [[PrivClass(BBPlayerCastBtnWidget) alloc] initWithContext:nil];

    object_setIvar(fakeWidget, castButtonIvar, [NJSponsorBlockPanelView sharedEntryButton]);
    
    [rightControlWidget addSubWidget:(BBPlayerWidget*)fakeWidget];
}



void initPlayerWidgetButtonHooks(void) {
    class_addMethod(objc_getClass("BBPlayerControlContainerWidgetView"),
                    @selector(viewWillDisappear:),
                    (IMP)hook_BBPlayerFlexContainerWidget_viewWillDisappear,
                    "v@:B");
    
    JRSwizzleInstanceMethod(PrivClass(BBPlayerHalfScreenTopWidget), @selector(setupSubWidgets),
                            (IMP)hook_BBPlayerHalfScreenTopWidget_setupSubWidgets,
                            (IMP*)&orig_BBPlayerHalfScreenTopWidget_setupSubWidgets);
    
    JRSwizzleInstanceMethod(PrivClass(BBPlayerFullScreenTopWidget), @selector(setupSubWidgets),
                            (IMP)hook_BBPlayerHalfScreenTopWidget_setupSubWidgets,
                            (IMP*)&orig_BBPlayerFullScreenTopWidget_setupSubWidgets);
    
    JRSwizzleInstanceMethod(PrivClass(BBHD2MPPlayerHalfScreenTopWidget), @selector(setupSubWidgets),
                            (IMP)hook_BBPlayerHalfScreenTopWidget_setupSubWidgets,
                            (IMP*)&orig_BBHD2MPPlayerHalfScreenTopWidget_setupSubWidgets);
    
    JRSwizzleInstanceMethod(PrivClass(BBHD2MPPlayerFullScreenTopWidget), @selector(setupSubWidgets),
                            (IMP)hook_BBPlayerHalfScreenTopWidget_setupSubWidgets,
                            (IMP*)&orig_BBHD2MPPlayerFullScreenTopWidget_setupSubWidgets);
}
