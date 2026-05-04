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

static const char* kTopWidgetClassNames[] = {
    "BBPlayerHalfScreenTopWidget",
    "BBPlayerFullScreenTopWidget",
    "BBHD2MPPlayerHalfScreenTopWidget",
    "BBHD2MPPlayerFullScreenTopWidget",
};
#define kTopWidgetClassCount 4

static void (*orig_setupSubWidgets[kTopWidgetClassCount])(BBPlayerWidget*, SEL);
static Ivar rightControlWidgetIvars[kTopWidgetClassCount];
static Ivar castButtonIvar;

void hook_setupSubWidgets(BBPlayerWidget* self, SEL sel) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        for (int i = 0; i < kTopWidgetClassCount; i++) {
            rightControlWidgetIvars[i] = class_getInstanceVariable(objc_lookUpClass(kTopWidgetClassNames[i]), "_rightControlWidget");
        }
        castButtonIvar = class_getInstanceVariable(PrivClass(BBPlayerCastBtnWidget), "_castBtn");
    });
    
    int classIndex = -1;
    for (int i = 0; i < kTopWidgetClassCount; i++) {
        if (![self isKindOfClass:objc_lookUpClass(kTopWidgetClassNames[i])]) continue;
        classIndex = i;
        break;
    }
    assert(classIndex != -1);

    orig_setupSubWidgets[classIndex](self, sel);


    if (![NJSponsorBlockSettings enabled]) return;
    if (![NJSponsorBlockSettings showSharedEntryButton]) return;
    Ivar rightControlWidgetIvar = rightControlWidgetIvars[classIndex];

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

    for (int i = 0; i < kTopWidgetClassCount; i++) {
        JRSwizzleInstanceMethod(objc_lookUpClass(kTopWidgetClassNames[i]),
                                @selector(setupSubWidgets),
                                (IMP)hook_setupSubWidgets,
                                (IMP*)&orig_setupSubWidgets[i]);
    }
}
