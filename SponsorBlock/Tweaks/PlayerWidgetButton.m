//
//  PlayerWidgetButton.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "Tweaks.h"
#include "../UI/NJSponsorBlockPanelView.h"
#include "../Settings/NJSponsorBlockSettings.h"
#import <objc/message.h>

static const char* kTopWidgetClassNames[] = {
    "BBPlayerHalfScreenTopWidget",
    "BBPlayerFullScreenTopWidget",
    "BBHD2MPPlayerHalfScreenTopWidget",
    "BBHD2MPPlayerFullScreenTopWidget",
};
#define kTopWidgetClassCount 4

static void (*orig_setupSubWidgets[kTopWidgetClassCount])(BBPlayerWidget*, SEL);
static Ivar rightControlWidgetIvars[kTopWidgetClassCount];

void hook_setupSubWidgets(BBPlayerWidget* self, SEL sel) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        for (int i = 0; i < kTopWidgetClassCount; i++) {
            rightControlWidgetIvars[i] = class_getInstanceVariable(objc_lookUpClass(kTopWidgetClassNames[i]), "_rightControlWidget");
        }
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
    
    BBPlayerCastBtnWidget* fakeWidget = [[PrivClass(OpenPanelButtonWidget) alloc] initWithContext:[self context]];

    [rightControlWidget addSubWidget:(BBPlayerWidget*)fakeWidget];
}

void registerOpenPanelButtonWidget(void);
void registerSponsorBlockPanelWidget(void);

void initPlayerWidgetButtonHooks(void) {
    registerOpenPanelButtonWidget();
    registerSponsorBlockPanelWidget();

    for (int i = 0; i < kTopWidgetClassCount; i++) {
        JRSwizzleInstanceMethod(objc_lookUpClass(kTopWidgetClassNames[i]),
                                @selector(setupSubWidgets),
                                (IMP)hook_setupSubWidgets,
                                (IMP*)&orig_setupSubWidgets[i]);
    }
}
