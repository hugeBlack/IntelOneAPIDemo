//
//  Seekbar.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "Tweaks.h"
#include "../UI/NJSponsorBlockPanelView.h"
#include "../UI/NJSponsorBlockTimelineView.h"
#include "../Settings/NJSponsorBlockSettings.h"

static const char* kSeekbarWidgetClassNames[] = {
    "BBPlayerSeekbarWidgetV2",
    "BBHD2MPSeekbarWidget",
    "BBPlayerSeekbarWidgetV3",
    "BBPlayerSeekbarWidget",
};
#define kSeekbarWidgetClassCount 4

static void (*orig_willLayoutSubWidgets[kSeekbarWidgetClassCount])(BBPlayerWidget*, SEL);
static Ivar progressContainerIvars[kSeekbarWidgetClassCount];

void hook_willLayoutSubWidgets(id self, SEL sel) {
    
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        for (int i = 0; i < kSeekbarWidgetClassCount; i++) {
            progressContainerIvars[i] = class_getInstanceVariable(objc_lookUpClass(kSeekbarWidgetClassNames[i]), "_progressContainer");
        }
    });
    int classIndex = -1;
    for (int i = 0; i < kSeekbarWidgetClassCount; i++) {
        if (![self isKindOfClass:objc_lookUpClass(kSeekbarWidgetClassNames[i])]) continue;
        classIndex = i;
        break;
    }
    assert(classIndex != -1);
    
    Ivar playerTrackViewIvar = progressContainerIvars[classIndex];
    UIView* playerTrackView = object_getIvar(self, playerTrackViewIvar);
    if ([NJSponsorBlockSettings showSegmentsInSeekbarWidget]) {
        NJSponsorBlockManager* manager = objc_getAssociatedObject([self context], sponsorBlockManagerKey);
        
        [NJSponsorBlockTimelineView installNativeTimelineInView:playerTrackView manager:manager];
    }
    
    orig_willLayoutSubWidgets[classIndex](self, sel);
}

void (*orig_BBPlayerProgressWidget_willLayoutSubWidgets)(id self, SEL sel) = nil;
void hook_BBPlayerProgressWidget_willLayoutSubWidgets(id self, SEL sel) {
    static Ivar seekViewIvar;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        seekViewIvar = class_getInstanceVariable([self class], "_seekView");
    });

    if ([NJSponsorBlockSettings showSegmentsInProgressWidget]) {
        UIView* seekView = object_getIvar(self, seekViewIvar);
        NJSponsorBlockManager* manager = objc_getAssociatedObject([self context], sponsorBlockManagerKey);
        
        [NJSponsorBlockTimelineView installNativeTimelineInView:seekView manager:manager];
    }
    orig_BBPlayerProgressWidget_willLayoutSubWidgets(self, sel);
}


void initSeekbarHooks(void) {
    for (int i = 0; i < kSeekbarWidgetClassCount; i++) {
        JRSwizzleInstanceMethod(objc_lookUpClass(kSeekbarWidgetClassNames[i]),
                                @selector(willLayoutSubWidgets),
                                (IMP)hook_willLayoutSubWidgets,
                                (IMP*)&orig_willLayoutSubWidgets[i]);
    }
    
    JRSwizzleInstanceMethod(objc_getClass("BBPlayerProgressWidget"), @selector(willLayoutSubWidgets),
                            (IMP)hook_BBPlayerProgressWidget_willLayoutSubWidgets,
                            (IMP*)&orig_BBPlayerProgressWidget_willLayoutSubWidgets);
}
