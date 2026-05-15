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

static Ivar contextIvar;
static Ivar viewIvar;
static Class BBPlayerCastBtnWidgetClass;

static id OpenPanelButtonWidget_initWithContext(id self, SEL _cmd, id context) {
    struct objc_super superInfo = {
        .receiver = self,
        .super_class = BBPlayerCastBtnWidgetClass
    };

    id (*objc_msgSendSuper_id)(struct objc_super *, SEL, id) =
        (id (*)(struct objc_super *, SEL, id))objc_msgSendSuper;

    self = objc_msgSendSuper_id(&superInfo, @selector(initWithContext:), context);
    if (!self) {
        return nil;
    }

    object_setIvar(self, contextIvar, context);

    return self;
}

static UIButton* createEntryButton(void) {
    UIButton* button = [UIButton buttonWithType:UIButtonTypeCustom];
    button.frame = CGRectMake(0, 0, 38, 38);
    button.accessibilityIdentifier = @"NJSponsorBlockEntryButton";
//        button.backgroundColor = [UIColor colorWithWhite:0 alpha:0.36];
//        button.layer.cornerRadius = 19;
//        button.layer.borderWidth = 1;
//        button.layer.borderColor = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95].CGColor;
    button.titleLabel.font = [UIFont boldSystemFontOfSize:23];
    [button setTitle:@"▷" forState:UIControlStateNormal];
    [button setTitleColor:[UIColor colorWithRed:0.02 green:0.78 blue:1 alpha:1] forState:UIControlStateNormal];

    return button;
}

static UIView *OpenPanelButtonWidget_view(id self, SEL _cmd) {
    UIButton* button = object_getIvar(self, viewIvar);
    if(!button) {
        button = createEntryButton();
        [button addTarget:self action:@selector(togglePanelFromEntryButton:) forControlEvents:UIControlEventTouchUpInside];
        object_setIvar(self, viewIvar, button);
    }
    
    return button;
}

@interface BBPlayerFeatureWidgetService : NSObject
-(void)pushWidget:(id)arg0;
@end

@interface BBPlayerContext : NSObject
- (id)featureWidgetService;
@end

static void OpenPanelButtonWidget_togglePanelFromEntryButton(id self, SEL _cmd, id sender) {
    BBPlayerContext* context = object_getIvar(self, contextIvar);
    [[context featureWidgetService] pushWidget:[[PrivClass(SponsorBlockPanelWidget) alloc] initWithContext:context]];
}


void registerOpenPanelButtonWidget(void) {
    BBPlayerCastBtnWidgetClass = PrivClass(BBPlayerCastBtnWidget);
    Class OpenPanelButtonWidgetClass = objc_allocateClassPair(BBPlayerCastBtnWidgetClass, "OpenPanelButtonWidget", 0);
    class_addIvar(OpenPanelButtonWidgetClass, "_context", sizeof(id), log2(sizeof(id)), "@");
    
    class_addMethod(
                    OpenPanelButtonWidgetClass,
                    @selector(initWithContext:),
                    (IMP)OpenPanelButtonWidget_initWithContext,
                    "@@:@"
                    );
    
    class_addMethod(
                    OpenPanelButtonWidgetClass,
                    @selector(view),
                    (IMP)OpenPanelButtonWidget_view,
                    "@@:"
                    );
    
    class_addMethod(
                    OpenPanelButtonWidgetClass,
                    @selector(togglePanelFromEntryButton:),
                    (IMP)OpenPanelButtonWidget_togglePanelFromEntryButton,
                    "v@:@"
                    );
    
    objc_registerClassPair(OpenPanelButtonWidgetClass);
    
    contextIvar = class_getInstanceVariable(OpenPanelButtonWidgetClass, "_context");
    viewIvar = class_getInstanceVariable(OpenPanelButtonWidgetClass, "_view");
}
