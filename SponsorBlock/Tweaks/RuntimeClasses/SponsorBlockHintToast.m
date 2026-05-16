//
//  SponsorBlockHintToast.m
//  SponsorBlock
//
//  Created by s s on 2026/5/16.
//

#include "../Tweaks.h"
#include "SponsorBlockHintToast.h"
@import ObjectiveC;
@import UIKit;

static Ivar viewIvar;

// Associated object keys for per-instance configuration
static void *NJSBToastTitleKey        = &NJSBToastTitleKey;
static void *NJSBToastDetailKey       = &NJSBToastDetailKey;
static void *NJSBToastActionTitleKey  = &NJSBToastActionTitleKey;
static void *NJSBToastSecTitleKey     = &NJSBToastSecTitleKey;
static void *NJSBToastActionBlockKey  = &NJSBToastActionBlockKey;
static void *NJSBToastSecBlockKey     = &NJSBToastSecBlockKey;
static void *NJSBToastCloseBlockKey   = &NJSBToastCloseBlockKey;
static void *NJSBToastContextKey      = &NJSBToastContextKey;
static void *NJSBToastShowCloseKey    = &NJSBToastShowCloseKey;

// ── view override ──────────────────────────────────────────────────────────

static UIView *SponsorBlockHintToast_view(id self, SEL _cmd) {
    UIView *cached = object_getIvar(self, viewIvar);
    if (cached) return cached;

    NSString *title       = objc_getAssociatedObject(self, NJSBToastTitleKey)       ?: @"";
    NSString *detail      = objc_getAssociatedObject(self, NJSBToastDetailKey)      ?: @"";
    NSString *actionTitle = objc_getAssociatedObject(self, NJSBToastActionTitleKey);
    NSString *secTitle    = objc_getAssociatedObject(self, NJSBToastSecTitleKey);

    // Container
    UIView *container = [[UIView alloc] init];
    container.backgroundColor = [UIColor colorWithRed:0.04 green:0.14 blue:0.20 alpha:0.96];
    container.layer.cornerRadius = 10;
    container.layer.masksToBounds = YES;

    // Labels
    UILabel *titleLabel = [[UILabel alloc] init];
    titleLabel.text = title;
    titleLabel.textColor = UIColor.whiteColor;
    titleLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightBold];

    UILabel *detailLabel = [[UILabel alloc] init];
    detailLabel.text = detail;
    detailLabel.textColor = [UIColor colorWithWhite:0.78 alpha:1.0];
    detailLabel.font = [UIFont systemFontOfSize:11 weight:UIFontWeightRegular];
    detailLabel.lineBreakMode = NSLineBreakByTruncatingTail;

    UIStackView *textStack = [[UIStackView alloc] initWithArrangedSubviews:@[titleLabel, detailLabel]];
    textStack.axis = UILayoutConstraintAxisVertical;
    textStack.spacing = 2;

    // Buttons
    NSMutableArray *buttonViews = [NSMutableArray array];

    if (actionTitle.length > 0) {
        UIButton *btn = [UIButton buttonWithType:UIButtonTypeSystem];
        [btn setTitle:actionTitle forState:UIControlStateNormal];
        [btn setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
        btn.titleLabel.font = [UIFont systemFontOfSize:12 weight:UIFontWeightBold];
        btn.backgroundColor = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95];
        btn.layer.cornerRadius = 6;
        [btn.widthAnchor constraintGreaterThanOrEqualToConstant:52].active = YES;
        [btn.heightAnchor constraintEqualToConstant:28].active = YES;
        [btn addTarget:self action:@selector(njsbToastActionTapped) forControlEvents:UIControlEventTouchUpInside];
        [buttonViews addObject:btn];
    }

    if (secTitle.length > 0) {
        UIButton *btn = [UIButton buttonWithType:UIButtonTypeSystem];
        [btn setTitle:secTitle forState:UIControlStateNormal];
        [btn setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
        btn.titleLabel.font = [UIFont systemFontOfSize:12 weight:UIFontWeightBold];
        btn.backgroundColor = [UIColor colorWithWhite:0.28 alpha:0.95];
        btn.layer.cornerRadius = 6;
        [btn.widthAnchor constraintGreaterThanOrEqualToConstant:58].active = YES;
        [btn.heightAnchor constraintEqualToConstant:28].active = YES;
        [btn addTarget:self action:@selector(njsbToastSecondaryTapped) forControlEvents:UIControlEventTouchUpInside];
        [buttonViews addObject:btn];
    }

    NSNumber *showCloseNum = objc_getAssociatedObject(self, NJSBToastShowCloseKey);
    BOOL showClose = showCloseNum ? showCloseNum.boolValue : YES;
    if (showClose) {
        UIButton *closeBtn = [UIButton buttonWithType:UIButtonTypeSystem];
        [closeBtn setTitle:@"×" forState:UIControlStateNormal];
        [closeBtn setTitleColor:[UIColor colorWithWhite:0.72 alpha:1] forState:UIControlStateNormal];
        closeBtn.titleLabel.font = [UIFont systemFontOfSize:16 weight:UIFontWeightRegular];
        [closeBtn.widthAnchor constraintEqualToConstant:24].active = YES;
        [closeBtn.heightAnchor constraintEqualToConstant:28].active = YES;
        [closeBtn addTarget:self action:@selector(njsbToastCloseTapped) forControlEvents:UIControlEventTouchUpInside];
        [buttonViews addObject:closeBtn];
    }

    UIStackView *buttonStack = [[UIStackView alloc] initWithArrangedSubviews:buttonViews];
    buttonStack.axis = UILayoutConstraintAxisHorizontal;
    buttonStack.alignment = UIStackViewAlignmentCenter;
    buttonStack.spacing = 5;

    UIStackView *mainStack = [[UIStackView alloc] initWithArrangedSubviews:@[textStack, buttonStack]];
    mainStack.axis = UILayoutConstraintAxisHorizontal;
    mainStack.alignment = UIStackViewAlignmentCenter;
    mainStack.spacing = 8;
    mainStack.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:mainStack];

    [NSLayoutConstraint activateConstraints:@[
        [mainStack.leadingAnchor constraintEqualToAnchor:container.leadingAnchor constant:12],
        [mainStack.trailingAnchor constraintEqualToAnchor:container.trailingAnchor constant:-8],
        [mainStack.topAnchor constraintEqualToAnchor:container.topAnchor constant:10],
        [mainStack.bottomAnchor constraintEqualToAnchor:container.bottomAnchor constant:-10],
    ]];

    container.frame = CGRectMake(0, 0, 200, 100);
    object_setIvar(self, viewIvar, container);
    return container;
}

// ── button target IMPs ─────────────────────────────────────────────────────

static void NJSBDismissSelf(id self) {
    BBPlayerContext *ctx = objc_getAssociatedObject(self, NJSBToastContextKey);
    if (ctx) {
        [ctx.toastWidgetService dismissToast:self];
    }
}

static void SponsorBlockHintToast_actionTapped(id self, SEL _cmd) {
    void (^block)(void) = objc_getAssociatedObject(self, NJSBToastActionBlockKey);
    if (block) block();
    NJSBDismissSelf(self);
}

static void SponsorBlockHintToast_secondaryTapped(id self, SEL _cmd) {
    void (^block)(void) = objc_getAssociatedObject(self, NJSBToastSecBlockKey);
    if (block) block();
    NJSBDismissSelf(self);
}

static void SponsorBlockHintToast_closeTapped(id self, SEL _cmd) {
    void (^block)(void) = objc_getAssociatedObject(self, NJSBToastCloseBlockKey);
    if (block) block();
    NJSBDismissSelf(self);
}

// ── registration ───────────────────────────────────────────────────────────

void registerSponsorBlockHintToast(void) {
    Class base = PrivClass(BBPlayerToastWidget);

    Class cls = objc_allocateClassPair(base, "SponsorBlockHintToast", 0);
    class_addMethod(cls, @selector(view),                     (IMP)SponsorBlockHintToast_view,          "@@:");
    class_addMethod(cls, @selector(njsbToastActionTapped),    (IMP)SponsorBlockHintToast_actionTapped,   "v@:");
    class_addMethod(cls, @selector(njsbToastSecondaryTapped), (IMP)SponsorBlockHintToast_secondaryTapped,"v@:");
    class_addMethod(cls, @selector(njsbToastCloseTapped),     (IMP)SponsorBlockHintToast_closeTapped,    "v@:");
    objc_registerClassPair(cls);
    viewIvar = class_getInstanceVariable(cls, "_view");
}

// ── public factory ─────────────────────────────────────────────────────────

id NJSponsorBlockCreateHintToast(
    BBPlayerContext *context,
    NSString *title,
    NSString *detail,
    NSString * _Nullable actionTitle,
    NSString * _Nullable secondaryTitle,
    void (^ _Nullable actionHandler)(void),
    void (^ _Nullable secondaryHandler)(void),
    void (^ _Nullable closeHandler)(void),
    NSTimeInterval duration,
    BOOL showCloseButton
) {
    Class cls = objc_lookUpClass("SponsorBlockHintToast");
    if (!cls || !context) return nil;
    id toast = [cls alloc];
    if (!toast) return nil;

    objc_setAssociatedObject(toast, NJSBToastTitleKey,       title,            OBJC_ASSOCIATION_COPY_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastDetailKey,      detail,           OBJC_ASSOCIATION_COPY_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastActionTitleKey, actionTitle,      OBJC_ASSOCIATION_COPY_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastSecTitleKey,    secondaryTitle,   OBJC_ASSOCIATION_COPY_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastActionBlockKey, actionHandler,    OBJC_ASSOCIATION_COPY_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastSecBlockKey,    secondaryHandler, OBJC_ASSOCIATION_COPY_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastCloseBlockKey,  closeHandler,     OBJC_ASSOCIATION_COPY_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastContextKey,     context,          OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    objc_setAssociatedObject(toast, NJSBToastShowCloseKey,   @(showCloseButton), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    toast =  [toast initWithContext:context];
    ((BBPlayerToastWidget *)toast).duration = duration;
    return toast;
}
