//
//  Tweaks.h
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "../NJSettingCache.h"
@import ObjectiveC;
@import Foundation;
@import UIKit;

BOOL JRSwizzleInstanceMethod(Class targetClass, SEL selector, IMP newIMP, IMP *origIMPPtr);
void swizzle(Class class, SEL originalAction, SEL swizzledAction);



@interface BBPlayerControlContainerWidgetView : UIView
@end

@interface BBPlayerSeekbarContainerView : UIView
@end

@interface BBPlayerWidget : NSObject
@property UIView* view;
@property (readonly, weak, nonatomic) BBPlayerWidget *superWidget;
@property (readonly, copy, nonatomic) NSArray *subWidgets;
- (void)addSubWidget:(BBPlayerWidget *)subWidget;
- (void)willLayoutSubWidgets;
@end

@interface BBPlayerCastBtnWidget : BBPlayerWidget
- (instancetype)initWithContext:(id)context;
@end

void initPlayerWidgetButtonHooks(void);
void initViewReplyHooks(void);
void initSettingsHooks(void);
void initSeekbarHooks(void);
void initThumbnailBadgeHooks(void);
