//
//  Tweaks.h
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include <Foundation/Foundation.h>
#include <UIKit/UIKit.h>
#include <objc/objc.h>
#include <objc/runtime.h>

#define PrivClass(name) ((Class)objc_lookUpClass(#name))

BOOL JRSwizzleInstanceMethod(Class targetClass, SEL selector, IMP newIMP, IMP *origIMPPtr);
void swizzle(Class clazz, SEL originalAction, SEL swizzledAction);



@interface BBPlayerControlContainerWidgetView : UIView
@end

@interface BBPlayerSeekbarContainerView : UIView
@end

@interface BBPlayerWidget : NSObject
@property UIView* view;
@property (readonly, weak, nonatomic) BBPlayerWidget *superWidget;
@property (readonly, copy, nonatomic) NSArray *subWidgets;
@property id context;
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
void NJSponsorBlockPlaybackHookInit(void);
