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

extern void* sponsorBlockManagerKey;

@interface BBPlayerControlContainerWidgetView : UIView
@end

@interface BBPlayerSeekbarContainerView : UIView
@end

@interface BBPlayerFeatureWidgetService : NSObject
-(void)pushWidget:(id)arg0;
@end

@class BBPlayerContext;
@interface BBPlayerObject : NSObject
@property (readonly, weak, nonatomic) BBPlayerContext *context;
@end

@interface BBPlayerPlayback : BBPlayerObject
@property (nonatomic) NSTimeInterval currentTime;
@property (retain, nonatomic) id currentItem;
-(void)seekTo:(NSTimeInterval)arg0 ;
@end

@interface BBPlayerContext : NSObject
@property (readonly, weak, nonatomic) BBPlayerPlayback *playback;
@property (readonly, weak, nonatomic) BBPlayerFeatureWidgetService* featureWidgetService;
@end



@interface BBPlayerWidget : BBPlayerObject
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
void initPlayerContextHooks(void);
void initPlayerPlaybackHooks(void);
