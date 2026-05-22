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
extern NSMutableDictionary* cachedCidVideoInfoDict;

@interface BBPlayerControlContainerWidgetView : UIView
@end

@interface BBPlayerSeekbarContainerView : UIView
@end

@interface BBPlayerFeatureWidgetService : NSObject
-(void)pushWidget:(id)arg0;
-(void)popWidget;
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

@interface BBPlayerToastWidgetService : NSObject
-(void)presentCustomToast:(id)arg0 ;
-(void)dismissToast:(id)arg0 ;
-(void)showToastContainerWithText:(NSString*)arg0 ;
@end

@interface BBPlayerContext : NSObject
@property (readonly, weak, nonatomic) BBPlayerPlayback *playback;
@property (readonly, weak, nonatomic) BBPlayerFeatureWidgetService* featureWidgetService;
@property (readonly, weak, nonatomic) BBPlayerToastWidgetService *toastWidgetService;

@end

@interface BBPlayerPlayItem: NSObject
-(NSInteger)cid;
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

@interface BBPlayerToastWidget : BBPlayerWidget

@property (retain, nonatomic) UILabel *hintLabel; // ivar: _hintLabel
@property (retain, nonatomic) UILabel *descLabel; // ivar: _descLabel
@property (retain, nonatomic) UIButton *closeButton; // ivar: _closeButton
@property (retain, nonatomic) UIButton *actionButton; // ivar: _actionButton
@property (retain, nonatomic) NSNumber *key; // ivar: _key
@property (readonly, nonatomic) NSUInteger style; // ivar: _style
@property (readonly, nonatomic) NSUInteger sizeMode; // ivar: _sizeMode
@property (nonatomic) NSInteger priority; // ivar: _priority
@property (nonatomic) CGFloat duration; // ivar: _duration
@property (readonly, nonatomic) BOOL dynamicHugging;


-(id)initWithContext:(id)arg0 ;
-(id)initWithContext:(id)arg0 style:(NSUInteger)arg1 ;
-(id)view;
-(void)sizeModeChangedTo:(NSUInteger)arg0 ;
-(void)setupToast;
-(NSInteger)compare:(id)arg0 ;
-(void)setupDefaultToast;
-(void)setupOperableToast;
-(void)setupOperableDescToast;

@end

void initPlayerWidgetButtonHooks(void);
void initViewReplyHooks(void);
void initSettingsHooks(void);
void initSeekbarHooks(void);
void initThumbnailBadgeHooks(void);
void initPlayerContextHooks(void);
void initPlayerPlaybackHooks(void);
