//
//  Tweak.m
//  nmsl
//
//  Created by s s on 2026/5/2.
//
@import UIKit;
@import ObjectiveC;
#include "NJCommonDefine.h"
#include "NJSettingCache.h"
#include "NJSponsorBlockSettings.h"
#include "NJSponsorBlockSettingViewController.h"
#import "NJSponsorBlockPanelView.h"
#import "NJSponsorBlockManager.h"
#include "NJSettingDefine.h"

@interface BBPlayerControlContainerWidgetView : UIView
@end

@interface BBPlayerSeekbarContainerView : UIView
@end

@interface BBPlayerWidget : NSObject

@property (readonly, weak, nonatomic) BBPlayerWidget *superWidget;
@property (readonly, copy, nonatomic) NSArray *subWidgets;

@end

static UIView *NJSponsorBlockFindDirectTopRightButtonContainer(UIView *controlContainer) {
    
    NSArray<UIView*>* subViews1 = [controlContainer subviews];
    if([subViews1 count] < 1) {
        return nil;
    }
    UIView* subView1 = subViews1[0];
    UIView* subView2 = nil;
    NSArray<UIView*>* subViews2 = [subView1 subviews];
    for(UIView* view in subViews2) {
        if(view.frame.size.height != 44) {
            continue;
        }
        subView2 = view;
        break;
    }
    if (!subView2) {
        return nil;
    }
    
    NSArray<UIView*>* subViews3 = [subView2 subviews];
    if([subViews3 count] < 1) {
        return nil;
    }
    UIView* subView3 = subViews3[0];
    
    NSArray<UIView*>* subViews4 = [subView3 subviews];
    if([subViews4 count] < 2) {
        return nil;
    }
    UIView* toolBarView = subViews4[0];

    return toolBarView;
}

static BOOL NJSponsorBlockInstallDirectTopEntryFromControlContainer(UIView *controlContainer) {
    if (!controlContainer || !NJ_MASTER_SWITCH_VALUE) {
        return NO;
    }

    UIView *targetContainer = NJSponsorBlockFindDirectTopRightButtonContainer(controlContainer);
    if (!targetContainer) {
        return NO;
    }

    [NJSponsorBlockPanelView installEntryDirectlyInContainer:targetContainer];
    return YES;
}

static BOOL NJSponsorBlockWidgetLooksLikeSeekbar(id widget) {
    NSString *className = NSStringFromClass([widget class]);
    return [className rangeOfString:@"Seekbar" options:NSCaseInsensitiveSearch].location != NSNotFound
        || [className rangeOfString:@"Slider" options:NSCaseInsensitiveSearch].location != NSNotFound
        || [className rangeOfString:@"Progress" options:NSCaseInsensitiveSearch].location != NSNotFound;
}

static BOOL NJSponsorBlockViewLooksLikeSeekbarHost(UIView *view) {
    if (!view || view.hidden || view.alpha <= 0.01) {
        return NO;
    }
    CGRect bounds = view.bounds;
    CGFloat width = CGRectGetWidth(bounds);
    CGFloat height = CGRectGetHeight(bounds);
    if (width < 80.0 || height <= 0.0 || height > 64.0) {
        return NO;
    }
    return YES;
}

static UIView *NJSponsorBlockSeekbarHostFromSelector(id widget, SEL selector) {
    if (![widget respondsToSelector:selector]) {
        return nil;
    }
    id value = nil;
    @try {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
        value = [widget performSelector:selector];
#pragma clang diagnostic pop
    } @catch (__unused NSException *exception) {
        return nil;
    }
    if ([value isKindOfClass:UIView.class] && NJSponsorBlockViewLooksLikeSeekbarHost(value)) {
        return value;
    }
    return nil;
}

static UIView *NJSponsorBlockFindSeekbarHostView(id widget, NSInteger depth) {
    if (!widget || depth > 1) {
        return nil;
    }
    if ([widget isKindOfClass:UIView.class] && NJSponsorBlockViewLooksLikeSeekbarHost((UIView *)widget)) {
        return (UIView *)widget;
    }

    SEL selectors[] = {
        sel_registerName("view"),
        sel_registerName("contentView"),
        sel_registerName("containerView"),
        sel_registerName("progressView"),
        sel_registerName("slider"),
        sel_registerName("seekbarView"),
    };
    for (NSUInteger i = 0; i < sizeof(selectors) / sizeof(SEL); i++) {
        UIView *view = NJSponsorBlockSeekbarHostFromSelector(widget, selectors[i]);
        if (view) {
            return view;
        }
    }
    return nil;
}

static void NJSponsorBlockInstallSeekbarTimelinesFromWidget(BBPlayerWidget *widget) {
    static NSMutableSet<NSString *> *loggedMissingHostClasses = nil;
    if (!loggedMissingHostClasses) {
        loggedMissingHostClasses = [NSMutableSet set];
    }

    NSArray *subWidgets = [widget subWidgets];
    for (id subWidget in subWidgets) {
        if (NJSponsorBlockWidgetLooksLikeSeekbar(subWidget)) {
            UIView *hostView = NJSponsorBlockFindSeekbarHostView(subWidget, 0);
            if (hostView) {
                [NJSponsorBlockPanelView installNativeTimelineInView:hostView];
            } else {
                NSString *className = NSStringFromClass([subWidget class]);
                if (![loggedMissingHostClasses containsObject:className]) {
                    [loggedMissingHostClasses addObject:className];
                    NSLog(@"[NJSponsorBlock] seekbar host not found for %@", className);
                }
            }
        }
        if ([subWidget respondsToSelector:@selector(subWidgets)]) {
            NJSponsorBlockInstallSeekbarTimelinesFromWidget(subWidget);
        }
    }
}

// hooks start
/**
 Hook (替换或重载) 一个类的实例方法
 如果该类没有实现该方法但父类有实现，会自动为该类添加一个该方法 (相当于重载/super调用版本)
 
 @param targetClass 要 hook 的目标类
 @param selector 要 hook 的方法选择器
 @param newIMP 新的函数实现
 @param origIMPPtr 传出原始的 IMP (如果不为 NULL)
 @return 是否成功
 */
BOOL JRSwizzleInstanceMethod(Class targetClass, SEL selector, IMP newIMP, IMP *origIMPPtr) {
    if (!targetClass || !selector || !newIMP || !origIMPPtr) {
        return NO;
    }
    
    // 1. 在类层级中查找原始方法（会顺着继承链向上找）
    Method originalMethod = class_getInstanceMethod(targetClass, selector);
    if (!originalMethod) {
        NSLog(@"[%s] Method not found: %@", class_getName(targetClass), NSStringFromSelector(selector));
        return NO;
    }
    
    // 2. 保存原始的 IMP
    *origIMPPtr = method_getImplementation(originalMethod);
    
    // 3. 核心魔法：尝试将原始方法添加到目标类中
    // - 如果目标类已经实现了该方法，class_addMethod 会返回 NO，什么也不做。
    // - 如果目标类没有实现（是继承父类的），这里会将其"复制"一份到目标类的方法列表中。
    // 这样做保证了我们后续的替换只影响当前 targetClass，不会污染父类。
    class_addMethod(targetClass,
                    selector,
                    *origIMPPtr,
                    method_getTypeEncoding(originalMethod));
    
    // 4. 获取现在肯定存在于 targetClass 中的本地方法
    Method localMethod = class_getInstanceMethod(targetClass, selector);
    
    // 5. 将本地方法的实现替换为我们的 newIMP
    method_setImplementation(localMethod, newIMP);
    
    return YES;
}

void (*orig_BBPlayerFlexContainerWidget_didLayoutSubWidgets)(id self, SEL sel) = nil;
void hook_BBPlayerFlexContainerWidget_didLayoutSubWidgets(id self, SEL sel) {
//    NSLog(@"%@:%@-%p-%s-subWidgets:%@", nj_logPrefix, NSStringFromClass([(id)self class]), self, __FUNCTION__, [self subWidgets]);
    orig_BBPlayerFlexContainerWidget_didLayoutSubWidgets(self, sel);
    NJSponsorBlockInstallSeekbarTimelinesFromWidget(self);
}

void (*orig_BBPlayerControlContainerWidgetView_layoutSubviews)(id self, SEL sel) = nil;
void hook_BBPlayerControlContainerWidgetView_layoutSubviews(id self, SEL sel) {
    orig_BBPlayerControlContainerWidgetView_layoutSubviews(self, sel);
    UIView *controlContainer = (UIView *)self;
    dispatch_time_t delay = dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC));

    dispatch_after(delay, dispatch_get_main_queue(), ^{
        NJSponsorBlockInstallDirectTopEntryFromControlContainer(controlContainer);
    });

//    NJSponsorBlockScheduleTopEntryInstall(controlContainer);
}

//void (*orig_BBPlayerControlContainerWidgetView_didMoveToWindow)(id self, SEL sel) = nil;
//void hook_BBPlayerControlContainerWidgetView_didMoveToWindow(id self, SEL sel) {
//    orig_BBPlayerControlContainerWidgetView_didMoveToWindow(self, sel);
//    UIView *controlContainer = (UIView *)self;
//    NJSponsorBlockScheduleTopEntryInstall(controlContainer);
//}

void hook_BBPlayerFlexContainerWidget_viewWillDisappear(id self, SEL sel, bool animated) {
    [NJSponsorBlockPanelView removePanel];
}

void (*orig_BBPlayerSeekbarContainerView_layoutSubviews)(id self, SEL sel) = nil;
void hook_BBPlayerSeekbarContainerView_layoutSubviews(id self, SEL sel) {
    [NJSponsorBlockPanelView installNativeTimelineInView:(UIView *)self];
    orig_BBPlayerSeekbarContainerView_layoutSubviews(self, sel);
}

id (*orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    id ret = orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(self, sel, data, registry, error);
    [[NJSponsorBlockManager sharedInstance] inspectModelObject:ret source:@"BAPIAppViewuniteV1ViewReply"];
    return ret;
}


// settings hooks

@interface NJSettingSkullViewModel : NSObject

/// 业务id
@property (nonatomic, copy) NSString *bizId;
/// cell重利用Id
@property (nonatomic, copy) NSString *cellId;
/// 标题
@property (nonatomic, copy) NSString *title;
/// 副标题
@property (nonatomic, copy) NSString *subTitle;


- (instancetype)initWithBizId:(NSString *)bizId
                       cellId:(NSString *)cellId
                        title:(NSString *)title;

@end

@interface NJSettingBizHandler : NSObject

/// 设置vc
@property (nonatomic, weak) UIViewController *settingViewController;

/// 处理业务
/// - Parameter viewModel: 数据
- (void)handleBizWithViewModel:(NJSettingSkullViewModel *)viewModel;
@end


void (*orig_NJSettingBizHandler_handleBizWithViewModel)(NJSettingBizHandler* self, SEL sel, NJSettingSkullViewModel* viewModel) = nil;
void hook_NJSettingBizHandler_handleBizWithViewModel(NJSettingBizHandler* self, SEL sel, NJSettingSkullViewModel* viewModel) {
    if ([[viewModel bizId] isEqualToString:NJ_SPONSOR_BLOCK_SETTING_PAGE_BIZ_ID]) {
        NJSponsorBlockSettingViewController *settingVC = [[NJSponsorBlockSettingViewController alloc] init];
        [self.settingViewController.navigationController pushViewController:settingVC animated:YES];
        return;
    }
    orig_NJSettingBizHandler_handleBizWithViewModel(self, sel, viewModel);
}

@interface NJSettingInjectDataProvider : NSObject

/// 要注入的数据
- (NSArray<NJSettingSkullViewModel *> *)injectDatas;

@end

NSArray<NJSettingSkullViewModel *>* (*orig_NJSettingInjectDataProvider_injectDatas)(NJSettingInjectDataProvider* self, SEL sel) = nil;
NSArray<NJSettingSkullViewModel *>* hook_NJSettingInjectDataProvider_injectDatas(NJSettingInjectDataProvider* self, SEL sel) {
    NSMutableArray* datas = [orig_NJSettingInjectDataProvider_injectDatas(self, sel) mutableCopy];
    
    NJSettingSkullViewModel *model = [[PrivClass(NJSettingSkullViewModel) alloc] initWithBizId:NJ_SPONSOR_BLOCK_SETTING_PAGE_BIZ_ID
                                                                                 cellId:NJ_ARROW_CELL_ID
                                                                                  title:@"SponsorBlock"];
    model.subTitle = [NJSponsorBlockSettings enabled] ? @"已启用" : @"已关闭";

    
    [datas insertObject:model atIndex:datas.count - 2];
    return [datas copy];
}


__attribute__((constructor)) void TweakInit(void) {
    NSLog(@"SposorBlock loaded.");
    
    JRSwizzleInstanceMethod(objc_getClass("BBPlayerFlexContainerWidget"), @selector(didLayoutSubWidgets),
                            (IMP)hook_BBPlayerFlexContainerWidget_didLayoutSubWidgets,
                            (IMP*)&orig_BBPlayerFlexContainerWidget_didLayoutSubWidgets);
    
    class_addMethod(objc_getClass("BBPlayerControlContainerWidgetView"),
                    @selector(viewWillDisappear:),
                    (IMP)hook_BBPlayerFlexContainerWidget_viewWillDisappear,
                    "v@:B");
    
    JRSwizzleInstanceMethod(objc_getClass("BBPlayerControlContainerWidgetView"), @selector(layoutSubviews),
                            (IMP)hook_BBPlayerControlContainerWidgetView_layoutSubviews,
                            (IMP*)&orig_BBPlayerControlContainerWidgetView_layoutSubviews
                            );
    
//    JRSwizzleInstanceMethod(objc_getClass("BBPlayerControlContainerWidgetView"), @selector(didMoveToWindow),
//                            (IMP)hook_BBPlayerControlContainerWidgetView_didMoveToWindow,
//                            (IMP*)&orig_BBPlayerControlContainerWidgetView_didMoveToWindow
//                            );
                            
    JRSwizzleInstanceMethod(objc_getClass("BBPlayerSeekbarContainerView"), @selector(layoutSubviews),
                            (IMP)hook_BBPlayerSeekbarContainerView_layoutSubviews,
                            (IMP*)&orig_BBPlayerSeekbarContainerView_layoutSubviews);
    
    JRSwizzleInstanceMethod(objc_getClass("BAPIAppViewuniteV1ViewReply"), @selector(initWithData:extensionRegistry:error:),
                            (IMP)hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error,
                            (IMP*)&orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error);
    
    
    // settings stuff
    
    JRSwizzleInstanceMethod(objc_getClass("NJSettingBizHandler"), @selector(handleBizWithViewModel:),
                            (IMP)hook_NJSettingBizHandler_handleBizWithViewModel,
                            (IMP*)&orig_NJSettingBizHandler_handleBizWithViewModel);
    
    JRSwizzleInstanceMethod(objc_getClass("NJSettingInjectDataProvider"), @selector(injectDatas),
                            (IMP)hook_NJSettingInjectDataProvider_injectDatas,
                            (IMP*)&orig_NJSettingInjectDataProvider_injectDatas);
    
}
