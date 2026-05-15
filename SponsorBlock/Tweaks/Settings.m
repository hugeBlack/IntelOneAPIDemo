//
//  Settings.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//

#include "Tweaks.h"
#include "../Settings/NJSettingDefine.h"
#include "../UI/NJSponsorBlockSettingViewController.h"
#include "../Settings/NJSponsorBlockSettings.h"

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

@interface FLEXManager : NSObject
+ (instancetype)sharedManager;
- (void)showExplorer;
@end

void (*orig_NJSettingBizHandler_handleBizWithViewModel)(NJSettingBizHandler* self, SEL sel, NJSettingSkullViewModel* viewModel) = nil;
void hook_NJSettingBizHandler_handleBizWithViewModel(NJSettingBizHandler* self, SEL sel, NJSettingSkullViewModel* viewModel) {
    if ([[viewModel bizId] isEqualToString:NJ_SPONSOR_BLOCK_SETTING_PAGE_BIZ_ID]) {
        NJSponsorBlockSettingViewController *settingVC = [[NJSponsorBlockSettingViewController alloc] init];
        [self.settingViewController.navigationController pushViewController:settingVC animated:YES];
        return;
    } else if ([[viewModel bizId] isEqualToString:NJ_OPEN_FLEX_BIZ_ID]) {
        [[PrivClass(FLEXManager) sharedManager] showExplorer];
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
    
    if(PrivClass(FLEXManager)) {
        NJSettingSkullViewModel *model2 = [[PrivClass(NJSettingSkullViewModel) alloc] initWithBizId:NJ_OPEN_FLEX_BIZ_ID
                                                                                     cellId:NJ_ARROW_CELL_ID
                                                                                      title:@"Open FLEX tool"];
        [datas addObject:model2];
    }
    
    return [datas copy];
}

void initSettingsHooks(void) {
    JRSwizzleInstanceMethod(objc_getClass("NJSettingBizHandler"), @selector(handleBizWithViewModel:),
                            (IMP)hook_NJSettingBizHandler_handleBizWithViewModel,
                            (IMP*)&orig_NJSettingBizHandler_handleBizWithViewModel);
    
    JRSwizzleInstanceMethod(objc_getClass("NJSettingInjectDataProvider"), @selector(injectDatas),
                            (IMP)hook_NJSettingInjectDataProvider_injectDatas,
                            (IMP*)&orig_NJSettingInjectDataProvider_injectDatas);
    
}
