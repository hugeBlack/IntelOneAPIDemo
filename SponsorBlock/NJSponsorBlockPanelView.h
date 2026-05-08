//
//  NJSponsorBlockPanelView.h
//  BiliBiliMDDylib
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockPanelView : UIView

+ (instancetype)sharedPanel;
+ (UIButton *)sharedEntryButton;
+ (nullable UIView *)currentHostView;
+ (void)installNativeTimelineInView:(UIView *)view;
+ (void)installInView:(UIView *)view;
+ (void)removePanel;
+ (void)hidePanelOnly;
+ (void)refresh;

@end

NS_ASSUME_NONNULL_END
