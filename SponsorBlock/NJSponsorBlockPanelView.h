//
//  NJSponsorBlockPanelView.h
//  BiliBiliMDDylib
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockPanelView : UIView

+ (instancetype)sharedPanel;
+ (nullable UIView *)currentHostView;
+ (void)installEntryInView:(UIView *)view;
+ (void)installEntryDirectlyInContainer:(UIView *)container;
+ (void)setEntryAnchorView:(nullable UIView *)view;
+ (void)installNativeTimelineInView:(UIView *)view;
+ (void)installInView:(UIView *)view;
+ (void)markPlaybackActive;
+ (void)hideOverlay;
+ (void)removePanel;
+ (void)refresh;

@end

NS_ASSUME_NONNULL_END
