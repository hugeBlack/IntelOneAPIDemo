//
//  NJSponsorBlockPanelView.h
//  BiliBiliMDDylib
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockPanelView : UIView

+ (instancetype)sharedPanel;
+ (void)removePanel;
+ (void)hidePanelOnly;
+ (void)refresh;
- (void)refreshContent;
@end

NS_ASSUME_NONNULL_END
