//
//  NJSponsorBlockPanelView.h
//  BiliBiliMDDylib
//

#import <UIKit/UIKit.h>
#import "../Services/NJSponsorBlockManager.h"
NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockPanelView : UIView
- (instancetype)initWithManager:(NJSponsorBlockManager*)manager;
- (void)refreshContent;
@end

NS_ASSUME_NONNULL_END
