//
//  NJSponsorBlockColorPickerController.h
//  BiliBiliMDDylib
//

#import <UIKit/UIKit.h>

@protocol NJSponsorBlockColorPickerDelegate <NSObject>
- (void)colorPickerDidSelectColor:(UIColor *)color;
@end

@interface NJSponsorBlockColorPickerController : UIViewController

@property (nonatomic, weak) id<NJSponsorBlockColorPickerDelegate> delegate;

- (instancetype)initWithColor:(UIColor *)color categoryTitle:(NSString *)categoryTitle;

@end
