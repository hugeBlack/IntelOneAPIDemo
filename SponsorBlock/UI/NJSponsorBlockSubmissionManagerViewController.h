//
//  NJSponsorBlockSubmissionManagerViewController.h
//  SponsorBlock
//

#import <UIKit/UIKit.h>
#import "../Services/NJSponsorBlockManager.h"

NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockSubmissionManagerViewController : UITableViewController
- (instancetype)initWithManager:(NJSponsorBlockManager*)manager;
@end

NS_ASSUME_NONNULL_END
