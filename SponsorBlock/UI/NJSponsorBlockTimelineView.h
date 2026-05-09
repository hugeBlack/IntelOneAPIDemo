//
//  NJSponsorBlockTimelineView.h
//  SponsorBlock
//
//  Created by s s on 2026/5/9.
//
@import UIKit;
#include "../Models/NJSponsorBlockSegment.h"

@interface NJSponsorBlockTimelineView : UIView {
    NSMutableArray<UIView *> *_segmentMarkViews;
    UIView *_playheadView;
}
@property (nonatomic, copy) NSArray<NJSponsorBlockSegment *> *segments;
@property (nonatomic, assign) NSTimeInterval duration;
@property (nonatomic, assign) NSTimeInterval currentPlaybackTime;
- (void)reload;
+ (void)installNativeTimelineInView:(UIView *)view;
@end
