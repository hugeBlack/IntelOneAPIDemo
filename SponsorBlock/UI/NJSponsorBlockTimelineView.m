//
//  NJSponsorBlockTimelineView.m
//  SponsorBlock
//
//  Created by s s on 2026/5/9.
//

#include "NJSponsorBlockTimelineView.h"
#include "../Settings/NJCommonDefine.h"
#include "../Settings/NJSponsorBlockSettings.h"

static void *NJSponsorBlockNativeTimelineKey = &NJSponsorBlockNativeTimelineKey;
static NSHashTable<NJSponsorBlockTimelineView *> *NJSponsorBlockNativeTimelineViews;

@implementation NJSponsorBlockTimelineView

- (instancetype)initWithFrame:(CGRect)frame manager:(NJSponsorBlockManager*)manager {
    self = [super initWithFrame:frame];
    if (self) {
        _manager = manager;
        _segmentMarkViews = [NSMutableArray array];
        self.userInteractionEnabled = NO;
        self.clipsToBounds = YES;
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(reload)
                                                     name:NJSponsorBlockStateDidChangeNotification
                                                   object:manager];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(onPlaybackTimeChanged)
                                                     name:NJSponsorBlockPlaybackTimeDidChangeNotification
                                                   object:manager];
    }
    return self;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)onPlaybackTimeChanged {
    NJSponsorBlockManager *manager = _manager;
    [self updatePlayhead:manager.currentPlaybackTime];
}

- (void)reload {
    [self.subviews makeObjectsPerformSelector:@selector(removeFromSuperview)];
    [_segmentMarkViews removeAllObjects];
    _playheadView = nil;
    NJSponsorBlockManager *manager = _manager;
    self.segments = [manager displaySegments];
    self.duration = manager.estimatedVideoDuration;
    self.currentPlaybackTime = manager.currentPlaybackTime;

    if (self.duration <= 0 || self.segments.count == 0) {
        return;
    }

    for (NJSponsorBlockSegment *segment in self.segments) {
        UIView *mark = [[UIView alloc] initWithFrame:CGRectZero];
        mark.backgroundColor = segment.isUnsubmitted
            ? [[NJSponsorBlockSettings colorForCategory:segment.category] colorWithAlphaComponent:0.48]
            : [NJSponsorBlockSettings colorForCategory:segment.category];
        if (segment.isUnsubmitted) {
            mark.layer.borderWidth = 1.0;
            mark.layer.borderColor = [UIColor colorWithWhite:1 alpha:0.75].CGColor;
        }
        [_segmentMarkViews addObject:mark];
        [self addSubview:mark];
    }

    _playheadView = [[UIView alloc] initWithFrame:CGRectZero];
    _playheadView.backgroundColor = UIColor.whiteColor;
    [self addSubview:_playheadView];

    [self setNeedsLayout];
}

- (void)layoutSubviews {
    [super layoutSubviews];

    CGFloat width = CGRectGetWidth(self.bounds);
    CGFloat height = CGRectGetHeight(self.bounds);
    
    NSTimeInterval duration = self.duration;
    NSArray<NJSponsorBlockSegment *> *segments = self.segments;

    if (duration <= 0 || segments.count == 0) {
        return;
    }

    for (NSUInteger i = 0; i < _segmentMarkViews.count && i < segments.count; i++) {
        NJSponsorBlockSegment *segment = segments[i];
        UIView *mark = _segmentMarkViews[i];
        CGFloat startX = MAX(0, MIN(width, width * segment.startTime / duration));
        CGFloat endX = MAX(startX + 2.0, MIN(width, width * segment.endTime / duration));
        mark.frame = CGRectMake(startX, 0, endX - startX, height);
    }

    if (_playheadView) {
        CGFloat playheadX = MAX(0, MIN(width, width * self.currentPlaybackTime / duration));
        _playheadView.frame = CGRectMake(playheadX - 1.0, 0, 2.0, height);
    }
}

- (void)updatePlayhead:(NSTimeInterval)time {
    if (!_playheadView || self.duration <= 0) {
        return;
    }
    self.currentPlaybackTime = time;
    CGFloat width = CGRectGetWidth(self.bounds);
    CGFloat height = CGRectGetHeight(self.bounds);
    if (width <= 0 || height <= 0) {
        return;
    }
    CGFloat playheadX = MAX(0, MIN(width, width * time / self.duration));
    _playheadView.frame = CGRectMake(playheadX - 1.0, 0, 2.0, height);
}

+ (void)installNativeTimelineInView:(UIView *)view manager:(NJSponsorBlockManager*)manager {
    if (!view || !NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    if (!NJSponsorBlockNativeTimelineViews) {
        NJSponsorBlockNativeTimelineViews = [NSHashTable weakObjectsHashTable];
    }

    NJSponsorBlockTimelineView *timeline = objc_getAssociatedObject(view, NJSponsorBlockNativeTimelineKey);
    if (!timeline) {
        timeline = [[NJSponsorBlockTimelineView alloc] initWithFrame:CGRectZero manager:manager];
        timeline.backgroundColor = UIColor.clearColor;
        objc_setAssociatedObject(view, NJSponsorBlockNativeTimelineKey, timeline, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [NJSponsorBlockNativeTimelineViews addObject:timeline];
    }

    if (timeline.superview != view) {
        [timeline removeFromSuperview];
        [view addSubview:timeline];
        [NSLayoutConstraint activateConstraints:@[
            [timeline.leftAnchor constraintEqualToAnchor:[view leftAnchor]],
            [timeline.rightAnchor constraintEqualToAnchor:[view rightAnchor]],
            [timeline.heightAnchor constraintEqualToAnchor:[view heightAnchor]]
        ]];
        timeline.translatesAutoresizingMaskIntoConstraints = NO;
        NSLog(@"[NJSponsorBlock] native timeline installed in %@ frame=%@", view, NSStringFromCGRect(view.frame));
        [view bringSubviewToFront:timeline];
        [timeline reload];
    }
}

@end
