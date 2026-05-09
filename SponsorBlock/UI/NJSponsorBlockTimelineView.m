//
//  NJSponsorBlockTimelineView.m
//  SponsorBlock
//
//  Created by s s on 2026/5/9.
//

#include "NJSponsorBlockTimelineView.h"
#include "../Services/NJSponsorBlockManager.h"
#include "../Settings/NJCommonDefine.h"
#include "../Settings/NJSponsorBlockSettings.h"

static void *NJSponsorBlockNativeTimelineKey = &NJSponsorBlockNativeTimelineKey;
static NSHashTable<NJSponsorBlockTimelineView *> *NJSponsorBlockNativeTimelineViews;

@implementation NJSponsorBlockTimelineView

- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        _segmentMarkViews = [NSMutableArray array];
        self.userInteractionEnabled = NO;
        self.clipsToBounds = YES;
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(reload)
                                                     name:NJSponsorBlockStateDidChangeNotification
                                                   object:nil];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(onPlaybackTimeChanged)
                                                     name:NJSponsorBlockPlaybackTimeDidChangeNotification
                                                   object:nil];
    }
    return self;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)onPlaybackTimeChanged {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    [self updatePlayhead:manager.currentPlaybackTime];
}

- (void)reload {
    [self.subviews makeObjectsPerformSelector:@selector(removeFromSuperview)];
    [_segmentMarkViews removeAllObjects];
    _playheadView = nil;
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
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

    CGFloat width = CGRectGetWidth(self.superview.bounds);
    CGFloat height = CGRectGetHeight(self.superview.bounds);
    
    CGFloat timelineHeight = MIN(4.0, MAX(2.0, height));
    CGFloat y = MAX(0, (height - timelineHeight) * 0.5);
    self.frame = CGRectMake(0, y, width, timelineHeight);
    
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

+ (void)installNativeTimelineInView:(UIView *)view {
    if (!view || !NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    if (!NJSponsorBlockNativeTimelineViews) {
        NJSponsorBlockNativeTimelineViews = [NSHashTable weakObjectsHashTable];
    }

    NJSponsorBlockTimelineView *timeline = objc_getAssociatedObject(view, NJSponsorBlockNativeTimelineKey);
    if (!timeline) {
        timeline = [[NJSponsorBlockTimelineView alloc] initWithFrame:CGRectZero];
        timeline.backgroundColor = UIColor.clearColor;
        objc_setAssociatedObject(view, NJSponsorBlockNativeTimelineKey, timeline, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [NJSponsorBlockNativeTimelineViews addObject:timeline];
    }

    if (timeline.superview != view) {
        [timeline removeFromSuperview];
        [view addSubview:timeline];
        NSLog(@"[NJSponsorBlock] native timeline installed in %@ frame=%@", view, NSStringFromCGRect(view.frame));
        [view bringSubviewToFront:timeline];
        [timeline reload];
    }
}

@end
