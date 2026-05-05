//
//  NJSponsorBlockPanelView.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockPanelView.h"
#import "NJCommonDefine.h"
#import "NJSettingCache.h"
#import "NJSponsorBlockManager.h"
#import "NJSponsorBlockSegment.h"
#import "NJSponsorBlockService.h"
#import "NJSponsorBlockSettings.h"
#import "NJSponsorBlockSubmissionManagerViewController.h"
#import <float.h>
#import <math.h>
#import <objc/runtime.h>

static CGFloat const NJSponsorBlockPanelWidth = 310.0;
static CGFloat const NJSponsorBlockPanelMinHeight = 154.0;
static CGFloat const NJSponsorBlockPanelMargin = 12.0;
static CGFloat const NJSponsorBlockPanelTopMargin = 48.0;
static CGFloat const NJSponsorBlockPanelBottomMargin = 24.0;
static NSTimeInterval const NJSponsorBlockOverlayIdleTimeout = 4.0;
static CGFloat const NJSponsorBlockSegmentRowHeight = 64.0;
static CGFloat const NJSponsorBlockSegmentRowSpacing = 6.0;
static CGFloat const NJSponsorBlockSegmentEmptyHeight = 30.0;
static CGFloat const NJSponsorBlockSegmentListMaxHeight = 250.0;
static CGFloat const NJSponsorBlockPanelChromeHeight = 134.0;

typedef NS_ENUM(NSInteger, NJSponsorBlockPanelNoticeMode) {
    NJSponsorBlockPanelNoticeModeNone = 0,
    NJSponsorBlockPanelNoticeModeAdvance,
    NJSponsorBlockPanelNoticeModeManualSkip,
    NJSponsorBlockPanelNoticeModeSkipped,
    NJSponsorBlockPanelNoticeModeMessage,
    NJSponsorBlockPanelNoticeModeSubmissionDraft,
};

@interface NJSponsorBlockOverlayWindow : UIWindow
@end

@implementation NJSponsorBlockOverlayWindow

- (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event {
    UIView *hitView = [super hitTest:point withEvent:event];
    if (hitView == self.rootViewController.view) {
        return nil;
    }
    return hitView;
}

@end

@interface NJSponsorBlockPanelView () <UIGestureRecognizerDelegate>

@property (nonatomic, strong) UIStackView *headerStack;
@property (nonatomic, strong) UIStackView *footerStack;
@property (nonatomic, strong) UILabel *iconLabel;
@property (nonatomic, strong) UILabel *titleLabel;
@property (nonatomic, strong) UILabel *subtitleLabel;
@property (nonatomic, strong) UIScrollView *segmentScrollView;
@property (nonatomic, strong) UIStackView *segmentStackView;
@property (nonatomic, strong) UIView *progressView;
@property (nonatomic, strong) UIView *noticeView;
@property (nonatomic, strong) UILabel *noticeTitleLabel;
@property (nonatomic, strong) UILabel *noticeDetailLabel;
@property (nonatomic, strong) UIButton *noticePrimaryButton;
@property (nonatomic, strong) UIButton *noticeSecondaryButton;
@property (nonatomic, strong) UIButton *noticeCloseButton;
@property (nonatomic, strong) UIButton *closeButton;
@property (nonatomic, strong) UIButton *toggleButton;
@property (nonatomic, strong) UIButton *submitButton;
@property (nonatomic, strong) UILabel *timeLabel;
@property (nonatomic, strong) UILabel *statsLabel;
@property (nonatomic, strong) NJSponsorBlockSegment *noticeSegment;
@property (nonatomic, copy) NSString *suppressedNoticeUUID;
@property (nonatomic, copy) NSString *submissionCategory;
@property (nonatomic, copy) NSString *submissionVideoID;
@property (nonatomic, strong) NJSponsorBlockService *service;
@property (nonatomic, strong) NSLayoutConstraint *noticeHeightConstraint;
@property (nonatomic, strong) NSLayoutConstraint *segmentScrollHeightConstraint;
@property (nonatomic, assign) NJSponsorBlockPanelNoticeMode noticeMode;
@property (nonatomic, assign) NSTimeInterval submissionStartTime;
@property (nonatomic, assign) NSTimeInterval submissionVideoDuration;
@property (nonatomic, assign) NSInteger submissionCID;
@property (nonatomic, assign) BOOL submissionInProgress;
@property (nonatomic, assign) BOOL submissionRequestInFlight;

- (UIButton *)noticeButtonWithTitle:(NSString *)title color:(UIColor *)color action:(SEL)action;
- (UIButton *)actionButtonWithTitle:(NSString *)title color:(UIColor *)color action:(SEL)action segment:(NJSponsorBlockSegment *)segment;
- (void)updateHeaderWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (void)updateEnabledButton:(BOOL)enabled;
- (void)updateFooterWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (void)updateNoticeWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (void)rebuildSegmentRowsWithSegments:(NSArray<NJSponsorBlockSegment *> *)segments manager:(NJSponsorBlockManager *)manager;
- (void)showNoticeMode:(NJSponsorBlockPanelNoticeMode)mode segment:(NJSponsorBlockSegment *)segment title:(NSString *)title detail:(NSString *)detail primaryTitle:(NSString *)primaryTitle secondaryTitle:(NSString *)secondaryTitle;
- (void)showTransientMessage:(NSString *)title detail:(NSString *)detail;
- (void)hideNotice;
- (BOOL)noticeSuppressedForSegment:(NJSponsorBlockSegment *)segment;
- (void)suppressCurrentNoticeSegment;
- (NJSponsorBlockSegment *)segmentFromSender:(id)sender;
- (void)voteForSegment:(NJSponsorBlockSegment *)segment type:(NSInteger)type;
- (void)submitSegmentTapped:(UIButton *)button;
- (void)presentSubmissionMenuFromView:(UIView *)sourceView;
- (BOOL)validateCurrentVideoForDraftCreation;
- (void)presentSubmissionCategoryPickerFromView:(UIView *)sourceView;
- (void)presentSubmissionManager;
- (void)submitCurrentVideoDraftsFromPanel;
- (void)beginSubmissionWithCategory:(NSString *)category;
- (void)confirmPOISubmissionWithCategory:(NSString *)category sourceView:(UIView *)sourceView;
- (void)finishSubmissionAtCurrentTime;
- (void)cancelSubmissionDraft;
- (void)saveDraftSegmentValues:(NSArray<NSNumber *> *)segment category:(NSString *)category actionType:(NSString *)actionType;
- (BOOL)currentPlaybackTimeIsValid:(NSTimeInterval)time;
- (NSArray<NSNumber *> *)roundedSegmentFromStart:(NSTimeInterval)start end:(NSTimeInterval)end;
- (NSNumber *)roundedTimeNumber:(NSTimeInterval)time;
- (NSString *)submissionActionTypeForCategory:(NSString *)category;
- (UIViewController *)presentationViewController;
- (NSTimeInterval)durationForSegment:(NJSponsorBlockSegment *)segment;
- (CGFloat)segmentRowsContentHeight;
- (NSString *)detailTextForSegment:(NJSponsorBlockSegment *)segment currentTime:(NSTimeInterval)currentTime;
- (NSString *)compactStringFromTime:(NSTimeInterval)time;

@end

@implementation NJSponsorBlockPanelView

static NJSponsorBlockOverlayWindow *NJSponsorBlockSharedOverlayWindow;
static UIViewController *NJSponsorBlockSharedOverlayController;
static NSHashTable<UIView *> *NJSponsorBlockNativeTimelineViews;
static NSTimeInterval NJSponsorBlockLastPlaybackActiveTime;
static void *NJSponsorBlockNativeTimelineKey = &NJSponsorBlockNativeTimelineKey;
static void *NJSponsorBlockManualSkipSegmentKey = &NJSponsorBlockManualSkipSegmentKey;
static void *NJSponsorBlockPanelSegmentKey = &NJSponsorBlockPanelSegmentKey;

+ (instancetype)sharedPanel {
    static NJSponsorBlockPanelView *panel = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        panel = [[NJSponsorBlockPanelView alloc] initWithFrame:CGRectMake(16, 88, NJSponsorBlockPanelWidth, NJSponsorBlockPanelMinHeight)];
    });
    return panel;
}

+ (UIView *)currentHostView {
    UIView *overlayHostView = [self overlayHostView];
    if (overlayHostView) {
        return overlayHostView;
    }

    NSArray<UIWindow *> *windows = @[];
    if (@available(iOS 13.0, *)) {
        UIWindowScene *scene = [self activeWindowScene];
        if (scene) {
            windows = scene.windows;
        }
    }
    if (windows.count == 0) {
        UIWindow *keyWindow = nil;
        for (UIWindowScene *scene in UIApplication.sharedApplication.connectedScenes) {
            if ([scene isKindOfClass:UIWindowScene.class] && scene.activationState == UISceneActivationStateForegroundActive) {
                for (UIWindow *window in ((UIWindowScene *)scene).windows) {
                    if (window.isKeyWindow) { keyWindow = window; break; }
                }
            }
        }
        if (keyWindow) { return keyWindow; }
        return nil;
    }

    UIWindow *bestWindow = nil;
    for (UIWindow *window in windows) {
        if (window.hidden || window.alpha <= 0.01 || CGRectIsEmpty(window.bounds)) {
            continue;
        }
        if (window.windowLevel != UIWindowLevelNormal && window.windowLevel != UIWindowLevelStatusBar) {
            continue;
        }
        bestWindow = window;
    }
    if (!bestWindow) {
        bestWindow = windows.firstObject;
    }
    return bestWindow;
}

+ (UIView *)overlayHostView {
    if (![NSThread isMainThread]) {
        return nil;
    }

    if (!NJSponsorBlockSharedOverlayController) {
        NJSponsorBlockSharedOverlayController = [[UIViewController alloc] init];
        NJSponsorBlockSharedOverlayController.view.backgroundColor = UIColor.clearColor;
        NJSponsorBlockSharedOverlayController.view.userInteractionEnabled = YES;
    }

    if (!NJSponsorBlockSharedOverlayWindow) {
        CGRect frame = UIScreen.mainScreen.bounds;
        NJSponsorBlockSharedOverlayWindow = [[NJSponsorBlockOverlayWindow alloc] initWithFrame:frame];
        NJSponsorBlockSharedOverlayWindow.backgroundColor = UIColor.clearColor;
        NJSponsorBlockSharedOverlayWindow.windowLevel = UIWindowLevelStatusBar + 20.0;
        NJSponsorBlockSharedOverlayWindow.rootViewController = NJSponsorBlockSharedOverlayController;
        NJSponsorBlockSharedOverlayWindow.hidden = NO;
        NSLog(@"[NJSponsorBlock] overlay window created %@", NJSponsorBlockSharedOverlayWindow);
    }

    if (@available(iOS 13.0, *)) {
        UIWindowScene *activeScene = [self activeWindowScene];
        if (activeScene && NJSponsorBlockSharedOverlayWindow.windowScene != activeScene) {
            NJSponsorBlockSharedOverlayWindow.windowScene = activeScene;
        }
    }

    NJSponsorBlockSharedOverlayWindow.frame = UIScreen.mainScreen.bounds;
    NJSponsorBlockSharedOverlayWindow.hidden = NO;
    [NJSponsorBlockSharedOverlayWindow bringSubviewToFront:NJSponsorBlockSharedOverlayController.view];
    return NJSponsorBlockSharedOverlayController.view;
}

+ (void)markPlaybackActive {
    dispatch_async(dispatch_get_main_queue(), ^{
        NJSponsorBlockLastPlaybackActiveTime = CACurrentMediaTime();
        if (NJSponsorBlockSharedOverlayWindow && NJSponsorBlockSharedOverlayWindow.hidden) {
            NJSponsorBlockSharedOverlayWindow.hidden = NO;
        }
        [NSObject cancelPreviousPerformRequestsWithTarget:self selector:@selector(hideOverlayIfIdle) object:nil];
        [self performSelector:@selector(hideOverlayIfIdle) withObject:nil afterDelay:NJSponsorBlockOverlayIdleTimeout + 0.5];
    });
}

+ (void)hideOverlayIfIdle {
    NSTimeInterval idleTime = CACurrentMediaTime() - NJSponsorBlockLastPlaybackActiveTime;
    if (idleTime < NJSponsorBlockOverlayIdleTimeout) {
        [self performSelector:@selector(hideOverlayIfIdle) withObject:nil afterDelay:NJSponsorBlockOverlayIdleTimeout - idleTime + 0.5];
        return;
    }
    [self hideOverlay];
}

+ (void)hideOverlay {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[self sharedPanel] removeFromSuperview];
        [[self sharedTimelineView] removeFromSuperview];
        if (NJSponsorBlockNativeTimelineViews) {
            for (UIView *timeline in NJSponsorBlockNativeTimelineViews.allObjects) {
                [timeline removeFromSuperview];
            }
            [NJSponsorBlockNativeTimelineViews removeAllObjects];
        }
        if (NJSponsorBlockSharedOverlayWindow) {
            NJSponsorBlockSharedOverlayWindow.hidden = YES;
        }
        NSLog(@"[NJSponsorBlock] overlay hidden");
    });
}

+ (UIWindowScene *)activeWindowScene API_AVAILABLE(ios(13.0)) {
    NSSet<UIScene *> *scenes = UIApplication.sharedApplication.connectedScenes;
    for (UIScene *scene in scenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) {
            continue;
        }
        if (scene.activationState == UISceneActivationStateForegroundActive) {
            return (UIWindowScene *)scene;
        }
    }
    for (UIScene *scene in scenes) {
        if ([scene isKindOfClass:UIWindowScene.class]) {
            return (UIWindowScene *)scene;
        }
    }
    return nil;
}

+ (UIButton *)sharedEntryButton {
    static UIButton *button = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        button = [UIButton buttonWithType:UIButtonTypeCustom];
        button.frame = CGRectMake(0, 0, 38, 38);
        button.accessibilityIdentifier = @"NJSponsorBlockEntryButton";
//        button.backgroundColor = [UIColor colorWithWhite:0 alpha:0.36];
//        button.layer.cornerRadius = 19;
//        button.layer.borderWidth = 1;
//        button.layer.borderColor = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95].CGColor;
        button.titleLabel.font = [UIFont boldSystemFontOfSize:23];
        [button setTitle:@"▷" forState:UIControlStateNormal];
        [button setTitleColor:[UIColor colorWithRed:0.02 green:0.78 blue:1 alpha:1] forState:UIControlStateNormal];
        [button addTarget:self action:@selector(togglePanelFromEntryButton:) forControlEvents:UIControlEventTouchUpInside];
    });
    return button;
}

+ (UIView *)sharedTimelineView {
    static UIView *view = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        view = [[UIView alloc] initWithFrame:CGRectZero];
        view.userInteractionEnabled = NO;
        view.backgroundColor = [UIColor colorWithWhite:0 alpha:0.18];
        view.layer.cornerRadius = 2;
        view.layer.masksToBounds = YES;
    });
    return view;
}

+ (void)installTimelineInView:(UIView *)view {
    UIView *timeline = [self sharedTimelineView];
    if (timeline.superview != view) {
        [timeline removeFromSuperview];
        [view addSubview:timeline];
    }
    [view bringSubviewToFront:timeline];
    [self layoutTimeline:timeline inView:view];
    [self renderTimeline:timeline];
}

+ (void)installNativeTimelineInView:(UIView *)view {
    if (!view || !NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    if (!NJSponsorBlockNativeTimelineViews) {
        NJSponsorBlockNativeTimelineViews = [NSHashTable weakObjectsHashTable];
    }

    UIView *timeline = objc_getAssociatedObject(view, NJSponsorBlockNativeTimelineKey);
    if (!timeline) {
        timeline = [[UIView alloc] initWithFrame:CGRectZero];
        timeline.userInteractionEnabled = NO;
        timeline.backgroundColor = UIColor.clearColor;
        timeline.clipsToBounds = YES;
        objc_setAssociatedObject(view, NJSponsorBlockNativeTimelineKey, timeline, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [NJSponsorBlockNativeTimelineViews addObject:timeline];
    }

    if (timeline.superview != view) {
        [timeline removeFromSuperview];
        [view addSubview:timeline];
        NSLog(@"[NJSponsorBlock] native timeline installed in %@ frame=%@", view, NSStringFromCGRect(view.frame));
    }
    [view bringSubviewToFront:timeline];
    [self layoutNativeTimeline:timeline inView:view];
    [self renderTimeline:timeline];
}

+ (void)installInView:(UIView *)view {
    if (!view || !NJ_MASTER_SWITCH_VALUE) {
        return;
    }

    NJSponsorBlockPanelView *panel = [self sharedPanel];
    if (panel.superview != view) {
        [panel removeFromSuperview];
        [view addSubview:panel];
    }
    [view bringSubviewToFront:panel];
    [panel keepInsideSuperview];
    [panel refreshContent];
}

+ (void)removePanel {
    [[self sharedPanel] removeFromSuperview];
}

+ (void)refresh {
    [[self sharedPanel] refreshContent];
    [self renderTimeline:[self sharedTimelineView]];
    [self refreshNativeTimelines];
}

+ (void)refreshNativeTimelines {
    for (UIView *timeline in NJSponsorBlockNativeTimelineViews.allObjects) {
        if (timeline.superview) {
            [self layoutNativeTimeline:timeline inView:timeline.superview];
            [self renderTimeline:timeline];
        }
    }
}

+ (void)togglePanelFromEntryButton:(UIButton *)button {
    UIView *view = [self panelHostViewForEntryButton:button];
    if (!view) {
        return;
    }

    NJSponsorBlockPanelView *panel = [self sharedPanel];
    if (panel.superview) {
        [self hidePanelAnimated];
        return;
    }

    [self installInView:view];
    CGRect buttonFrame = [view convertRect:button.bounds fromView:button];
    CGRect frame = panel.frame;
    CGFloat spacing = 8.0;
    frame.origin.x = CGRectGetMaxX(buttonFrame) - CGRectGetWidth(frame);
    frame.origin.y = CGRectGetMaxY(buttonFrame) + spacing;

    CGFloat minX = NJSponsorBlockPanelMargin;
    CGFloat maxX = CGRectGetWidth(view.bounds) - CGRectGetWidth(frame) - NJSponsorBlockPanelMargin;
    CGFloat minY = NJSponsorBlockPanelTopMargin;
    CGFloat maxY = CGRectGetHeight(view.bounds) - CGRectGetHeight(frame) - NJSponsorBlockPanelBottomMargin;
    frame.origin.x = MIN(MAX(minX, frame.origin.x), MAX(minX, maxX));
    frame.origin.y = MIN(MAX(minY, frame.origin.y), MAX(minY, maxY));
    panel.frame = frame;
    [panel keepInsideSuperview];
    [self showPanelAnimated:panel];
}

+ (UIView *)panelHostViewForEntryButton:(UIButton *)button {
    if (button.window) {
        return button.window;
    }
    if (button.superview) {
        return button.superview;
    }
    return [self currentHostView];
}

+ (void)showPanelAnimated:(NJSponsorBlockPanelView *)panel {
    panel.alpha = 0.0;
    panel.transform = CGAffineTransformMakeScale(0.96, 0.96);
    [UIView animateWithDuration:0.18
                          delay:0
                        options:UIViewAnimationOptionCurveEaseOut
                     animations:^{
        panel.alpha = 1.0;
        panel.transform = CGAffineTransformIdentity;
    } completion:nil];
}

+ (void)hidePanelAnimated {
    NJSponsorBlockPanelView *panel = [self sharedPanel];
    if (!panel.superview) {
        return;
    }

    [UIView animateWithDuration:0.16
                          delay:0
                        options:UIViewAnimationOptionCurveEaseIn
                     animations:^{
        panel.alpha = 0.0;
        panel.transform = CGAffineTransformMakeScale(0.96, 0.96);
    } completion:^(__unused BOOL finished) {
        [panel removeFromSuperview];
        panel.alpha = 1.0;
        panel.transform = CGAffineTransformIdentity;
    }];
}

+ (void)layoutEntryButton:(UIButton *)button inView:(UIView *)view {
    UIEdgeInsets insets = view.safeAreaInsets;
    CGRect bounds = view.bounds;
    BOOL portrait = CGRectGetHeight(bounds) >= CGRectGetWidth(bounds);

    CGFloat x = CGRectGetWidth(bounds) - insets.right - 132.0;
    CGFloat y = portrait ? insets.top + 64.0 : insets.top + 22.0;
    button.frame = CGRectMake(MAX(insets.left + 8.0, x), y, 38.0, 38.0);
}

+ (void)layoutTimeline:(UIView *)timeline inView:(UIView *)view {
    CGRect bounds = view.bounds;
    UIEdgeInsets insets = view.safeAreaInsets;
    BOOL portrait = CGRectGetHeight(bounds) >= CGRectGetWidth(bounds);
    CGFloat x = insets.left + 106.0;
    CGFloat width = CGRectGetWidth(bounds) - insets.left - insets.right - 206.0;
    CGFloat y = portrait ? insets.top + 244.0 : CGRectGetHeight(bounds) - insets.bottom - 42.0;
    timeline.frame = CGRectMake(MAX(insets.left + 72.0, x), y, MAX(120.0, width), 4.0);
}

+ (void)layoutNativeTimeline:(UIView *)timeline inView:(UIView *)view {
    CGRect bounds = view.bounds;
    CGFloat width = CGRectGetWidth(bounds);
    CGFloat height = CGRectGetHeight(bounds);
    CGFloat timelineHeight = MIN(4.0, MAX(2.0, height));
    CGFloat y = MAX(0, (height - timelineHeight) * 0.5);
    timeline.frame = CGRectMake(0, y, width, timelineHeight);
}

+ (void)renderTimeline:(UIView *)timeline {
    if (!timeline.superview) {
        return;
    }

    [timeline.subviews makeObjectsPerformSelector:@selector(removeFromSuperview)];
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    NSArray<NJSponsorBlockSegment *> *segments = [manager displaySegments];
    NSTimeInterval duration = manager.estimatedVideoDuration;
    if (duration <= 0 || segments.count == 0) {
        return;
    }

    CGFloat width = CGRectGetWidth(timeline.bounds);
    CGFloat height = CGRectGetHeight(timeline.bounds);
    for (NJSponsorBlockSegment *segment in segments) {
        CGFloat startX = MAX(0, MIN(width, width * segment.startTime / duration));
        CGFloat endX = MAX(startX + 2.0, MIN(width, width * segment.endTime / duration));
        UIView *mark = [[UIView alloc] initWithFrame:CGRectMake(startX, 0, endX - startX, height)];
        mark.backgroundColor = segment.isUnsubmitted ? [[self colorForCategory:segment.category] colorWithAlphaComponent:0.48] : [self colorForCategory:segment.category];
        if (segment.isUnsubmitted) {
            mark.layer.borderWidth = 1.0;
            mark.layer.borderColor = [UIColor colorWithWhite:1 alpha:0.75].CGColor;
        }
        [timeline addSubview:mark];
    }

    CGFloat playheadX = MAX(0, MIN(width, width * manager.currentPlaybackTime / duration));
    UIView *playhead = [[UIView alloc] initWithFrame:CGRectMake(playheadX - 1.0, 0, 2.0, height)];
    playhead.backgroundColor = UIColor.whiteColor;
    [timeline addSubview:playhead];
}

+ (UIColor *)colorForCategory:(NSString *)category {
    return [NJSponsorBlockSettings colorForCategory:category];
}

+ (void)hidePanelOnly {
    [self hidePanelAnimated];
}

- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        self.service = [[NJSponsorBlockService alloc] init];
        [self setupViews];
        [self refreshContent];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(refreshContent)
                                                     name:NJSponsorBlockStateDidChangeNotification
                                                   object:nil];
    }
    return self;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (UIButton *)noticeButtonWithTitle:(NSString *)title color:(UIColor *)color action:(SEL)action {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
    [button setTitle:title forState:UIControlStateNormal];
    [button setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    button.titleLabel.font = [UIFont systemFontOfSize:12 weight:UIFontWeightBold];
    button.backgroundColor = color;
    button.layer.cornerRadius = 6;
    [button addTarget:self action:action forControlEvents:UIControlEventTouchUpInside];
    return button;
}

- (UIButton *)actionButtonWithTitle:(NSString *)title color:(UIColor *)color action:(SEL)action segment:(NJSponsorBlockSegment *)segment {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
    [button setTitle:title forState:UIControlStateNormal];
    [button setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    button.titleLabel.font = [UIFont systemFontOfSize:11 weight:UIFontWeightBold];
    button.backgroundColor = color;
    button.layer.cornerRadius = 6;
    objc_setAssociatedObject(button, NJSponsorBlockPanelSegmentKey, segment, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [button addTarget:self action:action forControlEvents:UIControlEventTouchUpInside];
    [button.heightAnchor constraintEqualToConstant:24].active = YES;
    return button;
}

- (void)setupViews {
    self.backgroundColor = [UIColor colorWithWhite:0.06 alpha:0.86];
    self.layer.cornerRadius = 14.0;
    self.layer.borderWidth = 0.5;
    self.layer.borderColor = [UIColor colorWithWhite:1 alpha:0.16].CGColor;
    self.layer.shadowColor = UIColor.blackColor.CGColor;
    self.layer.shadowOpacity = 0.28;
    self.layer.shadowRadius = 18.0;
    self.layer.shadowOffset = CGSizeMake(0, 8);
    self.layer.masksToBounds = NO;

    self.iconLabel = [[UILabel alloc] init];
    self.iconLabel.text = @"▷";
    self.iconLabel.textColor = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:1.0];
    self.iconLabel.font = [UIFont boldSystemFontOfSize:28];
    self.iconLabel.textAlignment = NSTextAlignmentCenter;

    self.titleLabel = [[UILabel alloc] init];
    self.titleLabel.textColor = UIColor.whiteColor;
    self.titleLabel.font = [UIFont boldSystemFontOfSize:17];

    self.subtitleLabel = [[UILabel alloc] init];
    self.subtitleLabel.textColor = [UIColor colorWithWhite:0.72 alpha:1];
    self.subtitleLabel.font = [UIFont systemFontOfSize:12 weight:UIFontWeightMedium];

    UIStackView *headerTextStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.titleLabel, self.subtitleLabel]];
    headerTextStack.axis = UILayoutConstraintAxisVertical;
    headerTextStack.spacing = 2;

    self.headerStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.iconLabel, headerTextStack]];
    self.headerStack.axis = UILayoutConstraintAxisHorizontal;
    self.headerStack.alignment = UIStackViewAlignmentCenter;
    self.headerStack.spacing = 8;
    self.headerStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:self.headerStack];

    self.noticeView = [[UIView alloc] init];
    self.noticeView.backgroundColor = [UIColor colorWithRed:0.02 green:0.18 blue:0.24 alpha:0.94];
    self.noticeView.layer.cornerRadius = 10;
    self.noticeView.layer.borderWidth = 0.5;
    self.noticeView.layer.borderColor = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.45].CGColor;
    self.noticeView.hidden = YES;
    self.noticeView.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:self.noticeView];

    self.noticeTitleLabel = [[UILabel alloc] init];
    self.noticeTitleLabel.textColor = UIColor.whiteColor;
    self.noticeTitleLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightBold];

    self.noticeDetailLabel = [[UILabel alloc] init];
    self.noticeDetailLabel.textColor = [UIColor colorWithWhite:0.78 alpha:1];
    self.noticeDetailLabel.font = [UIFont systemFontOfSize:11 weight:UIFontWeightMedium];
    self.noticeDetailLabel.lineBreakMode = NSLineBreakByTruncatingTail;

    self.noticePrimaryButton = [self noticeButtonWithTitle:@"" color:[UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95] action:@selector(noticePrimaryTapped:)];
    self.noticeSecondaryButton = [self noticeButtonWithTitle:@"" color:[UIColor colorWithWhite:0.32 alpha:0.95] action:@selector(noticeSecondaryTapped:)];
    self.noticeCloseButton = [self noticeButtonWithTitle:@"×" color:[UIColor clearColor] action:@selector(noticeCloseTapped:)];

    UIStackView *noticeTextStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.noticeTitleLabel, self.noticeDetailLabel]];
    noticeTextStack.axis = UILayoutConstraintAxisVertical;
    noticeTextStack.spacing = 2;

    UIStackView *noticeButtonStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.noticePrimaryButton, self.noticeSecondaryButton, self.noticeCloseButton]];
    noticeButtonStack.axis = UILayoutConstraintAxisHorizontal;
    noticeButtonStack.alignment = UIStackViewAlignmentCenter;
    noticeButtonStack.spacing = 5;

    UIStackView *noticeStack = [[UIStackView alloc] initWithArrangedSubviews:@[noticeTextStack, noticeButtonStack]];
    noticeStack.axis = UILayoutConstraintAxisHorizontal;
    noticeStack.alignment = UIStackViewAlignmentCenter;
    noticeStack.spacing = 8;
    noticeStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.noticeView addSubview:noticeStack];

    self.segmentScrollView = [[UIScrollView alloc] init];
    self.segmentScrollView.showsVerticalScrollIndicator = YES;
    self.segmentScrollView.alwaysBounceVertical = NO;
    self.segmentScrollView.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:self.segmentScrollView];

    self.segmentStackView = [[UIStackView alloc] init];
    self.segmentStackView.axis = UILayoutConstraintAxisVertical;
    self.segmentStackView.spacing = NJSponsorBlockSegmentRowSpacing;
    self.segmentStackView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.segmentScrollView addSubview:self.segmentStackView];

    self.progressView = [[UIView alloc] init];
    self.progressView.backgroundColor = [UIColor colorWithWhite:1 alpha:0.10];
    self.progressView.layer.cornerRadius = 3;
    self.progressView.layer.masksToBounds = YES;
    self.progressView.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:self.progressView];

    self.toggleButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.toggleButton.layer.cornerRadius = 15;
    self.toggleButton.titleLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightSemibold];
    [self.toggleButton addTarget:self action:@selector(toggleEnabled) forControlEvents:UIControlEventTouchUpInside];

    self.submitButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.submitButton.layer.cornerRadius = 15;
    self.submitButton.titleLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightSemibold];
    self.submitButton.backgroundColor = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95];
    [self.submitButton setTitle:@"提交" forState:UIControlStateNormal];
    [self.submitButton setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    [self.submitButton addTarget:self action:@selector(submitSegmentTapped:) forControlEvents:UIControlEventTouchUpInside];

    self.statsLabel = [[UILabel alloc] init];
    self.statsLabel.textColor = [UIColor colorWithWhite:0.72 alpha:1];
    self.statsLabel.font = [UIFont systemFontOfSize:11 weight:UIFontWeightMedium];
    self.statsLabel.numberOfLines = 2;

    self.timeLabel = [[UILabel alloc] init];
    self.timeLabel.textColor = [UIColor colorWithWhite:0.82 alpha:1];
    self.timeLabel.font = [UIFont monospacedDigitSystemFontOfSize:12 weight:UIFontWeightSemibold];
    self.timeLabel.textAlignment = NSTextAlignmentRight;

    self.footerStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.toggleButton, self.submitButton, self.statsLabel, self.timeLabel]];
    self.footerStack.axis = UILayoutConstraintAxisHorizontal;
    self.footerStack.alignment = UIStackViewAlignmentCenter;
    self.footerStack.distribution = UIStackViewDistributionFill;
    self.footerStack.spacing = 8;
    self.footerStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:self.footerStack];

    self.closeButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.closeButton setTitle:@"✕" forState:UIControlStateNormal];
    [self.closeButton setTitleColor:[UIColor colorWithWhite:0.72 alpha:1] forState:UIControlStateNormal];
    self.closeButton.titleLabel.font = [UIFont systemFontOfSize:15 weight:UIFontWeightMedium];
    self.closeButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.closeButton addTarget:self action:@selector(closePanelTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self addSubview:self.closeButton];

    UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handlePan:)];
    pan.delegate = self;
    [self addGestureRecognizer:pan];

    self.noticeHeightConstraint = [self.noticeView.heightAnchor constraintEqualToConstant:0];
    self.segmentScrollHeightConstraint = [self.segmentScrollView.heightAnchor constraintEqualToConstant:NJSponsorBlockSegmentEmptyHeight];

    [NSLayoutConstraint activateConstraints:@[
        [self.iconLabel.widthAnchor constraintEqualToConstant:34],
        [self.toggleButton.widthAnchor constraintEqualToConstant:70],
        [self.toggleButton.heightAnchor constraintEqualToConstant:30],
        [self.submitButton.widthAnchor constraintEqualToConstant:52],
        [self.submitButton.heightAnchor constraintEqualToConstant:30],
        [self.noticePrimaryButton.widthAnchor constraintEqualToConstant:52],
        [self.noticeSecondaryButton.widthAnchor constraintEqualToConstant:58],
        [self.noticeCloseButton.widthAnchor constraintEqualToConstant:24],

        [self.closeButton.topAnchor constraintEqualToAnchor:self.topAnchor constant:8],
        [self.closeButton.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-8],
        [self.closeButton.widthAnchor constraintEqualToConstant:28],
        [self.closeButton.heightAnchor constraintEqualToConstant:28],

        [self.headerStack.topAnchor constraintEqualToAnchor:self.topAnchor constant:12],
        [self.headerStack.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.headerStack.trailingAnchor constraintEqualToAnchor:self.closeButton.leadingAnchor constant:-4],

        [self.noticeView.topAnchor constraintEqualToAnchor:self.headerStack.bottomAnchor constant:10],
        [self.noticeView.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.noticeView.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
        self.noticeHeightConstraint,
        [noticeStack.leadingAnchor constraintEqualToAnchor:self.noticeView.leadingAnchor constant:8],
        [noticeStack.trailingAnchor constraintEqualToAnchor:self.noticeView.trailingAnchor constant:-6],
        [noticeStack.centerYAnchor constraintEqualToAnchor:self.noticeView.centerYAnchor],

        [self.segmentScrollView.topAnchor constraintEqualToAnchor:self.noticeView.bottomAnchor constant:10],
        [self.segmentScrollView.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.segmentScrollView.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
        self.segmentScrollHeightConstraint,
        [self.segmentStackView.topAnchor constraintEqualToAnchor:self.segmentScrollView.topAnchor],
        [self.segmentStackView.leadingAnchor constraintEqualToAnchor:self.segmentScrollView.leadingAnchor],
        [self.segmentStackView.trailingAnchor constraintEqualToAnchor:self.segmentScrollView.trailingAnchor],
        [self.segmentStackView.bottomAnchor constraintEqualToAnchor:self.segmentScrollView.bottomAnchor],
        [self.segmentStackView.widthAnchor constraintEqualToAnchor:self.segmentScrollView.widthAnchor],

        [self.progressView.topAnchor constraintEqualToAnchor:self.segmentScrollView.bottomAnchor constant:10],
        [self.progressView.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.progressView.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
        [self.progressView.heightAnchor constraintEqualToConstant:6],

        [self.footerStack.topAnchor constraintEqualToAnchor:self.progressView.bottomAnchor constant:10],
        [self.footerStack.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.footerStack.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
        [self.footerStack.bottomAnchor constraintEqualToAnchor:self.bottomAnchor constant:-12],
    ]];
}

- (void)refreshContent {
    if ([NSThread isMainThread]) {
        [self refreshContentOnMainThread];
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self refreshContentOnMainThread];
        });
    }
}

- (void)refreshContentOnMainThread {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    NSArray<NJSponsorBlockSegment *> *segments = [manager displaySegments] ?: @[];

    self.hidden = !NJ_MASTER_SWITCH_VALUE;
    [self updateHeaderWithManager:manager segments:segments];
    [self updateEnabledButton:[NJSponsorBlockSettings enabled]];
    [self updateFooterWithManager:manager segments:segments];
    [self updateNoticeWithManager:manager segments:segments];
    [self rebuildSegmentRowsWithSegments:segments manager:manager];
    [self renderPanelProgressWithSegments:segments duration:manager.estimatedVideoDuration];
    [self resizeForContent];
    [[self class] renderTimeline:[[self class] sharedTimelineView]];
    [[self class] refreshNativeTimelines];
}

- (void)updateHeaderWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments {
    self.titleLabel.text = @"小电视空降助手";
    if (segments.count > 0) {
        self.subtitleLabel.text = [NSString stringWithFormat:@"数据库中有 %lu 个可用片段", (unsigned long)segments.count];
    } else if (manager.videoID.length > 0) {
        self.subtitleLabel.text = @"当前视频暂无可跳过片段";
    } else {
        self.subtitleLabel.text = @"等待识别当前视频";
    }
}

- (void)updateEnabledButton:(BOOL)enabled {
    [self.toggleButton setTitle:(enabled ? @"启用" : @"关闭") forState:UIControlStateNormal];
    self.toggleButton.backgroundColor = enabled ? [UIColor colorWithRed:0 green:0.70 blue:0.05 alpha:1] : [UIColor colorWithWhite:0.30 alpha:1];
    [self.toggleButton setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    self.submitButton.enabled = enabled && !self.submissionRequestInFlight;
    self.submitButton.alpha = self.submitButton.enabled ? 1.0 : 0.45;
}

- (void)updateFooterWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments {
    self.timeLabel.text = [NSString stringWithFormat:@"当前 %@", [self stringFromTime:manager.currentPlaybackTime]];
    NSTimeInterval skippedDuration = [manager skippedDurationBeforePlaybackTime:manager.currentPlaybackTime];
    NSTimeInterval cleanTime = [manager playbackTimeWithoutSkippedSegments:manager.currentPlaybackTime];
    NSUInteger totalCount = [manager allSegments].count;
    NSString *segmentCountText = totalCount > segments.count ? [NSString stringWithFormat:@"%lu/%lu 段", (unsigned long)segments.count, (unsigned long)totalCount] : [NSString stringWithFormat:@"%lu 段", (unsigned long)segments.count];
    self.statsLabel.text = [NSString stringWithFormat:@"%@ · 省 %@\n净 %@",
                            segmentCountText,
                            [self compactStringFromTime:skippedDuration],
                            [self compactStringFromTime:cleanTime]];
}

- (void)updateNoticeWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments {
    (void)segments;
    if (![NJSponsorBlockSettings enabled]) {
        [self hideNotice];
        return;
    }

    NSTimeInterval currentTime = manager.currentPlaybackTime;
    if (self.submissionRequestInFlight) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeMessage
                     segment:nil
                       title:@"正在提交"
                      detail:@"请稍候"
                primaryTitle:nil
              secondaryTitle:nil];
        return;
    }

    if (self.submissionInProgress) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeSubmissionDraft
                     segment:nil
                       title:@"已记录起点"
                      detail:[NSString stringWithFormat:@"%@ · 起点 %@",
                              [self titleForCategory:self.submissionCategory],
                              [self stringFromTime:self.submissionStartTime]]
                primaryTitle:@"终点提交"
              secondaryTitle:@"取消"];
        return;
    }

    NJSponsorBlockSegment *manualSegment = [manager manualSkipSegmentAtPlaybackTime:currentTime];
    if (manualSegment && ![self noticeSuppressedForSegment:manualSegment]) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeManualSkip
                     segment:manualSegment
                       title:@"可手动跳过"
                      detail:[self detailTextForSegment:manualSegment currentTime:currentTime]
                primaryTitle:@"跳过"
              secondaryTitle:@"隐藏"];
        return;
    }

    NSTimeInterval advanceSeconds = [NJSponsorBlockSettings advanceNoticeDuration];
    NJSponsorBlockSegment *upcomingSegment = [manager upcomingAutoSkipSegmentAtPlaybackTime:currentTime withinSeconds:advanceSeconds];
    if (upcomingSegment && ![self noticeSuppressedForSegment:upcomingSegment]) {
        NSTimeInterval remaining = MAX(0, upcomingSegment.startTime - currentTime);
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeAdvance
                     segment:upcomingSegment
                       title:@"即将自动跳过"
                      detail:[NSString stringWithFormat:@"%@ · 还有 %@",
                              [self titleForCategory:upcomingSegment.category],
                              [self compactStringFromTime:remaining]]
                primaryTitle:@"立即"
              secondaryTitle:@"本次不跳"];
        return;
    }

    NJSponsorBlockSegment *lastSkippedSegment = [manager lastSkippedSegment];
    if (lastSkippedSegment && ![self noticeSuppressedForSegment:lastSkippedSegment]) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeSkipped
                     segment:lastSkippedSegment
                       title:@"已跳过片段"
                      detail:[self detailTextForSegment:lastSkippedSegment currentTime:currentTime]
                primaryTitle:@"撤销"
              secondaryTitle:nil];
        return;
    }

    [self hideNotice];
}

- (void)rebuildSegmentRowsWithSegments:(NSArray<NJSponsorBlockSegment *> *)segments manager:(NJSponsorBlockManager *)manager {
    [self clearSegmentRows];

    UIView *activeRow = nil;
    for (NJSponsorBlockSegment *segment in segments) {
        UIView *row = [self rowForSegment:segment currentTime:manager.currentPlaybackTime];
        [self.segmentStackView addArrangedSubview:row];
        if (!activeRow && [segment containsPlaybackTime:manager.currentPlaybackTime]) {
            activeRow = row;
        }
    }
    if (segments.count == 0) {
        [self.segmentStackView addArrangedSubview:[self emptyRow]];
    } else if (activeRow) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.segmentScrollView scrollRectToVisible:activeRow.frame animated:NO];
        });
    }
}

- (BOOL)noticeSuppressedForSegment:(NJSponsorBlockSegment *)segment {
    return segment.uuid.length > 0 && [segment.uuid isEqualToString:self.suppressedNoticeUUID];
}

- (void)suppressCurrentNoticeSegment {
    self.suppressedNoticeUUID = self.noticeSegment.uuid.length > 0 ? self.noticeSegment.uuid : nil;
}

- (void)showNoticeMode:(NJSponsorBlockPanelNoticeMode)mode
               segment:(NJSponsorBlockSegment *)segment
                 title:(NSString *)title
                detail:(NSString *)detail
          primaryTitle:(NSString *)primaryTitle
        secondaryTitle:(NSString *)secondaryTitle {
    [NSObject cancelPreviousPerformRequestsWithTarget:self selector:@selector(hideNotice) object:nil];
    self.noticeMode = mode;
    self.noticeSegment = segment;
    self.noticeTitleLabel.text = title;
    self.noticeDetailLabel.text = detail;
    [self.noticePrimaryButton setTitle:primaryTitle ?: @"" forState:UIControlStateNormal];
    [self.noticeSecondaryButton setTitle:secondaryTitle ?: @"" forState:UIControlStateNormal];
    self.noticePrimaryButton.hidden = primaryTitle.length == 0;
    self.noticeSecondaryButton.hidden = secondaryTitle.length == 0;
    self.noticeView.hidden = NO;
    self.noticeHeightConstraint.constant = 52;
}

- (void)showTransientMessage:(NSString *)title detail:(NSString *)detail {
    [NSObject cancelPreviousPerformRequestsWithTarget:self selector:@selector(hideNotice) object:nil];
    [self showNoticeMode:NJSponsorBlockPanelNoticeModeMessage
                 segment:nil
                   title:title
                  detail:detail
            primaryTitle:nil
          secondaryTitle:nil];
    [self performSelector:@selector(hideNotice) withObject:nil afterDelay:2.0];
}

- (void)hideNotice {
    self.noticeMode = NJSponsorBlockPanelNoticeModeNone;
    self.noticeSegment = nil;
    self.noticeView.hidden = YES;
    self.noticeHeightConstraint.constant = 0;
}

- (void)clearSegmentRows {
    for (UIView *view in self.segmentStackView.arrangedSubviews.copy) {
        [self.segmentStackView removeArrangedSubview:view];
        [view removeFromSuperview];
    }
}

- (void)renderPanelProgressWithSegments:(NSArray<NJSponsorBlockSegment *> *)segments duration:(NSTimeInterval)duration {
    [self.progressView.subviews makeObjectsPerformSelector:@selector(removeFromSuperview)];
    if (duration <= 0 || segments.count == 0) {
        return;
    }

    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    CGFloat width = CGRectGetWidth(self.bounds) > 0 ? CGRectGetWidth(self.bounds) - 24.0 : NJSponsorBlockPanelWidth - 24.0;
    for (NJSponsorBlockSegment *segment in segments) {
        CGFloat startX = MAX(0, MIN(width, width * segment.startTime / duration));
        CGFloat endX = MAX(startX + 2.0, MIN(width, width * segment.endTime / duration));
        UIView *mark = [[UIView alloc] initWithFrame:CGRectMake(startX, 0, endX - startX, 6.0)];
        mark.backgroundColor = segment.isUnsubmitted ? [[[self class] colorForCategory:segment.category] colorWithAlphaComponent:0.48] : [[self class] colorForCategory:segment.category];
        if (segment.isUnsubmitted) {
            mark.layer.borderWidth = 1.0;
            mark.layer.borderColor = [UIColor colorWithWhite:1 alpha:0.75].CGColor;
        }
        mark.userInteractionEnabled = YES;
        objc_setAssociatedObject(mark, NJSponsorBlockPanelSegmentKey, segment, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [mark addGestureRecognizer:[[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(progressSegmentTapped:)]];
        [self.progressView addSubview:mark];
    }

    CGFloat playheadX = MAX(0, MIN(width, width * manager.currentPlaybackTime / duration));
    UIView *playhead = [[UIView alloc] initWithFrame:CGRectMake(playheadX - 1.0, 0, 2.0, 6.0)];
    playhead.backgroundColor = UIColor.whiteColor;
    [self.progressView addSubview:playhead];
}

- (UIView *)rowForSegment:(NJSponsorBlockSegment *)segment currentTime:(NSTimeInterval)currentTime {
    UIView *row = [[UIView alloc] init];
    BOOL active = [segment containsPlaybackTime:currentTime];
    BOOL skipped = [[NJSponsorBlockManager sharedInstance] hasActuallySkippedSegment:segment];
    if (active) {
        row.backgroundColor = [UIColor colorWithRed:0.00 green:0.55 blue:0.58 alpha:0.92];
    } else if (skipped) {
        row.backgroundColor = [UIColor colorWithRed:0.18 green:0.45 blue:0.20 alpha:0.82];
    } else {
        row.backgroundColor = [UIColor colorWithWhite:1 alpha:0.08];
    }
    row.layer.cornerRadius = 8;

    UILabel *dot = [[UILabel alloc] init];
    dot.text = @"●";
    dot.textColor = [[self class] colorForCategory:segment.category];
    dot.font = [UIFont systemFontOfSize:14 weight:UIFontWeightBold];

    NJSponsorBlockCategoryAction action = [NJSponsorBlockSettings actionForCategory:segment.category];
    UILabel *categoryLabel = [[UILabel alloc] init];
    NSString *categoryText = [NSString stringWithFormat:@"%@ · %@", [self titleForCategory:segment.category], [NJSponsorBlockSettings titleForAction:action]];
    categoryLabel.text = segment.isUnsubmitted ? [@"未提交 · " stringByAppendingString:categoryText] : categoryText;
    categoryLabel.textColor = UIColor.whiteColor;
    categoryLabel.font = [UIFont systemFontOfSize:13 weight:UIFontWeightBold];
    categoryLabel.lineBreakMode = NSLineBreakByTruncatingTail;

    NSString *stateText = segment.isUnsubmitted ? @"未提交" : (active ? @"播放中" : (skipped ? @"已跳过" : @"已加载"));
    UILabel *detailLabel = [[UILabel alloc] init];
    detailLabel.text = [NSString stringWithFormat:@"%@ · %@", stateText, [self detailTextForSegment:segment currentTime:currentTime]];
    detailLabel.textColor = [UIColor colorWithWhite:0.80 alpha:1];
    detailLabel.font = [UIFont monospacedDigitSystemFontOfSize:11 weight:UIFontWeightSemibold];
    detailLabel.textAlignment = NSTextAlignmentRight;

    UIStackView *topStack = [[UIStackView alloc] initWithArrangedSubviews:@[dot, categoryLabel, detailLabel]];
    topStack.axis = UILayoutConstraintAxisHorizontal;
    topStack.alignment = UIStackViewAlignmentCenter;
    topStack.spacing = 6;

    UIColor *blue = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95];
    UIColor *green = [UIColor colorWithRed:0.00 green:0.62 blue:0.18 alpha:0.95];
    UIColor *red = [UIColor colorWithRed:0.86 green:0.22 blue:0.18 alpha:0.95];
    UIColor *gray = [UIColor colorWithWhite:0.32 alpha:0.95];
    NSArray<UIView *> *buttons = segment.isUnsubmitted ? @[
        [self actionButtonWithTitle:@"起点" color:gray action:@selector(seekToSegmentStartTapped:) segment:segment],
        [self actionButtonWithTitle:@"跳过" color:blue action:@selector(skipSegmentTapped:) segment:segment],
    ] : @[
        [self actionButtonWithTitle:@"起点" color:gray action:@selector(seekToSegmentStartTapped:) segment:segment],
        [self actionButtonWithTitle:@"跳过" color:blue action:@selector(skipSegmentTapped:) segment:segment],
        [self actionButtonWithTitle:@"赞" color:green action:@selector(upvoteSegmentTapped:) segment:segment],
        [self actionButtonWithTitle:@"踩" color:red action:@selector(downvoteSegmentTapped:) segment:segment],
        [self actionButtonWithTitle:@"复制" color:gray action:@selector(copySegmentUUIDTapped:) segment:segment],
    ];
    UIStackView *buttonStack = [[UIStackView alloc] initWithArrangedSubviews:buttons];
    buttonStack.axis = UILayoutConstraintAxisHorizontal;
    buttonStack.alignment = UIStackViewAlignmentCenter;
    buttonStack.distribution = UIStackViewDistributionFillEqually;
    buttonStack.spacing = 5;

    UIStackView *stack = [[UIStackView alloc] initWithArrangedSubviews:@[topStack, buttonStack]];
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 5;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [row addSubview:stack];

    [NSLayoutConstraint activateConstraints:@[
        [row.heightAnchor constraintEqualToConstant:NJSponsorBlockSegmentRowHeight],
        [dot.widthAnchor constraintEqualToConstant:16],
        [stack.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:10],
        [stack.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-10],
        [stack.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
    ]];
    return row;
}

- (UIView *)emptyRow {
    UILabel *label = [[UILabel alloc] init];
    label.text = @"未加载片段或该视频暂无数据";
    label.textColor = [UIColor colorWithWhite:0.72 alpha:1];
    label.font = [UIFont systemFontOfSize:13 weight:UIFontWeightMedium];
    label.textAlignment = NSTextAlignmentCenter;
    [label.heightAnchor constraintEqualToConstant:NJSponsorBlockSegmentEmptyHeight].active = YES;
    return label;
}

- (void)closePanelTapped:(UIButton *)button {
    [NJSponsorBlockPanelView hidePanelAnimated];
}

- (void)toggleEnabled {
    [NJSponsorBlockSettings setEnabled:![NJSponsorBlockSettings enabled]];
    [self refreshContent];
}

- (void)submitSegmentTapped:(UIButton *)button {
    if (![NJSponsorBlockSettings enabled]) {
        [self showTransientMessage:@"无法提交" detail:@"请先启用 SponsorBlock"];
        return;
    }
    if (self.submissionRequestInFlight) {
        [self showTransientMessage:@"正在提交" detail:@"请等待当前请求完成"];
        return;
    }
    if (self.submissionInProgress) {
        [self refreshContent];
        return;
    }

    [self presentSubmissionMenuFromView:button];
}

- (void)presentSubmissionMenuFromView:(UIView *)sourceView {
    UIViewController *presenter = [self presentationViewController];
    if (!presenter) {
        [self showTransientMessage:@"无法提交" detail:@"无法打开提交菜单"];
        return;
    }

    NSUInteger draftCount = [NJSponsorBlockManager sharedInstance].unsubmittedSegmentsForCurrentVideo.count;
    NSString *message = draftCount > 0 ? [NSString stringWithFormat:@"当前视频有 %lu 个未提交片段", (unsigned long)draftCount] : @"可保存新草稿或管理已有草稿";
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"片段提交" message:message preferredStyle:UIAlertControllerStyleActionSheet];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"新增片段草稿" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        if (!strongSelf || ![strongSelf validateCurrentVideoForDraftCreation]) {
            return;
        }
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.2 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
            [strongSelf presentSubmissionCategoryPickerFromView:sourceView];
        });
    }]];
    UIAlertAction *submitAction = [UIAlertAction actionWithTitle:@"提交当前视频草稿" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        [weakSelf submitCurrentVideoDraftsFromPanel];
    }];
    submitAction.enabled = draftCount > 0;
    [alert addAction:submitAction];
    [alert addAction:[UIAlertAction actionWithTitle:@"管理未提交片段" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.2 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
            [weakSelf presentSubmissionManager];
        });
    }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    alert.popoverPresentationController.sourceView = sourceView ?: self;
    alert.popoverPresentationController.sourceRect = sourceView ? sourceView.bounds : self.bounds;
    [presenter presentViewController:alert animated:YES completion:nil];
}

- (BOOL)validateCurrentVideoForDraftCreation {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    if (manager.videoID.length == 0 || manager.cid <= 0) {
        [self showTransientMessage:@"无法提交" detail:@"尚未识别当前视频"];
        return NO;
    }
    if (![self currentPlaybackTimeIsValid:manager.currentPlaybackTime]) {
        [self showTransientMessage:@"无法提交" detail:@"无法获取当前播放时间"];
        return NO;
    }
    if (manager.estimatedVideoDuration <= 0 || !isfinite(manager.estimatedVideoDuration)) {
        [self showTransientMessage:@"无法提交" detail:@"暂未获取视频时长，稍后再试"];
        return NO;
    }
    return YES;
}

- (void)presentSubmissionCategoryPickerFromView:(UIView *)sourceView {
    UIViewController *presenter = [self presentationViewController];
    if (!presenter) {
        [self showTransientMessage:@"无法提交" detail:@"无法打开分类选择"];
        return;
    }

    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"新增片段草稿"
                                                                   message:@"选择本次草稿的片段分类"
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    __weak typeof(self) weakSelf = self;
    for (NJSponsorBlockCategoryOption *option in [NJSponsorBlockSettings categoryOptions]) {
        [alert addAction:[UIAlertAction actionWithTitle:option.title style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            if ([option.category isEqualToString:@"poi_highlight"]) {
                dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.2 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                    [strongSelf confirmPOISubmissionWithCategory:option.category sourceView:sourceView];
                });
            } else {
                [strongSelf beginSubmissionWithCategory:option.category];
            }
        }]];
    }
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    alert.popoverPresentationController.sourceView = sourceView ?: self;
    alert.popoverPresentationController.sourceRect = sourceView ? sourceView.bounds : self.bounds;
    [presenter presentViewController:alert animated:YES completion:nil];
}

- (void)presentSubmissionManager {
    UIViewController *presenter = [self presentationViewController];
    if (!presenter) {
        [self showTransientMessage:@"无法打开" detail:@"无法打开未提交片段管理"];
        return;
    }
    NJSponsorBlockSubmissionManagerViewController *controller = [[NJSponsorBlockSubmissionManagerViewController alloc] init];
    UINavigationController *navigationController = [[UINavigationController alloc] initWithRootViewController:controller];
    navigationController.modalPresentationStyle = UIModalPresentationPageSheet;
    [presenter presentViewController:navigationController animated:YES completion:nil];
}

- (void)submitCurrentVideoDraftsFromPanel {
    if (self.submissionRequestInFlight) {
        return;
    }
    self.submissionRequestInFlight = YES;
    [self refreshContent];
    __weak typeof(self) weakSelf = self;
    [[NJSponsorBlockManager sharedInstance] submitUnsubmittedSegmentsForCurrentVideoWithCompletion:^(BOOL success, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            strongSelf.submissionRequestInFlight = NO;
            [strongSelf showTransientMessage:(success ? @"提交成功" : @"提交失败")
                                      detail:(success ? @"当前视频草稿已提交" : (error.localizedDescription ?: @"请稍后重试"))];
            [strongSelf refreshContent];
        });
    }];
}

- (void)beginSubmissionWithCategory:(NSString *)category {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    NSTimeInterval currentTime = manager.currentPlaybackTime;
    if (![self currentPlaybackTimeIsValid:currentTime]) {
        [self showTransientMessage:@"无法提交" detail:@"无法获取当前播放时间"];
        return;
    }
    self.submissionCategory = category;
    self.submissionVideoID = manager.videoID;
    self.submissionCID = manager.cid;
    self.submissionVideoDuration = manager.estimatedVideoDuration;
    self.submissionStartTime = currentTime;
    self.submissionInProgress = YES;
    self.suppressedNoticeUUID = nil;
    [self showNoticeMode:NJSponsorBlockPanelNoticeModeSubmissionDraft
                 segment:nil
                   title:@"已记录起点"
                  detail:[NSString stringWithFormat:@"%@ · 起点 %@", [self titleForCategory:category], [self stringFromTime:currentTime]]
            primaryTitle:@"终点提交"
          secondaryTitle:@"取消"];
    [self resizeForContent];
}

- (void)confirmPOISubmissionWithCategory:(NSString *)category sourceView:(UIView *)sourceView {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    NSTimeInterval currentTime = manager.currentPlaybackTime;
    NSString *videoID = manager.videoID;
    NSInteger cid = manager.cid;
    NSTimeInterval videoDuration = manager.estimatedVideoDuration;
    if (![self currentPlaybackTimeIsValid:currentTime]) {
        [self showTransientMessage:@"无法提交" detail:@"无法获取当前播放时间"];
        return;
    }
    if (videoID.length == 0 || cid <= 0 || videoDuration <= 0 || !isfinite(videoDuration)) {
        [self showTransientMessage:@"无法提交" detail:@"当前视频信息不完整"];
        return;
    }

    UIViewController *presenter = [self presentationViewController];
    if (!presenter) {
        [self showTransientMessage:@"无法提交" detail:@"无法打开确认框"];
        return;
    }

    NSString *message = [NSString stringWithFormat:@"保存当前时间 %@ 为精彩片段草稿", [self stringFromTime:currentTime]];
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"保存精彩片段草稿" message:message preferredStyle:UIAlertControllerStyleActionSheet];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"保存草稿" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        if (!strongSelf) {
            return;
        }
        NJSponsorBlockManager *currentManager = [NJSponsorBlockManager sharedInstance];
        if (![currentManager.videoID isEqualToString:videoID] || currentManager.cid != cid) {
            [strongSelf showTransientMessage:@"无法提交" detail:@"当前视频已切换，请重新选择"];
            return;
        }
        [strongSelf saveDraftSegmentValues:@[[strongSelf roundedTimeNumber:MIN(currentTime, videoDuration)]]
                                  category:category
                                actionType:@"poi"];
    }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    alert.popoverPresentationController.sourceView = sourceView ?: self;
    alert.popoverPresentationController.sourceRect = sourceView ? sourceView.bounds : self.bounds;
    [presenter presentViewController:alert animated:YES completion:nil];
}

- (void)finishSubmissionAtCurrentTime {
    if (!self.submissionInProgress || self.submissionCategory.length == 0) {
        return;
    }

    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    NSTimeInterval currentTime = manager.currentPlaybackTime;
    if (![self currentPlaybackTimeIsValid:currentTime]) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeSubmissionDraft
                     segment:nil
                       title:@"无法提交"
                      detail:@"无法获取当前播放时间"
                primaryTitle:@"重试"
              secondaryTitle:@"取消"];
        return;
    }
    if (![manager.videoID isEqualToString:self.submissionVideoID] || manager.cid != self.submissionCID) {
        [self cancelSubmissionDraft];
        [self showTransientMessage:@"已取消提交" detail:@"当前视频已切换，请重新记录片段"];
        return;
    }

    NSArray<NSNumber *> *segment = [self roundedSegmentFromStart:self.submissionStartTime end:currentTime];
    NSTimeInterval start = segment.firstObject.doubleValue;
    NSTimeInterval end = segment.lastObject.doubleValue;
    NSTimeInterval duration = self.submissionVideoDuration;
    if (duration <= 0 || !isfinite(duration)) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeSubmissionDraft
                     segment:nil
                       title:@"无法提交"
                      detail:@"暂未获取视频时长，稍后再试"
                primaryTitle:@"重试"
              secondaryTitle:@"取消"];
        return;
    }
    if (end > duration) {
        end = duration;
        segment = @[@(start), @(end)];
    }

    NSTimeInterval minDuration = MAX([NJSponsorBlockSettings minDuration], 0.5);
    if (end - start < minDuration) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeSubmissionDraft
                     segment:nil
                       title:@"片段太短"
                      detail:[NSString stringWithFormat:@"至少需要 %@", [self compactStringFromTime:minDuration]]
                primaryTitle:@"重试"
              secondaryTitle:@"取消"];
        return;
    }

    [self saveDraftSegmentValues:segment
                        category:self.submissionCategory
                      actionType:[self submissionActionTypeForCategory:self.submissionCategory]];
}

- (void)cancelSubmissionDraft {
    self.submissionInProgress = NO;
    self.submissionCategory = nil;
    self.submissionVideoID = nil;
    self.submissionCID = 0;
    self.submissionVideoDuration = 0;
    self.submissionStartTime = 0;
    [self hideNotice];
    [self resizeForContent];
}

- (void)saveDraftSegmentValues:(NSArray<NSNumber *> *)segment category:(NSString *)category actionType:(NSString *)actionType {
    NJSponsorBlockSegment *localSegment = [[NJSponsorBlockManager sharedInstance] addUnsubmittedSegmentWithCategory:category actionType:actionType segment:segment];
    if (!localSegment) {
        [self showNoticeMode:NJSponsorBlockPanelNoticeModeSubmissionDraft
                     segment:nil
                       title:@"保存失败"
                      detail:@"无法保存本地未提交片段"
                primaryTitle:(self.submissionInProgress ? @"重试" : nil)
              secondaryTitle:(self.submissionInProgress ? @"取消" : nil)];
        return;
    }

    self.submissionInProgress = NO;
    self.submissionCategory = nil;
    self.submissionVideoID = nil;
    self.submissionCID = 0;
    self.submissionVideoDuration = 0;
    self.submissionStartTime = 0;
    [self showTransientMessage:@"已保存草稿" detail:@"可在提交菜单中提交或管理"];
    [self refreshContent];
}

- (BOOL)currentPlaybackTimeIsValid:(NSTimeInterval)time {
    return time >= 0 && isfinite(time);
}

- (NSArray<NSNumber *> *)roundedSegmentFromStart:(NSTimeInterval)start end:(NSTimeInterval)end {
    NSTimeInterval roundedStart = [self roundedTimeNumber:MIN(start, end)].doubleValue;
    NSTimeInterval roundedEnd = [self roundedTimeNumber:MAX(start, end)].doubleValue;
    return @[@(roundedStart), @(roundedEnd)];
}

- (NSNumber *)roundedTimeNumber:(NSTimeInterval)time {
    return @(round((time + DBL_EPSILON) * 1000.0) / 1000.0);
}

- (NSString *)submissionActionTypeForCategory:(NSString *)category {
    return [category isEqualToString:@"poi_highlight"] ? @"poi" : @"skip";
}

- (UIViewController *)presentationViewController {
    UIViewController *controller = NJSponsorBlockSharedOverlayController;
    if (!controller) {
        controller = [[[self class] currentHostView] window].rootViewController;
    }
    while (controller.presentedViewController && ![controller.presentedViewController isKindOfClass:UIAlertController.class]) {
        controller = controller.presentedViewController;
    }
    return controller;
}

- (NJSponsorBlockSegment *)segmentFromSender:(id)sender {
    return [sender isKindOfClass:[UIView class]] ? objc_getAssociatedObject(sender, NJSponsorBlockPanelSegmentKey) : nil;
}

- (void)manualSkipButtonTapped:(UIButton *)button {
    NJSponsorBlockSegment *segment = [self segmentFromSender:button] ?: objc_getAssociatedObject(button, NJSponsorBlockManualSkipSegmentKey);
    if (!segment) {
        return;
    }
    [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockManualSkipRequestNotification object:segment];
}

- (void)seekToSegmentStartTapped:(UIButton *)button {
    NJSponsorBlockSegment *segment = [self segmentFromSender:button];
    if (!segment) {
        return;
    }
    [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockSeekRequestNotification object:@(segment.startTime)];
}

- (void)skipSegmentTapped:(UIButton *)button {
    [self manualSkipButtonTapped:button];
}

- (void)upvoteSegmentTapped:(UIButton *)button {
    [self voteForSegment:[self segmentFromSender:button] type:1];
}

- (void)downvoteSegmentTapped:(UIButton *)button {
    [self voteForSegment:[self segmentFromSender:button] type:0];
}

- (void)copySegmentUUIDTapped:(UIButton *)button {
    NJSponsorBlockSegment *segment = [self segmentFromSender:button];
    if (segment.uuid.length == 0) {
        [self showTransientMessage:@"复制失败" detail:@"片段 UUID 为空"];
        return;
    }
    UIPasteboard.generalPasteboard.string = segment.uuid;
    [self showTransientMessage:@"已复制 UUID" detail:segment.uuid];
}

- (void)voteForSegment:(NJSponsorBlockSegment *)segment type:(NSInteger)type {
    if (segment.isUnsubmitted) {
        [self showTransientMessage:@"无法投票" detail:@"本地未提交片段不能投票"];
        return;
    }
    if (segment.uuid.length == 0) {
        [self showTransientMessage:@"无法投票" detail:@"片段 UUID 为空"];
        return;
    }

    __weak typeof(self) weakSelf = self;
    [self.service voteForSegmentWithUUID:segment.uuid type:type completion:^(BOOL success, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            if (success) {
                [strongSelf showTransientMessage:(type == 1 ? @"已点赞" : @"已点踩") detail:@"感谢反馈"];
            } else {
                [strongSelf showTransientMessage:@"投票失败" detail:error.localizedDescription ?: @"请稍后重试"];
            }
        });
    }];
}

- (void)noticePrimaryTapped:(UIButton *)button {
    if (self.noticeMode == NJSponsorBlockPanelNoticeModeSubmissionDraft) {
        [self finishSubmissionAtCurrentTime];
        return;
    }

    NJSponsorBlockSegment *segment = self.noticeSegment;
    if (!segment) {
        return;
    }
    self.suppressedNoticeUUID = nil;
    if (self.noticeMode == NJSponsorBlockPanelNoticeModeSkipped) {
        [[NJSponsorBlockManager sharedInstance] clearSkippedSegment:segment];
        [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockSeekRequestNotification object:@(segment.startTime)];
        [self hideNotice];
        return;
    }
    [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockManualSkipRequestNotification object:segment];
}

- (void)noticeSecondaryTapped:(UIButton *)button {
    if (self.noticeMode == NJSponsorBlockPanelNoticeModeSubmissionDraft) {
        [self cancelSubmissionDraft];
        return;
    }

    NJSponsorBlockSegment *segment = self.noticeSegment;
    if (self.noticeMode == NJSponsorBlockPanelNoticeModeAdvance && segment) {
        [[NJSponsorBlockManager sharedInstance] markSegmentSkipped:segment];
    } else {
        [self suppressCurrentNoticeSegment];
    }
    [self hideNotice];
}

- (void)noticeCloseTapped:(UIButton *)button {
    if (self.noticeMode == NJSponsorBlockPanelNoticeModeSubmissionDraft) {
        [self cancelSubmissionDraft];
        return;
    }
    [self suppressCurrentNoticeSegment];
    [self hideNotice];
}

- (void)progressSegmentTapped:(UITapGestureRecognizer *)gesture {
    NJSponsorBlockSegment *segment = objc_getAssociatedObject(gesture.view, NJSponsorBlockPanelSegmentKey);
    if (!segment) {
        return;
    }
    [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockSeekRequestNotification object:@(segment.startTime)];
}

- (BOOL)gestureRecognizer:(UIGestureRecognizer *)gestureRecognizer shouldReceiveTouch:(UITouch *)touch {
    if ([touch.view isKindOfClass:[UIControl class]]) {
        return NO;
    }
    return YES;
}

- (void)handlePan:(UIPanGestureRecognizer *)pan {
    UIView *superview = self.superview;
    if (!superview) {
        return;
    }

    CGPoint translation = [pan translationInView:superview];
    self.center = CGPointMake(self.center.x + translation.x, self.center.y + translation.y);
    [pan setTranslation:CGPointZero inView:superview];
    if (pan.state == UIGestureRecognizerStateEnded || pan.state == UIGestureRecognizerStateCancelled) {
        [self keepInsideSuperview];
    }
}

- (void)resizeForContent {
    UIView *superview = self.superview;
    if (!superview) {
        return;
    }

    CGFloat contentHeight = [self segmentRowsContentHeight];
    CGFloat listHeight = MIN(contentHeight, NJSponsorBlockSegmentListMaxHeight);
    self.segmentScrollHeightConstraint.constant = listHeight;
    self.segmentScrollView.scrollEnabled = contentHeight > listHeight + 1.0;

    CGFloat height = NJSponsorBlockPanelChromeHeight + listHeight + (self.noticeView.hidden ? 0 : 52.0);

    CGRect frame = self.frame;
    CGFloat availableWidth = CGRectGetWidth(superview.bounds) - NJSponsorBlockPanelMargin * 2.0;
    frame.size.width = MIN(NJSponsorBlockPanelWidth, availableWidth);
    frame.size.width = MAX(240.0, frame.size.width);
    frame.size.height = height;
    self.frame = frame;
    [self keepInsideSuperview];
}

- (void)keepInsideSuperview {
    UIView *superview = self.superview;
    if (!superview) {
        return;
    }

    UIEdgeInsets insets = UIEdgeInsetsMake(NJSponsorBlockPanelTopMargin,
                                           NJSponsorBlockPanelMargin,
                                           NJSponsorBlockPanelBottomMargin,
                                           NJSponsorBlockPanelMargin);
    CGRect frame = self.frame;
    CGFloat maxX = CGRectGetWidth(superview.bounds) - insets.right - CGRectGetWidth(frame);
    CGFloat maxY = CGRectGetHeight(superview.bounds) - insets.bottom - CGRectGetHeight(frame);
    frame.origin.x = MIN(MAX(insets.left, frame.origin.x), MAX(insets.left, maxX));
    frame.origin.y = MIN(MAX(insets.top, frame.origin.y), MAX(insets.top, maxY));
    self.frame = frame;
}

- (NSTimeInterval)durationForSegment:(NJSponsorBlockSegment *)segment {
    if (!segment || segment.endTime <= segment.startTime) {
        return 0;
    }
    return segment.endTime - segment.startTime;
}

- (CGFloat)segmentRowsContentHeight {
    NSUInteger count = self.segmentStackView.arrangedSubviews.count;
    if (count == 0) {
        return 0;
    }
    if (count == 1 && [self.segmentStackView.arrangedSubviews.firstObject isKindOfClass:[UILabel class]]) {
        return NJSponsorBlockSegmentEmptyHeight;
    }
    return count * NJSponsorBlockSegmentRowHeight + (count - 1) * NJSponsorBlockSegmentRowSpacing;
}

- (NSString *)detailTextForSegment:(NJSponsorBlockSegment *)segment currentTime:(NSTimeInterval)currentTime {
    (void)currentTime;
    NSString *actionType = segment.actionType.length > 0 ? segment.actionType : @"skip";
    return [NSString stringWithFormat:@"%@-%@ · %@ · %@",
            [self stringFromTime:segment.startTime],
            [self stringFromTime:segment.endTime],
            [self compactStringFromTime:[self durationForSegment:segment]],
            actionType];
}

- (NSString *)compactStringFromTime:(NSTimeInterval)time {
    if (time <= 0 || isnan(time) || isinf(time)) {
        return @"0s";
    }
    NSInteger seconds = (NSInteger)round(time);
    if (seconds < 60) {
        return [NSString stringWithFormat:@"%lds", (long)seconds];
    }
    return [NSString stringWithFormat:@"%ld:%02ld", (long)(seconds / 60), (long)(seconds % 60)];
}

- (NSString *)titleForCategory:(NSString *)category {
    if ([category isEqualToString:@"sponsor"]) {
        return @"赞助/恰饭";
    }
    if ([category isEqualToString:@"intro"]) {
        return @"开场动画";
    }
    if ([category isEqualToString:@"outro"]) {
        return @"结束片段";
    }
    if ([category isEqualToString:@"interaction"]) {
        return @"互动提醒";
    }
    if ([category isEqualToString:@"selfpromo"]) {
        return @"自我推广";
    }
    if ([category isEqualToString:@"preview"]) {
        return @"前情/预览";
    }
    if ([category isEqualToString:@"poi_highlight"]) {
        return @"精彩片段";
    }
    if ([category isEqualToString:@"filler"]) {
        return @"填充片段";
    }
    if ([category isEqualToString:@"music_offtopic"]) {
        return @"音乐/跑题";
    }
    if ([category isEqualToString:@"padding"]) {
        return @"空白/填充";
    }
    if ([category isEqualToString:@"exclusive_access"]) {
        return @"会员专享";
    }
    return category.length > 0 ? category : @"片段";
}

- (NSString *)stringFromTime:(NSTimeInterval)time {
    if (time <= 0 || isnan(time) || isinf(time)) {
        return @"0:00.000";
    }
    NSInteger minutes = (NSInteger)(time / 60.0);
    double seconds = time - minutes * 60.0;
    return [NSString stringWithFormat:@"%ld:%06.3f", (long)minutes, seconds];
}

@end
