//
//  NJSponsorBlockPanelView.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockPanelView.h"
#import "NJCommonDefine.h"
#import "NJSettingCache.h"
#import "NJSponsorBlockManager.h"
#import "NJSponsorBlockSegment.h"
#import <objc/runtime.h>
@import OSLog;

static CGFloat const NJSponsorBlockPanelWidth = 310.0;
static CGFloat const NJSponsorBlockPanelMinHeight = 154.0;
static NSTimeInterval const NJSponsorBlockOverlayIdleTimeout = 4.0;

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

@interface NJSponsorBlockPanelView ()

@property (nonatomic, strong) UILabel *iconLabel;
@property (nonatomic, strong) UILabel *titleLabel;
@property (nonatomic, strong) UILabel *subtitleLabel;
@property (nonatomic, strong) UIStackView *segmentStackView;
@property (nonatomic, strong) UIView *progressView;
@property (nonatomic, strong) UIButton *toggleButton;
@property (nonatomic, strong) UIButton *collapseButton;
@property (nonatomic, strong) UILabel *timeLabel;
@property (nonatomic, assign) BOOL collapsed;

@end

@implementation NJSponsorBlockPanelView

static NJSponsorBlockOverlayWindow *NJSponsorBlockSharedOverlayWindow;
static UIViewController *NJSponsorBlockSharedOverlayController;
static NSHashTable<UIView *> *NJSponsorBlockNativeTimelineViews;
static __weak UIView *NJSponsorBlockEntryAnchorView;
static NSTimeInterval NJSponsorBlockLastPlaybackActiveTime;
static void *NJSponsorBlockNativeTimelineKey = &NJSponsorBlockNativeTimelineKey;

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
    
    NSArray<UIWindow *> *windows = UIApplication.sharedApplication.windows;
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
        if (NJSponsorBlockSharedOverlayWindow.hidden) {
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
        [[self sharedEntryButton] removeFromSuperview];
        [[self sharedTimelineView] removeFromSuperview];
        for (UIView *timeline in NJSponsorBlockNativeTimelineViews.allObjects) {
            [timeline removeFromSuperview];
        }
        [NJSponsorBlockNativeTimelineViews removeAllObjects];
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
        button.backgroundColor = [UIColor colorWithWhite:0 alpha:0.36];
        button.layer.cornerRadius = 19;
        button.layer.borderWidth = 1;
        button.layer.borderColor = [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95].CGColor;
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

+ (void)installEntryInView:(UIView *)view {
    if (!NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    UIView *hostView = [self overlayHostView] ?: view;
    if (!hostView) {
        return;
    }
    [self markPlaybackActive];
    
    UIButton *button = [self sharedEntryButton];
    if (button.superview != hostView) {
        [button removeFromSuperview];
        [hostView addSubview:button];
        NSLog(@"[NJSponsorBlock] overlay entry installed in %@", hostView);
    }
    [hostView bringSubviewToFront:button];
    [self layoutEntryButton:button inView:hostView];
    [self installTimelineInView:hostView];
    [self refresh];
}

+ (void)installEntryDirectlyInContainer:(UIView *)container {
    if (![NSThread isMainThread]) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self installEntryDirectlyInContainer:container];
        });
        return;
    }

    if (!container || !NJ_MASTER_SWITCH_VALUE || !container.window) {
        return;
    }
    
    CGFloat origWidth = container.frame.size.width;

    [self markPlaybackActive];

    UIView *hostView = container;
    if (!hostView || !hostView.window) {
        return;
    }
    
    UIButton *button = [self sharedEntryButton];
    if (button.superview != hostView) {
        [button removeFromSuperview];
        [hostView insertSubview:button atIndex:1];
        NSLog(@"[NJSponsorBlock] entry installed as sibling near %@ host=%@ containerFrame=%@ hostBounds=%@",
              container,
              hostView,
              NSStringFromCGRect([container convertRect:container.bounds toView:hostView]),
              NSStringFromCGRect(hostView.bounds));
    }

    hostView.clipsToBounds = NO;

    [hostView bringSubviewToFront:button];
    
    button.frame = CGRectMake(origWidth - 41, 3, 38, 38);
    
    [self refresh];
}

+ (void)setEntryAnchorView:(UIView *)view {
    if (!view || view.hidden || view.alpha <= 0.01 || !view.window) {
        return;
    }
    if (NJSponsorBlockEntryAnchorView == view) {
        return;
    }
    NJSponsorBlockEntryAnchorView = view;
    NSLog(@"[NJSponsorBlock] entry anchor updated %@ frame=%@", view, NSStringFromCGRect(view.frame));
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
    UIView *view = [self currentHostView] ?: button.superview;
    if (!view) {
        return;
    }
    
    NJSponsorBlockPanelView *panel = [self sharedPanel];
    if (panel.superview) {
        [panel removeFromSuperview];
        return;
    }
    
    [self installInView:view];
    CGRect buttonFrame = [view convertRect:button.bounds fromView:button];
    CGRect frame = panel.frame;
    frame.origin.x = MIN(MAX(12, CGRectGetMinX(buttonFrame) - 206), CGRectGetWidth(view.bounds) - CGRectGetWidth(frame) - 12);
    frame.origin.y = CGRectGetMaxY(buttonFrame) + 10;
    panel.frame = frame;
    [panel keepInsideSuperview];
}

+ (void)layoutEntryButton:(UIButton *)button inView:(UIView *)view {
    UIEdgeInsets insets = view.safeAreaInsets;
    CGRect bounds = view.bounds;
    BOOL portrait = CGRectGetHeight(bounds) >= CGRectGetWidth(bounds);
    
    UIView *anchorView = NJSponsorBlockEntryAnchorView;
    if (anchorView && !anchorView.hidden && anchorView.alpha > 0.01 && anchorView.window) {
        CGRect anchorFrame = [anchorView convertRect:anchorView.bounds toView:nil];
        CGFloat x = CGRectGetMinX(anchorFrame) - 44.0;
        CGFloat y = CGRectGetMidY(anchorFrame) - 19.0;
        CGFloat maxX = CGRectGetWidth(bounds) - insets.right - 38.0 - 4.0;
        CGFloat maxY = CGRectGetHeight(bounds) - insets.bottom - 38.0 - 4.0;
        button.frame = CGRectMake(MIN(MAX(insets.left + 4.0, x), maxX),
                                  MIN(MAX(insets.top + 4.0, y), maxY),
                                  38.0,
                                  38.0);
        return;
    }
    
    CGFloat x = portrait ? CGRectGetWidth(bounds) - insets.right - 132.0 : CGRectGetWidth(bounds) - insets.right - 132.0;
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
    NSTimeInterval duration = manager.estimatedVideoDuration;
    if (duration <= 0 || manager.segments.count == 0) {
        return;
    }
    
    CGFloat width = CGRectGetWidth(timeline.bounds);
    CGFloat height = CGRectGetHeight(timeline.bounds);
    for (NJSponsorBlockSegment *segment in manager.segments) {
        CGFloat startX = MAX(0, MIN(width, width * segment.startTime / duration));
        CGFloat endX = MAX(startX + 2.0, MIN(width, width * segment.endTime / duration));
        UIView *mark = [[UIView alloc] initWithFrame:CGRectMake(startX, 0, endX - startX, height)];
        mark.backgroundColor = [self colorForCategory:segment.category];
        [timeline addSubview:mark];
    }
}

+ (UIColor *)colorForCategory:(NSString *)category {
    if ([category isEqualToString:@"sponsor"]) {
        return [UIColor colorWithRed:0 green:0.90 blue:0.10 alpha:0.95];
    }
    if ([category isEqualToString:@"intro"]) {
        return UIColor.cyanColor;
    }
    if ([category isEqualToString:@"outro"]) {
        return [UIColor colorWithRed:0.92 green:0.40 blue:0.95 alpha:0.95];
    }
    return [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:0.95];
}

+ (void)hidePanelOnly {
    [[self sharedPanel] removeFromSuperview];
}

- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
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

- (void)setupViews {
    self.backgroundColor = [UIColor colorWithWhite:0.08 alpha:0.88];
    self.layer.cornerRadius = 12;
    self.layer.borderWidth = 1;
    self.layer.borderColor = [UIColor colorWithWhite:1 alpha:0.14].CGColor;
    self.layer.masksToBounds = YES;
    
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
    
    self.collapseButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.collapseButton setTitle:@"－" forState:UIControlStateNormal];
    self.collapseButton.tintColor = UIColor.whiteColor;
    self.collapseButton.titleLabel.font = [UIFont boldSystemFontOfSize:20];
    [self.collapseButton addTarget:self action:@selector(toggleCollapsed) forControlEvents:UIControlEventTouchUpInside];
    
    UIStackView *headerTextStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.titleLabel, self.subtitleLabel]];
    headerTextStack.axis = UILayoutConstraintAxisVertical;
    headerTextStack.spacing = 2;
    
    UIStackView *headerStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.iconLabel, headerTextStack, self.collapseButton]];
    headerStack.axis = UILayoutConstraintAxisHorizontal;
    headerStack.alignment = UIStackViewAlignmentCenter;
    headerStack.spacing = 8;
    headerStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:headerStack];
    
    self.segmentStackView = [[UIStackView alloc] init];
    self.segmentStackView.axis = UILayoutConstraintAxisVertical;
    self.segmentStackView.spacing = 6;
    self.segmentStackView.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:self.segmentStackView];
    
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
    
    self.timeLabel = [[UILabel alloc] init];
    self.timeLabel.textColor = [UIColor colorWithWhite:0.82 alpha:1];
    self.timeLabel.font = [UIFont monospacedDigitSystemFontOfSize:12 weight:UIFontWeightSemibold];
    self.timeLabel.textAlignment = NSTextAlignmentRight;
    
    UIStackView *footerStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.toggleButton, self.timeLabel]];
    footerStack.axis = UILayoutConstraintAxisHorizontal;
    footerStack.alignment = UIStackViewAlignmentCenter;
    footerStack.distribution = UIStackViewDistributionFill;
    footerStack.spacing = 10;
    footerStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:footerStack];
    
    UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handlePan:)];
    [self addGestureRecognizer:pan];
    
    [NSLayoutConstraint activateConstraints:@[
        [self.iconLabel.widthAnchor constraintEqualToConstant:34],
        [self.collapseButton.widthAnchor constraintEqualToConstant:32],
        [self.toggleButton.widthAnchor constraintEqualToConstant:84],
        [self.toggleButton.heightAnchor constraintEqualToConstant:30],
        
        [headerStack.topAnchor constraintEqualToAnchor:self.topAnchor constant:12],
        [headerStack.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [headerStack.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-10],
        
        [self.segmentStackView.topAnchor constraintEqualToAnchor:headerStack.bottomAnchor constant:10],
        [self.segmentStackView.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.segmentStackView.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
        
        [self.progressView.topAnchor constraintEqualToAnchor:self.segmentStackView.bottomAnchor constant:10],
        [self.progressView.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.progressView.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
        [self.progressView.heightAnchor constraintEqualToConstant:6],
        
        [footerStack.topAnchor constraintEqualToAnchor:self.progressView.bottomAnchor constant:10],
        [footerStack.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [footerStack.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
        [footerStack.bottomAnchor constraintEqualToAnchor:self.bottomAnchor constant:-12],
    ]];
}

- (void)refreshContent {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self refreshContentOnMainThread];
    });
}

- (void)refreshContentOnMainThread {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    NSArray<NJSponsorBlockSegment *> *segments = manager.segments ?: @[];
    BOOL enabled = NJ_SPONSOR_BLOCK_VALUE;
    
    self.hidden = !NJ_MASTER_SWITCH_VALUE;
    self.titleLabel.text = @"小电视空降助手";
    if (segments.count > 0) {
        self.subtitleLabel.text = [NSString stringWithFormat:@"数据库中有 %lu 个可跳过片段", (unsigned long)segments.count];
    } else if (manager.videoID.length > 0) {
        self.subtitleLabel.text = @"当前视频暂无可跳过片段";
    } else {
        self.subtitleLabel.text = @"等待识别当前视频";
    }
    
    [self.toggleButton setTitle:(enabled ? @"已启用" : @"已关闭") forState:UIControlStateNormal];
    self.toggleButton.backgroundColor = enabled ? [UIColor colorWithRed:0 green:0.70 blue:0.05 alpha:1] : [UIColor colorWithWhite:0.30 alpha:1];
    [self.toggleButton setTitleColor:UIColor.whiteColor forState:UIControlStateNormal];
    self.timeLabel.text = [NSString stringWithFormat:@"当前 %@", [self stringFromTime:manager.currentPlaybackTime]];
    
    [self.segmentStackView.arrangedSubviews makeObjectsPerformSelector:@selector(removeFromSuperview)];
    [self renderPanelProgressWithSegments:segments duration:manager.estimatedVideoDuration];
    if (!self.collapsed) {
        NSUInteger count = MIN(segments.count, 3);
        for (NSUInteger i = 0; i < count; i++) {
            [self.segmentStackView addArrangedSubview:[self rowForSegment:segments[i] currentTime:manager.currentPlaybackTime]];
        }
        if (segments.count == 0) {
            [self.segmentStackView addArrangedSubview:[self emptyRow]];
        }
    }
    [self.collapseButton setTitle:(self.collapsed ? @"＋" : @"－") forState:UIControlStateNormal];
    [self resizeForContent];
    [[self class] renderTimeline:[[self class] sharedTimelineView]];
    [[self class] refreshNativeTimelines];
}

- (void)renderPanelProgressWithSegments:(NSArray<NJSponsorBlockSegment *> *)segments duration:(NSTimeInterval)duration {
    [self.progressView.subviews makeObjectsPerformSelector:@selector(removeFromSuperview)];
    if (duration <= 0 || segments.count == 0) {
        return;
    }
    
    CGFloat width = CGRectGetWidth(self.bounds) > 0 ? CGRectGetWidth(self.bounds) - 24.0 : NJSponsorBlockPanelWidth - 24.0;
    for (NJSponsorBlockSegment *segment in segments) {
        CGFloat startX = MAX(0, MIN(width, width * segment.startTime / duration));
        CGFloat endX = MAX(startX + 2.0, MIN(width, width * segment.endTime / duration));
        UIView *mark = [[UIView alloc] initWithFrame:CGRectMake(startX, 0, endX - startX, 6.0)];
        mark.backgroundColor = [[self class] colorForCategory:segment.category];
        [self.progressView addSubview:mark];
    }
}

- (UIView *)rowForSegment:(NJSponsorBlockSegment *)segment currentTime:(NSTimeInterval)currentTime {
    UIView *row = [[UIView alloc] init];
    BOOL active = [segment containsPlaybackTime:currentTime];
    row.backgroundColor = active ? [UIColor colorWithRed:0.00 green:0.55 blue:0.58 alpha:0.92] : [UIColor colorWithWhite:1 alpha:0.08];
    row.layer.cornerRadius = 8;
    
    UILabel *dot = [[UILabel alloc] init];
    dot.text = @"●";
    dot.textColor = [segment.category isEqualToString:@"sponsor"] ? [UIColor colorWithRed:0 green:0.90 blue:0.10 alpha:1] : UIColor.cyanColor;
    dot.font = [UIFont systemFontOfSize:14 weight:UIFontWeightBold];
    
    UILabel *categoryLabel = [[UILabel alloc] init];
    categoryLabel.text = [self titleForCategory:segment.category];
    categoryLabel.textColor = UIColor.whiteColor;
    categoryLabel.font = [UIFont systemFontOfSize:14 weight:UIFontWeightBold];
    
    UILabel *timeLabel = [[UILabel alloc] init];
    timeLabel.text = [NSString stringWithFormat:@"%@ 到 %@", [self stringFromTime:segment.startTime], [self stringFromTime:segment.endTime]];
    timeLabel.textColor = UIColor.whiteColor;
    timeLabel.font = [UIFont monospacedDigitSystemFontOfSize:13 weight:UIFontWeightSemibold];
    timeLabel.textAlignment = NSTextAlignmentRight;
    
    UIStackView *stack = [[UIStackView alloc] initWithArrangedSubviews:@[dot, categoryLabel, timeLabel]];
    stack.axis = UILayoutConstraintAxisHorizontal;
    stack.alignment = UIStackViewAlignmentCenter;
    stack.spacing = 8;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [row addSubview:stack];
    
    [NSLayoutConstraint activateConstraints:@[
        [row.heightAnchor constraintEqualToConstant:34],
        [dot.widthAnchor constraintEqualToConstant:16],
        [stack.leadingAnchor constraintEqualToAnchor:row.leadingAnchor constant:10],
        [stack.trailingAnchor constraintEqualToAnchor:row.trailingAnchor constant:-10],
        [stack.topAnchor constraintEqualToAnchor:row.topAnchor],
        [stack.bottomAnchor constraintEqualToAnchor:row.bottomAnchor],
    ]];
    return row;
}

- (UIView *)emptyRow {
    UILabel *label = [[UILabel alloc] init];
    label.text = @"未加载片段或该视频暂无数据";
    label.textColor = [UIColor colorWithWhite:0.72 alpha:1];
    label.font = [UIFont systemFontOfSize:13 weight:UIFontWeightMedium];
    label.textAlignment = NSTextAlignmentCenter;
    [label.heightAnchor constraintEqualToConstant:30].active = YES;
    return label;
}

- (void)toggleEnabled {
    BOOL next = !NJ_SPONSOR_BLOCK_VALUE;
    [NJ_SETTING_CACHE setObject:@(next) forKey:NJ_SPONSOR_BLOCK_KEY withBlock:nil];
    [self refreshContent];
}

- (void)toggleCollapsed {
    self.collapsed = !self.collapsed;
    [self refreshContent];
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
    CGFloat height = self.collapsed ? 72.0 : NJSponsorBlockPanelMinHeight;
    if (!self.collapsed) {
        height += MAX(0, self.segmentStackView.arrangedSubviews.count - 1) * 40.0;
    }
    
    CGRect frame = self.frame;
    frame.size.width = MIN(NJSponsorBlockPanelWidth, self.superview.bounds.size.width - 24);
    frame.size.height = height;
    self.frame = frame;
    [self keepInsideSuperview];
}

- (void)keepInsideSuperview {
    UIView *superview = self.superview;
    if (!superview) {
        return;
    }
    
    UIEdgeInsets insets = UIEdgeInsetsMake(48, 12, 24, 12);
    CGRect frame = self.frame;
    CGFloat maxX = CGRectGetWidth(superview.bounds) - insets.right - CGRectGetWidth(frame);
    CGFloat maxY = CGRectGetHeight(superview.bounds) - insets.bottom - CGRectGetHeight(frame);
    frame.origin.x = MIN(MAX(insets.left, frame.origin.x), MAX(insets.left, maxX));
    frame.origin.y = MIN(MAX(insets.top, frame.origin.y), MAX(insets.top, maxY));
    self.frame = frame;
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
