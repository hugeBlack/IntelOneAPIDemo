//
//  NJSponsorBlockPanelView.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockPanelView.h"
#import "../Settings/NJCommonDefine.h"
#import "../Models/NJSponsorBlockSegment.h"
#import "../Services/NJSponsorBlockService.h"
#import "../Settings/NJSponsorBlockSettings.h"
#import "NJSponsorBlockSubmissionManagerViewController.h"
#import "NJSponsorBlockTimelineView.h"
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

@interface NJSponsorBlockPanelView () <UIGestureRecognizerDelegate>

@property (nonatomic, strong) UIStackView *headerStack;
@property (nonatomic, strong) UIStackView *footerStack;
@property (nonatomic, strong) UILabel *iconLabel;
@property (nonatomic, strong) UILabel *titleLabel;
@property (nonatomic, strong) UILabel *subtitleLabel;
@property (nonatomic, strong) UIScrollView *segmentScrollView;
@property (nonatomic, strong) UIStackView *segmentStackView;
@property (nonatomic, strong) UIView *progressView;
@property (nonatomic, strong) UIButton *refreshButton;
@property (nonatomic, strong) UIButton *toggleButton;
@property (nonatomic, strong) UIButton *submitButton;
@property (nonatomic, strong) UILabel *statsLabel;
@property (nonatomic, strong) NJSponsorBlockService *service;
@property (nonatomic, strong) UIView *progressPlayheadView;
@property (nonatomic, copy) NSArray<NJSponsorBlockSegment *> *displayedSegments;
@property (nonatomic, copy) NSArray<UIView *> *segmentRowViews;

@property (nonatomic, weak) NJSponsorBlockManager* manager;

- (UIButton *)actionButtonWithTitle:(NSString *)title color:(UIColor *)color action:(SEL)action segment:(NJSponsorBlockSegment *)segment;
- (void)updateHeaderWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (void)updateEnabledButton:(BOOL)enabled;
- (void)updateFooterWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments;
- (void)rebuildSegmentRowsWithSegments:(NSArray<NJSponsorBlockSegment *> *)segments manager:(NJSponsorBlockManager *)manager;
- (NJSponsorBlockSegment *)segmentFromSender:(id)sender;
- (void)voteForSegment:(NJSponsorBlockSegment *)segment type:(NSInteger)type;
- (void)submitSegmentTapped:(UIButton *)button;
- (void)presentSubmissionMenuFromView:(UIView *)sourceView;
- (BOOL)validateCurrentVideoForDraftCreation;
- (void)presentSubmissionCategoryPickerFromView:(UIView *)sourceView;
- (void)presentSubmissionManager;
- (void)submitCurrentVideoDraftsFromPanel;
- (void)beginSubmissionWithCategory:(NSString *)category;
- (UIViewController *)presentationViewController;
- (NSTimeInterval)durationForSegment:(NJSponsorBlockSegment *)segment;
- (CGFloat)segmentRowsContentHeight;
- (NSString *)detailTextForSegment:(NJSponsorBlockSegment *)segment currentTime:(NSTimeInterval)currentTime;
- (NSString *)compactStringFromTime:(NSTimeInterval)time;
- (void)onPlaybackTimeChanged;
- (void)updateForPlaybackTimeOnMainThread;
- (void)updateProgressPlayheadForTime:(NSTimeInterval)time duration:(NSTimeInterval)duration;
- (void)updateSegmentRowHighlightsForTime:(NSTimeInterval)time manager:(NJSponsorBlockManager *)manager;

@end

@implementation NJSponsorBlockPanelView

static UIViewController *NJSponsorBlockSharedOverlayController;
static void *NJSponsorBlockManualSkipSegmentKey = &NJSponsorBlockManualSkipSegmentKey;
static void *NJSponsorBlockPanelSegmentKey = &NJSponsorBlockPanelSegmentKey;

+ (UIColor *)colorForCategory:(NSString *)category {
    return [NJSponsorBlockSettings colorForCategory:category];
}

- (instancetype)initWithManager:(NJSponsorBlockManager*)manager {
    CGRect frame = CGRectMake(16, 88, NJSponsorBlockPanelWidth, NJSponsorBlockPanelMinHeight);
    return [self initWithFrame:frame manager:manager];
}

- (instancetype)initWithFrame:(CGRect)frame manager:(NJSponsorBlockManager*)manager {
    self = [super initWithFrame:frame];
    if (self) {
        self.manager = manager;
        self.service = [[NJSponsorBlockService alloc] init];
        [self setupViews];
        [self refreshContent];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(refreshContent)
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
    self.layer.borderWidth = 0.5;
    self.layer.borderColor = [UIColor colorWithWhite:1 alpha:0.16].CGColor;
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


    self.footerStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.toggleButton, self.submitButton, self.statsLabel]];
    self.footerStack.axis = UILayoutConstraintAxisHorizontal;
    self.footerStack.alignment = UIStackViewAlignmentCenter;
    self.footerStack.distribution = UIStackViewDistributionFill;
    self.footerStack.spacing = 8;
    self.footerStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self addSubview:self.footerStack];

    self.refreshButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [self.refreshButton setImage:[UIImage systemImageNamed:@"arrow.clockwise"] forState:UIControlStateNormal];
    [self.refreshButton setTintColor:[UIColor colorWithWhite:0.72 alpha:1]];
    self.refreshButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.refreshButton addTarget:self action:@selector(refreshButtonTapped:) forControlEvents:UIControlEventTouchUpInside];
    [self addSubview:self.refreshButton];

    [NSLayoutConstraint activateConstraints:@[
        [self.iconLabel.widthAnchor constraintEqualToConstant:34],
        [self.toggleButton.widthAnchor constraintEqualToConstant:70],
        [self.toggleButton.heightAnchor constraintEqualToConstant:30],
        [self.submitButton.widthAnchor constraintEqualToConstant:52],
        [self.submitButton.heightAnchor constraintEqualToConstant:30],

        [self.refreshButton.topAnchor constraintEqualToAnchor:self.topAnchor constant:16],
        [self.refreshButton.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-16],
        [self.refreshButton.widthAnchor constraintEqualToConstant:20],
        [self.refreshButton.heightAnchor constraintEqualToConstant:20],

        [self.headerStack.topAnchor constraintEqualToAnchor:self.topAnchor constant:12],
        [self.headerStack.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.headerStack.trailingAnchor constraintEqualToAnchor:self.refreshButton.leadingAnchor constant:-4],

        [self.segmentScrollView.topAnchor constraintEqualToAnchor:self.headerStack.bottomAnchor constant:10],
        [self.segmentScrollView.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:12],
        [self.segmentScrollView.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-12],
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
    NJSponsorBlockManager *manager = _manager;
    NSArray<NJSponsorBlockSegment *> *segments = [manager displaySegments] ?: @[];

    [self updateHeaderWithManager:manager segments:segments];
    [self updateEnabledButton:[NJSponsorBlockSettings enabled]];
    [self updateFooterWithManager:manager segments:segments];
    [self rebuildSegmentRowsWithSegments:segments manager:manager];
    [self renderPanelProgressWithSegments:segments duration:manager.estimatedVideoDuration];
    
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
    self.submitButton.enabled = enabled && !_manager.isSubmissionInFlight;
    self.submitButton.alpha = self.submitButton.enabled ? 1.0 : 0.45;
}

- (void)updateFooterWithManager:(NJSponsorBlockManager *)manager segments:(NSArray<NJSponsorBlockSegment *> *)segments {
    NSTimeInterval skippedDuration = [manager skippedDurationBeforePlaybackTime:manager.currentPlaybackTime];
    NSUInteger totalCount = [manager allSegments].count;
    NSString *segmentCountText = totalCount > segments.count ? [NSString stringWithFormat:@"%lu/%lu 段", (unsigned long)segments.count, (unsigned long)totalCount] : [NSString stringWithFormat:@"%lu 段", (unsigned long)segments.count];
    self.statsLabel.text = [NSString stringWithFormat:@"%@ · 省 %@",
                            segmentCountText,
                            [self compactStringFromTime:skippedDuration]];
}

- (void)rebuildSegmentRowsWithSegments:(NSArray<NJSponsorBlockSegment *> *)segments manager:(NJSponsorBlockManager *)manager {
    [self clearSegmentRows];

    NSMutableArray<UIView *> *rowViews = [NSMutableArray array];
    UIView *activeRow = nil;
    for (NJSponsorBlockSegment *segment in segments) {
        UIView *row = [self rowForSegment:segment currentTime:manager.currentPlaybackTime];
        [self.segmentStackView addArrangedSubview:row];
        [rowViews addObject:row];
        if (!activeRow && [segment containsPlaybackTime:manager.currentPlaybackTime]) {
            activeRow = row;
        }
    }
    self.displayedSegments = [segments copy];
    self.segmentRowViews = [rowViews copy];

    if (segments.count == 0) {
        [self.segmentStackView addArrangedSubview:[self emptyRow]];
    } else if (activeRow) {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self.segmentScrollView scrollRectToVisible:activeRow.frame animated:NO];
        });
    }
}

- (void)clearSegmentRows {
    for (UIView *view in self.segmentStackView.arrangedSubviews.copy) {
        [self.segmentStackView removeArrangedSubview:view];
        [view removeFromSuperview];
    }
    self.displayedSegments = nil;
    self.segmentRowViews = nil;
}

- (void)onPlaybackTimeChanged {
    if ([NSThread isMainThread]) {
        [self updateForPlaybackTimeOnMainThread];
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{
            [self updateForPlaybackTimeOnMainThread];
        });
    }
}

- (void)updateForPlaybackTimeOnMainThread {
    NJSponsorBlockManager *manager = _manager;
    NSTimeInterval time = manager.currentPlaybackTime;
    NSTimeInterval duration = manager.estimatedVideoDuration;

    if (self.superview) {
        NSArray<NJSponsorBlockSegment *> *segments = self.displayedSegments ?: @[];
        [self updateFooterWithManager:manager segments:segments];
        [self updateProgressPlayheadForTime:time duration:duration];
        [self updateSegmentRowHighlightsForTime:time manager:manager];
    }
}

- (void)updateProgressPlayheadForTime:(NSTimeInterval)time duration:(NSTimeInterval)duration {
    if (!self.progressPlayheadView || duration <= 0) {
        return;
    }
    CGFloat width = CGRectGetWidth(self.progressView.bounds);
    if (width <= 0) {
        width = NJSponsorBlockPanelWidth - 24.0;
    }
    CGFloat playheadX = MAX(0, MIN(width, width * time / duration));
    self.progressPlayheadView.frame = CGRectMake(playheadX - 1.0, 0, 2.0, 6.0);
}

- (void)updateSegmentRowHighlightsForTime:(NSTimeInterval)time manager:(NJSponsorBlockManager *)manager {
    NSArray<NJSponsorBlockSegment *> *segments = self.displayedSegments ?: @[];
    NSArray<UIView *> *rows = self.segmentRowViews ?: @[];
    NSUInteger count = MIN(segments.count, rows.count);
    for (NSUInteger i = 0; i < count; i++) {
        NJSponsorBlockSegment *segment = segments[i];
        UIView *row = rows[i];
        BOOL active = [segment containsPlaybackTime:time];
        BOOL skipped = [manager hasActuallySkippedSegment:segment];
        if (active) {
            row.backgroundColor = [UIColor colorWithRed:0.00 green:0.55 blue:0.58 alpha:0.92];
        } else if (skipped) {
            row.backgroundColor = [UIColor colorWithRed:0.18 green:0.45 blue:0.20 alpha:0.82];
        } else {
            row.backgroundColor = [UIColor colorWithWhite:1 alpha:0.08];
        }
    }
}


- (void)renderPanelProgressWithSegments:(NSArray<NJSponsorBlockSegment *> *)segments duration:(NSTimeInterval)duration {
    [self.progressView.subviews makeObjectsPerformSelector:@selector(removeFromSuperview)];
    self.progressPlayheadView = nil;
    if (duration <= 0 || segments.count == 0) {
        return;
    }

    NJSponsorBlockManager *manager = _manager;
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
    self.progressPlayheadView = playhead;
}

- (UIView *)rowForSegment:(NJSponsorBlockSegment *)segment currentTime:(NSTimeInterval)currentTime {
    UIView *row = [[UIView alloc] init];
    BOOL active = [segment containsPlaybackTime:currentTime];
    BOOL skipped = [_manager hasActuallySkippedSegment:segment];
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

- (void)refreshButtonTapped:(UIButton *)button {
    [_manager refresh];
}

- (void)toggleEnabled {
    [NJSponsorBlockSettings setEnabled:![NJSponsorBlockSettings enabled]];
    [self refreshContent];
}

- (void)submitSegmentTapped:(UIButton *)button {
    if (![NJSponsorBlockSettings enabled]) {
        [_manager showInfoToast:@"无法提交" detail:@"请先启用 SponsorBlock"];
        return;
    }
    if (_manager.isSubmissionInFlight) {
        [_manager showInfoToast:@"正在提交" detail:@"请等待当前请求完成"];
        return;
    }
    if (_manager.submissionDraftInProgress) {
        return; // 提交草稿流程由 Manager 的 Toast 管理
    }
    [self presentSubmissionMenuFromView:button];
}

- (void)presentSubmissionMenuFromView:(UIView *)sourceView {
    UIViewController *presenter = [self presentationViewController];
    if (!presenter) {
        [_manager showInfoToast:@"无法提交" detail:@"无法打开提交菜单"];
        return;
    }

    NSUInteger draftCount = _manager.unsubmittedSegmentsForCurrentVideo.count;
    NSString *message = draftCount > 0 ? [NSString stringWithFormat:@"当前视频有 %lu 个未提交片段", (unsigned long)draftCount] : @"可保存新草稿或管理已有草稿";
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"片段提交" message:message preferredStyle:UIAlertControllerStyleActionSheet];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"新增片段草稿" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        __strong typeof(weakSelf) strongSelf = weakSelf;

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


- (void)presentSubmissionCategoryPickerFromView:(UIView *)sourceView {
    UIViewController *presenter = [self presentationViewController];

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
            [strongSelf->_manager beginSubmissionDraftWithCategory:option.category];
            
        }]];
    }
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    alert.popoverPresentationController.sourceView = sourceView ?: self;
    alert.popoverPresentationController.sourceRect = sourceView ? sourceView.bounds : self.bounds;
    [presenter presentViewController:alert animated:YES completion:nil];
}

- (void)presentSubmissionManager {
    UIViewController *presenter = [self presentationViewController];

    NJSponsorBlockSubmissionManagerViewController *controller = [[NJSponsorBlockSubmissionManagerViewController alloc] initWithManager:_manager];
    UINavigationController *navigationController = [[UINavigationController alloc] initWithRootViewController:controller];
    navigationController.modalPresentationStyle = UIModalPresentationPageSheet;
    [presenter presentViewController:navigationController animated:YES completion:nil];
}

- (UIViewController *)presentationViewController {
    // https://stackoverflow.com/a/12418527
    return [[[[UIApplication sharedApplication] delegate] window] rootViewController];
}

- (NJSponsorBlockSegment *)segmentFromSender:(id)sender {
    return [sender isKindOfClass:[UIView class]] ? objc_getAssociatedObject(sender, NJSponsorBlockPanelSegmentKey) : nil;
}

- (void)manualSkipButtonTapped:(UIButton *)button {
    NJSponsorBlockSegment *segment = [self segmentFromSender:button] ?: objc_getAssociatedObject(button, NJSponsorBlockManualSkipSegmentKey);
    if (!segment) {
        return;
    }
    [_manager skipSegment:segment];
}

- (void)seekToSegmentStartTapped:(UIButton *)button {
    NJSponsorBlockSegment *segment = [self segmentFromSender:button];
    if (!segment) {
        return;
    }
    [_manager seekTo:segment.startTime];
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
        [_manager showInfoToast:@"复制失败" detail:@"片段 UUID 为空"];
        return;
    }
    UIPasteboard.generalPasteboard.string = segment.uuid;
    [_manager showInfoToast:@"已复制 UUID" detail:segment.uuid];
}

- (void)voteForSegment:(NJSponsorBlockSegment *)segment type:(NSInteger)type {
    if (segment.isUnsubmitted) {
        [_manager showInfoToast:@"无法投票" detail:@"本地未提交片段不能投票"];
        return;
    }
    if (segment.uuid.length == 0) {
        [_manager showInfoToast:@"无法投票" detail:@"片段 UUID 为空"];
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
                [strongSelf->_manager showInfoToast:(type == 1 ? @"已点赞" : @"已点踩") detail:@"感谢反馈"];
            } else {
                [strongSelf->_manager showInfoToast:@"投票失败" detail:error.localizedDescription ?: @"请稍后重试"];
            }
        });
    }];
}

- (void)progressSegmentTapped:(UITapGestureRecognizer *)gesture {
    NJSponsorBlockSegment *segment = objc_getAssociatedObject(gesture.view, NJSponsorBlockPanelSegmentKey);
    if (!segment) {
        return;
    }
    [_manager seekTo:segment.startTime];
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
