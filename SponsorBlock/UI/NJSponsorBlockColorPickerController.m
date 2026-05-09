//
//  NJSponsorBlockColorPickerController.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockColorPickerController.h"

#pragma mark - Preset Colors Grid

@interface NJPresetColorsView : UIView
@property (nonatomic, strong) NSArray<UIColor *> *colors;
@property (nonatomic, copy) void (^onColorSelected)(UIColor *color);
@end

@implementation NJPresetColorsView

- (void)layoutSubviews {
    [super layoutSubviews];
    for (UIView *subview in self.subviews) {
        [subview removeFromSuperview];
    }
    if (self.colors.count == 0) return;

    CGFloat spacing = 8;
    CGFloat itemSize = 44;
    NSInteger columns = MAX(3, (NSInteger)((self.bounds.size.width + spacing) / (itemSize + spacing)));
    columns = MIN(columns, (NSInteger)self.colors.count);
    CGFloat usedWidth = columns * itemSize + (columns - 1) * spacing;
    CGFloat startX = MAX(0, floor((self.bounds.size.width - usedWidth) / 2.0));
    for (NSUInteger i = 0; i < self.colors.count; i++) {
        NSInteger row = i / columns;
        NSInteger col = i % columns;
        CGFloat x = startX + col * (itemSize + spacing);
        CGFloat y = row * (itemSize + spacing);
        UIButton *btn = [UIButton buttonWithType:UIButtonTypeCustom];
        btn.frame = CGRectMake(x, y, itemSize, itemSize);
        btn.layer.cornerRadius = itemSize / 2;
        btn.layer.borderWidth = 2;
        btn.layer.borderColor = [UIColor separatorColor].CGColor;
        btn.backgroundColor = self.colors[i];
        btn.tag = i;
        btn.accessibilityLabel = [NSString stringWithFormat:@"快速选择 %lu", (unsigned long)i + 1];
        [btn addTarget:self action:@selector(presetTapped:) forControlEvents:UIControlEventTouchUpInside];
        [self addSubview:btn];
    }
}

- (CGFloat)preferredHeightForWidth:(CGFloat)width {
    if (self.colors.count == 0 || width <= 0) {
        return 0;
    }
    CGFloat spacing = 8;
    CGFloat itemSize = 44;
    NSInteger columns = MAX(3, (NSInteger)((width + spacing) / (itemSize + spacing)));
    columns = MIN(columns, (NSInteger)self.colors.count);
    NSInteger rows = (self.colors.count + columns - 1) / columns;
    return rows * itemSize + MAX(0, rows - 1) * spacing;
}

- (void)presetTapped:(UIButton *)sender {
    if (self.onColorSelected && sender.tag < (NSInteger)self.colors.count) {
        self.onColorSelected(self.colors[sender.tag]);
    }
}

@end

#pragma mark - Main Color Picker

@interface NJSponsorBlockColorPickerController ()

@property (nonatomic, strong) UIColor *currentColor;
@property (nonatomic, copy) NSString *categoryTitle;

@property (nonatomic, strong) UIScrollView *scrollView;
@property (nonatomic, strong) UIView *contentView;
@property (nonatomic, strong) UIView *headerView;
@property (nonatomic, strong) UISegmentedControl *modeControl;
@property (nonatomic, strong) UIView *previewView;
@property (nonatomic, strong) UILabel *previewLabel;
@property (nonatomic, strong) NJPresetColorsView *presetView;
@property (nonatomic, strong) UILabel *hexLabel;
@property (nonatomic, strong) UIView *modeContainer;
@property (nonatomic, strong) UIView *gridContainer;
@property (nonatomic, strong) UIView *slidersContainer;
@property (nonatomic, strong) UISlider *redSlider;
@property (nonatomic, strong) UISlider *greenSlider;
@property (nonatomic, strong) UISlider *blueSlider;
@property (nonatomic, strong) UILabel *redValueLabel;
@property (nonatomic, strong) UILabel *greenValueLabel;
@property (nonatomic, strong) UILabel *blueValueLabel;
@property (nonatomic, strong) NSLayoutConstraint *presetHeightConstraint;
@property (nonatomic, strong) NSLayoutConstraint *modeHeightConstraint;

@end

@implementation NJSponsorBlockColorPickerController

- (instancetype)initWithColor:(UIColor *)color categoryTitle:(NSString *)categoryTitle {
    self = [super init];
    if (self) {
        _currentColor = color ?: [UIColor whiteColor];
        _categoryTitle = categoryTitle ?: @"";
        self.modalPresentationStyle = UIModalPresentationPageSheet;
    }
    return self;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor colorWithRed:0.05 green:0.05 blue:0.06 alpha:1.0];
    self.title = nil;
    self.navigationController.navigationBarHidden = YES;
    self.preferredContentSize = CGSizeMake(0, 620);
    if (@available(iOS 15.0, *)) {
        UISheetPresentationController *sheet = self.navigationController.sheetPresentationController ?: self.sheetPresentationController;
        sheet.detents = @[[UISheetPresentationControllerDetent mediumDetent], [UISheetPresentationControllerDetent largeDetent]];
        sheet.prefersGrabberVisible = YES;
        sheet.preferredCornerRadius = 28;
    }

    self.scrollView = [[UIScrollView alloc] init];
    self.scrollView.alwaysBounceVertical = YES;
    self.scrollView.showsVerticalScrollIndicator = NO;
    self.scrollView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.scrollView];

    self.contentView = [[UIView alloc] init];
    self.contentView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.scrollView addSubview:self.contentView];

    self.headerView = [[UIView alloc] init];
    self.headerView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.contentView addSubview:self.headerView];

    UIImageView *pipetteView = [[UIImageView alloc] init];
    pipetteView.translatesAutoresizingMaskIntoConstraints = NO;
    if (@available(iOS 13.0, *)) {
        pipetteView.image = [UIImage systemImageNamed:@"eyedropper.full"];
    }
    pipetteView.tintColor = UIColor.whiteColor;
    pipetteView.contentMode = UIViewContentModeScaleAspectFit;
    [self.headerView addSubview:pipetteView];

    UILabel *titleLabel = [[UILabel alloc] init];
    titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
    titleLabel.text = self.categoryTitle.length > 0 ? self.categoryTitle : @"颜色";
    titleLabel.textColor = UIColor.whiteColor;
    titleLabel.font = [UIFont systemFontOfSize:22 weight:UIFontWeightBold];
    titleLabel.textAlignment = NSTextAlignmentCenter;
    titleLabel.adjustsFontSizeToFitWidth = YES;
    titleLabel.minimumScaleFactor = 0.78;
    [self.headerView addSubview:titleLabel];

    UIButton *closeButton = [UIButton buttonWithType:UIButtonTypeSystem];
    closeButton.translatesAutoresizingMaskIntoConstraints = NO;
    if (@available(iOS 13.0, *)) {
        [closeButton setImage:[UIImage systemImageNamed:@"xmark"] forState:UIControlStateNormal];
    } else {
        [closeButton setTitle:@"×" forState:UIControlStateNormal];
    }
    closeButton.tintColor = UIColor.whiteColor;
    closeButton.accessibilityLabel = @"完成";
    [closeButton addTarget:self action:@selector(doneTapped) forControlEvents:UIControlEventTouchUpInside];
    [self.headerView addSubview:closeButton];

    self.modeControl = [[UISegmentedControl alloc] initWithItems:@[@"网格", @"滑杆"]];
    self.modeControl.translatesAutoresizingMaskIntoConstraints = NO;
    self.modeControl.selectedSegmentIndex = 0;
    if (@available(iOS 13.0, *)) {
        self.modeControl.selectedSegmentTintColor = [UIColor colorWithWhite:0.46 alpha:1.0];
    }
    [self.modeControl setTitleTextAttributes:@{NSForegroundColorAttributeName: UIColor.whiteColor,
                                               NSFontAttributeName: [UIFont systemFontOfSize:15 weight:UIFontWeightSemibold]}
                                    forState:UIControlStateNormal];
    [self.modeControl setTitleTextAttributes:@{NSForegroundColorAttributeName: UIColor.whiteColor,
                                               NSFontAttributeName: [UIFont systemFontOfSize:15 weight:UIFontWeightBold]}
                                    forState:UIControlStateSelected];
    [self.modeControl addTarget:self action:@selector(modeChanged:) forControlEvents:UIControlEventValueChanged];
    [self.contentView addSubview:self.modeControl];

    self.previewView = [[UIView alloc] init];
    self.previewView.translatesAutoresizingMaskIntoConstraints = NO;
    self.previewView.layer.cornerRadius = 20;
    self.previewView.layer.borderWidth = 1.0;
    self.previewView.layer.borderColor = [UIColor colorWithWhite:1 alpha:0.16].CGColor;
    self.previewView.clipsToBounds = YES;
    [self.contentView addSubview:self.previewView];

    self.previewLabel = [[UILabel alloc] init];
    self.previewLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.previewLabel.textColor = [UIColor whiteColor];
    self.previewLabel.font = [UIFont systemFontOfSize:18 weight:UIFontWeightBold];
    self.previewLabel.textAlignment = NSTextAlignmentCenter;
    self.previewLabel.text = @"当前选择";
    [self.previewView addSubview:self.previewLabel];

    self.hexLabel = [[UILabel alloc] init];
    self.hexLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.hexLabel.font = [UIFont monospacedSystemFontOfSize:16 weight:UIFontWeightSemibold];
    self.hexLabel.textColor = [UIColor colorWithWhite:0.84 alpha:1.0];
    self.hexLabel.textAlignment = NSTextAlignmentCenter;
    [self.contentView addSubview:self.hexLabel];

    self.modeContainer = [[UIView alloc] init];
    self.modeContainer.translatesAutoresizingMaskIntoConstraints = NO;
    [self.contentView addSubview:self.modeContainer];

    self.gridContainer = [[UIView alloc] init];
    self.gridContainer.translatesAutoresizingMaskIntoConstraints = NO;
    [self.modeContainer addSubview:self.gridContainer];

    self.slidersContainer = [[UIView alloc] init];
    self.slidersContainer.translatesAutoresizingMaskIntoConstraints = NO;
    [self.modeContainer addSubview:self.slidersContainer];

    __weak typeof(self) weakSelf = self;
    UILabel *presetTitle = [self sectionLabelWithText:@"快速选择"];
    [self.gridContainer addSubview:presetTitle];

    self.presetView = [[NJPresetColorsView alloc] init];
    self.presetView.translatesAutoresizingMaskIntoConstraints = NO;
    self.presetView.colors = [self presetColors];
    self.presetView.onColorSelected = ^(UIColor *color) {
        weakSelf.currentColor = color;
        [weakSelf updatePreview];
        [weakSelf syncSlidersFromCurrentColor];
    };
    [self.gridContainer addSubview:self.presetView];

    UILabel *slidersTitle = [self sectionLabelWithText:@"RGB"];
    [self.slidersContainer addSubview:slidersTitle];
    self.redSlider = [self colorSliderWithTint:[UIColor systemRedColor]];
    self.greenSlider = [self colorSliderWithTint:[UIColor systemGreenColor]];
    self.blueSlider = [self colorSliderWithTint:[UIColor systemBlueColor]];
    self.redValueLabel = [self valueLabel];
    self.greenValueLabel = [self valueLabel];
    self.blueValueLabel = [self valueLabel];
    UIView *redRow = [self sliderRowWithTitle:@"R" slider:self.redSlider valueLabel:self.redValueLabel];
    UIView *greenRow = [self sliderRowWithTitle:@"G" slider:self.greenSlider valueLabel:self.greenValueLabel];
    UIView *blueRow = [self sliderRowWithTitle:@"B" slider:self.blueSlider valueLabel:self.blueValueLabel];
    [self.slidersContainer addSubview:redRow];
    [self.slidersContainer addSubview:greenRow];
    [self.slidersContainer addSubview:blueRow];

    CGFloat padding = 16;
    self.presetHeightConstraint = [self.presetView.heightAnchor constraintEqualToConstant:96];
    self.modeHeightConstraint = [self.modeContainer.heightAnchor constraintEqualToConstant:160];
    [NSLayoutConstraint activateConstraints:@[
        [self.scrollView.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.scrollView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.scrollView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.scrollView.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],
        [self.contentView.topAnchor constraintEqualToAnchor:self.scrollView.contentLayoutGuide.topAnchor],
        [self.contentView.leadingAnchor constraintEqualToAnchor:self.scrollView.contentLayoutGuide.leadingAnchor],
        [self.contentView.trailingAnchor constraintEqualToAnchor:self.scrollView.contentLayoutGuide.trailingAnchor],
        [self.contentView.bottomAnchor constraintEqualToAnchor:self.scrollView.contentLayoutGuide.bottomAnchor],
        [self.contentView.widthAnchor constraintEqualToAnchor:self.scrollView.frameLayoutGuide.widthAnchor],

        [self.headerView.topAnchor constraintEqualToAnchor:self.contentView.topAnchor constant:18],
        [self.headerView.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor constant:padding],
        [self.headerView.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor constant:-padding],
        [self.headerView.heightAnchor constraintEqualToConstant:44],
        [pipetteView.leadingAnchor constraintEqualToAnchor:self.headerView.leadingAnchor],
        [pipetteView.centerYAnchor constraintEqualToAnchor:self.headerView.centerYAnchor],
        [pipetteView.widthAnchor constraintEqualToConstant:34],
        [pipetteView.heightAnchor constraintEqualToConstant:34],
        [closeButton.trailingAnchor constraintEqualToAnchor:self.headerView.trailingAnchor],
        [closeButton.centerYAnchor constraintEqualToAnchor:self.headerView.centerYAnchor],
        [closeButton.widthAnchor constraintEqualToConstant:44],
        [closeButton.heightAnchor constraintEqualToConstant:44],
        [titleLabel.leadingAnchor constraintEqualToAnchor:pipetteView.trailingAnchor constant:8],
        [titleLabel.trailingAnchor constraintEqualToAnchor:closeButton.leadingAnchor constant:-8],
        [titleLabel.centerYAnchor constraintEqualToAnchor:self.headerView.centerYAnchor],

        [self.modeControl.topAnchor constraintEqualToAnchor:self.headerView.bottomAnchor constant:18],
        [self.modeControl.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor constant:padding],
        [self.modeControl.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor constant:-padding],
        [self.modeControl.heightAnchor constraintEqualToConstant:44],

        [self.previewView.topAnchor constraintEqualToAnchor:self.modeControl.bottomAnchor constant:18],
        [self.previewView.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor constant:padding],
        [self.previewView.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor constant:-padding],
        [self.previewView.heightAnchor constraintEqualToConstant:72],
        [self.previewLabel.leadingAnchor constraintEqualToAnchor:self.previewView.leadingAnchor constant:12],
        [self.previewLabel.trailingAnchor constraintEqualToAnchor:self.previewView.trailingAnchor constant:-12],
        [self.previewLabel.centerYAnchor constraintEqualToAnchor:self.previewView.centerYAnchor],

        [self.hexLabel.topAnchor constraintEqualToAnchor:self.previewView.bottomAnchor constant:10],
        [self.hexLabel.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor constant:padding],
        [self.hexLabel.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor constant:-padding],
        [self.hexLabel.heightAnchor constraintEqualToConstant:24],

        [self.modeContainer.topAnchor constraintEqualToAnchor:self.hexLabel.bottomAnchor constant:18],
        [self.modeContainer.leadingAnchor constraintEqualToAnchor:self.contentView.leadingAnchor constant:padding],
        [self.modeContainer.trailingAnchor constraintEqualToAnchor:self.contentView.trailingAnchor constant:-padding],
        self.modeHeightConstraint,

        [self.gridContainer.topAnchor constraintEqualToAnchor:self.modeContainer.topAnchor],
        [self.gridContainer.leadingAnchor constraintEqualToAnchor:self.modeContainer.leadingAnchor],
        [self.gridContainer.trailingAnchor constraintEqualToAnchor:self.modeContainer.trailingAnchor],
        [self.gridContainer.bottomAnchor constraintEqualToAnchor:self.modeContainer.bottomAnchor],
        [presetTitle.topAnchor constraintEqualToAnchor:self.gridContainer.topAnchor],
        [presetTitle.leadingAnchor constraintEqualToAnchor:self.gridContainer.leadingAnchor],
        [presetTitle.trailingAnchor constraintEqualToAnchor:self.gridContainer.trailingAnchor],
        [self.presetView.topAnchor constraintEqualToAnchor:presetTitle.bottomAnchor constant:10],
        [self.presetView.leadingAnchor constraintEqualToAnchor:self.gridContainer.leadingAnchor],
        [self.presetView.trailingAnchor constraintEqualToAnchor:self.gridContainer.trailingAnchor],
        self.presetHeightConstraint,

        [self.slidersContainer.topAnchor constraintEqualToAnchor:self.modeContainer.topAnchor],
        [self.slidersContainer.leadingAnchor constraintEqualToAnchor:self.modeContainer.leadingAnchor],
        [self.slidersContainer.trailingAnchor constraintEqualToAnchor:self.modeContainer.trailingAnchor],
        [self.slidersContainer.bottomAnchor constraintEqualToAnchor:self.modeContainer.bottomAnchor],
        [slidersTitle.topAnchor constraintEqualToAnchor:self.slidersContainer.topAnchor],
        [slidersTitle.leadingAnchor constraintEqualToAnchor:self.slidersContainer.leadingAnchor],
        [slidersTitle.trailingAnchor constraintEqualToAnchor:self.slidersContainer.trailingAnchor],
        [redRow.topAnchor constraintEqualToAnchor:slidersTitle.bottomAnchor constant:12],
        [greenRow.topAnchor constraintEqualToAnchor:redRow.bottomAnchor constant:12],
        [blueRow.topAnchor constraintEqualToAnchor:greenRow.bottomAnchor constant:12],
        [redRow.leadingAnchor constraintEqualToAnchor:self.slidersContainer.leadingAnchor],
        [redRow.trailingAnchor constraintEqualToAnchor:self.slidersContainer.trailingAnchor],
        [greenRow.leadingAnchor constraintEqualToAnchor:self.slidersContainer.leadingAnchor],
        [greenRow.trailingAnchor constraintEqualToAnchor:self.slidersContainer.trailingAnchor],
        [blueRow.leadingAnchor constraintEqualToAnchor:self.slidersContainer.leadingAnchor],
        [blueRow.trailingAnchor constraintEqualToAnchor:self.slidersContainer.trailingAnchor],
        [redRow.heightAnchor constraintEqualToConstant:36],
        [greenRow.heightAnchor constraintEqualToConstant:36],
        [blueRow.heightAnchor constraintEqualToConstant:36],
        [self.modeContainer.bottomAnchor constraintEqualToAnchor:self.contentView.bottomAnchor constant:-24],
    ]];

    [self syncSlidersFromCurrentColor];
    [self updateVisibleMode];
    [self updatePreview];
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    CGFloat width = CGRectGetWidth(self.presetView.bounds);
    CGFloat height = [self.presetView preferredHeightForWidth:width];
    if (height > 0 && fabs(self.presetHeightConstraint.constant - height) > 0.5) {
        self.presetHeightConstraint.constant = height;
        [self updateVisibleMode];
    }
}

- (UILabel *)sectionLabelWithText:(NSString *)text {
    UILabel *label = [[UILabel alloc] init];
    label.translatesAutoresizingMaskIntoConstraints = NO;
    label.text = text;
    label.font = [UIFont systemFontOfSize:13 weight:UIFontWeightSemibold];
    label.textColor = [UIColor colorWithWhite:0.68 alpha:1.0];
    return label;
}

- (UISlider *)colorSliderWithTint:(UIColor *)tintColor {
    UISlider *slider = [[UISlider alloc] init];
    slider.translatesAutoresizingMaskIntoConstraints = NO;
    slider.minimumValue = 0;
    slider.maximumValue = 255;
    slider.minimumTrackTintColor = tintColor;
    slider.maximumTrackTintColor = [UIColor colorWithWhite:0.24 alpha:1.0];
    [slider addTarget:self action:@selector(sliderChanged:) forControlEvents:UIControlEventValueChanged];
    return slider;
}

- (UILabel *)valueLabel {
    UILabel *label = [[UILabel alloc] init];
    label.translatesAutoresizingMaskIntoConstraints = NO;
    label.textColor = UIColor.whiteColor;
    label.font = [UIFont monospacedDigitSystemFontOfSize:14 weight:UIFontWeightSemibold];
    label.textAlignment = NSTextAlignmentRight;
    return label;
}

- (UIView *)sliderRowWithTitle:(NSString *)title slider:(UISlider *)slider valueLabel:(UILabel *)valueLabel {
    UIView *row = [[UIView alloc] init];
    row.translatesAutoresizingMaskIntoConstraints = NO;

    UILabel *titleLabel = [[UILabel alloc] init];
    titleLabel.translatesAutoresizingMaskIntoConstraints = NO;
    titleLabel.text = title;
    titleLabel.textColor = UIColor.whiteColor;
    titleLabel.font = [UIFont systemFontOfSize:15 weight:UIFontWeightBold];
    [row addSubview:titleLabel];
    [row addSubview:slider];
    [row addSubview:valueLabel];

    [NSLayoutConstraint activateConstraints:@[
        [titleLabel.leadingAnchor constraintEqualToAnchor:row.leadingAnchor],
        [titleLabel.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
        [titleLabel.widthAnchor constraintEqualToConstant:22],
        [valueLabel.trailingAnchor constraintEqualToAnchor:row.trailingAnchor],
        [valueLabel.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
        [valueLabel.widthAnchor constraintEqualToConstant:42],
        [slider.leadingAnchor constraintEqualToAnchor:titleLabel.trailingAnchor constant:8],
        [slider.trailingAnchor constraintEqualToAnchor:valueLabel.leadingAnchor constant:-10],
        [slider.centerYAnchor constraintEqualToAnchor:row.centerYAnchor],
    ]];
    return row;
}

- (void)modeChanged:(UISegmentedControl *)sender {
    [UIView animateWithDuration:0.18 animations:^{
        [self updateVisibleMode];
        [self.view layoutIfNeeded];
    }];
}

- (void)updateVisibleMode {
    NSInteger mode = self.modeControl.selectedSegmentIndex;
    CGFloat gridHeight = self.presetHeightConstraint.constant + 30;
    self.gridContainer.hidden = mode != 0;
    self.slidersContainer.hidden = mode != 1;
    self.gridContainer.userInteractionEnabled = mode == 0;
    self.slidersContainer.userInteractionEnabled = mode == 1;
    if (mode == 0) {
        self.modeHeightConstraint.constant = gridHeight;
    } else {
        self.modeHeightConstraint.constant = 176;
    }
}

- (void)sliderChanged:(UISlider *)slider {
    self.currentColor = [UIColor colorWithRed:self.redSlider.value / 255.0
                                        green:self.greenSlider.value / 255.0
                                         blue:self.blueSlider.value / 255.0
                                        alpha:1.0];
    [self updatePreview];
}

- (void)syncSlidersFromCurrentColor {
    CGFloat r, g, b;
    [self.currentColor getRed:&r green:&g blue:&b alpha:NULL];
    self.redSlider.value = r * 255.0;
    self.greenSlider.value = g * 255.0;
    self.blueSlider.value = b * 255.0;
    self.redValueLabel.text = [NSString stringWithFormat:@"%03d", (int)round(self.redSlider.value)];
    self.greenValueLabel.text = [NSString stringWithFormat:@"%03d", (int)round(self.greenSlider.value)];
    self.blueValueLabel.text = [NSString stringWithFormat:@"%03d", (int)round(self.blueSlider.value)];
}

- (void)updatePreview {
    self.previewView.backgroundColor = self.currentColor;

    CGFloat r, g, b;
    [self.currentColor getRed:&r green:&g blue:&b alpha:NULL];
    CGFloat luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    CGFloat whiteContrast = luminance > 0.62 ? 0.0 : 1.0;
    self.previewLabel.textColor = [UIColor colorWithWhite:whiteContrast alpha:1.0];
    self.hexLabel.text = [NSString stringWithFormat:@"#%02X%02X%02X", (int)(r * 255), (int)(g * 255), (int)(b * 255)];
    self.redValueLabel.text = [NSString stringWithFormat:@"%03d", (int)round(r * 255)];
    self.greenValueLabel.text = [NSString stringWithFormat:@"%03d", (int)round(g * 255)];
    self.blueValueLabel.text = [NSString stringWithFormat:@"%03d", (int)round(b * 255)];
}

- (void)doneTapped {
    if (self.delegate) {
        [self.delegate colorPickerDidSelectColor:self.currentColor];
    }
    [self dismissViewControllerAnimated:YES completion:nil];
}

- (NSArray<UIColor *> *)presetColors {
    return @[
        [UIColor colorWithRed:0.00 green:0.90 blue:0.10 alpha:1.0],  // green
        [UIColor colorWithRed:0.00 green:1.00 blue:1.00 alpha:1.0],  // cyan
        [UIColor colorWithRed:0.02 green:0.70 blue:0.95 alpha:1.0],  // blue
        [UIColor colorWithRed:0.55 green:0.72 blue:1.00 alpha:1.0],  // light blue
        [UIColor colorWithRed:0.40 green:0.40 blue:1.00 alpha:1.0],  // indigo
        [UIColor colorWithRed:0.92 green:0.40 blue:0.95 alpha:1.0],  // purple
        [UIColor colorWithRed:1.00 green:0.20 blue:0.40 alpha:1.0],  // pink
        [UIColor colorWithRed:0.90 green:0.10 blue:0.10 alpha:1.0],  // red
        [UIColor colorWithRed:1.00 green:0.65 blue:0.10 alpha:1.0],  // orange
        [UIColor colorWithRed:1.00 green:0.86 blue:0.18 alpha:1.0],  // yellow
        [UIColor colorWithRed:0.60 green:0.80 blue:0.00 alpha:1.0],  // lime
        [UIColor colorWithRed:0.50 green:0.50 blue:0.50 alpha:1.0],  // gray
    ];
}

@end
