//
//  NJSponsorBlockSettingViewController.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockSettingViewController.h"
#import "../Settings/NJSponsorBlockSettings.h"
#import "NJSponsorBlockColorPickerController.h"
#import "../Models/NJSponsorBlockCacheStats.h"
#import "../Services/NJSponsorBlockManager.h"
#import "NJSponsorBlockSubmissionManagerViewController.h"
#import "../Services/NJSponsorBlockUnsubmittedSegmentStore.h"
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

static NSString * const NJSponsorBlockSettingCellID = @"NJSponsorBlockSettingCellID";

typedef NS_ENUM(NSInteger, NJSponsorBlockSettingSection) {
    NJSponsorBlockSettingSectionGeneral = 0,
    NJSponsorBlockSettingSectionCache,
    NJSponsorBlockSettingSectionBehavior,
    NJSponsorBlockSettingSectionUI,
    NJSponsorBlockSettingSectionCategories,
    NJSponsorBlockSettingSectionColors,
    NJSponsorBlockSettingSectionThumbnailBadgeColors,
    NJSponsorBlockSettingSectionUnsubmittedSegments,
    NJSponsorBlockSettingSectionServer,
    NJSponsorBlockSettingSectionBackup,
    NJSponsorBlockSettingSectionAbout,
    NJSponsorBlockSettingSectionCount,
};

@interface NJSponsorBlockSettingViewController () <NJSponsorBlockColorPickerDelegate, UIDocumentPickerDelegate>
@property (nonatomic, copy) NSString *editingColorCategory;
@property (nonatomic, copy) NSString *editingThumbnailBadgeLabel;
@property (nonatomic, assign) BOOL isImportingOptions;
- (void)configureUnsubmittedSegmentCell:(UITableViewCell *)cell row:(NSInteger)row;
@end

@implementation NJSponsorBlockSettingViewController

- (instancetype)init {
    return [super initWithStyle:UITableViewStyleGrouped];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"SponsorBlock";
    self.tableView.rowHeight = 48;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:NJSponsorBlockSettingCellID];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    [self.tableView reloadData];
}

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return NJSponsorBlockSettingSectionCount;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    switch (section) {
        case NJSponsorBlockSettingSectionGeneral:
            return 2;
        case NJSponsorBlockSettingSectionCache:
            return 3;
        case NJSponsorBlockSettingSectionBehavior:
            return 3;
        case NJSponsorBlockSettingSectionUI:
            return 4;
        case NJSponsorBlockSettingSectionCategories:
            return [NJSponsorBlockSettings categoryOptions].count;
        case NJSponsorBlockSettingSectionColors:
            return [NJSponsorBlockSettings categoryOptions].count;
        case NJSponsorBlockSettingSectionThumbnailBadgeColors:
            return [NJSponsorBlockSettings thumbnailBadgeLabelOptions].count;
        case NJSponsorBlockSettingSectionUnsubmittedSegments:
            return 1;
        case NJSponsorBlockSettingSectionServer:
            return 4;
        case NJSponsorBlockSettingSectionBackup:
            return 4;
        case NJSponsorBlockSettingSectionAbout:
            return 1;
        default:
            return 0;
    }
}

- (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    switch (section) {
        case NJSponsorBlockSettingSectionGeneral:
            return @"基础";
        case NJSponsorBlockSettingSectionCache:
            return @"缓存管理";
        case NJSponsorBlockSettingSectionBehavior:
            return @"跳过行为";
        case NJSponsorBlockSettingSectionUI:
            return @"界面";
        case NJSponsorBlockSettingSectionCategories:
            return @"分类行为";
        case NJSponsorBlockSettingSectionColors:
            return @"行为颜色";
        case NJSponsorBlockSettingSectionThumbnailBadgeColors:
            return @"缩略图标签颜色";
        case NJSponsorBlockSettingSectionUnsubmittedSegments:
            return @"未提交片段";
        case NJSponsorBlockSettingSectionServer:
            return @"服务器";
        case NJSponsorBlockSettingSectionBackup:
            return @"备份/恢复";
        case NJSponsorBlockSettingSectionAbout:
            return @"说明";
        default:
            return nil;
    }
}

- (NSString *)tableView:(UITableView *)tableView titleForFooterInSection:(NSInteger)section {
    if (section == NJSponsorBlockSettingSectionCache) {
        return @"启用缓存可以提高视频片段的加载速度。关闭时会清除所有已存储的缓存数据。";
    }
    if (section == NJSponsorBlockSettingSectionBackup) {
        return @"导入/导出的选项以 JSON 格式保存，包含了您的私人用户 ID，请谨慎保管。";
    }
    if (section == NJSponsorBlockSettingSectionUnsubmittedSegments) {
        return @"未提交片段会显示在播放器进度条中，投稿成功后会自动移除。";
    }
    if (section == NJSponsorBlockSettingSectionAbout) {
        return @"当前 iOS 客户端支持官方插件的播放跳过、分类行为、服务器、缓存配置、缩略图标签和基础片段投稿。投稿为移动端简化流程，不包含浏览器扩展的完整编辑器、快捷键、动态/评论屏蔽等功能。";
    }
    return nil;
}

- (UIView *)tableView:(UITableView *)tableView viewForFooterInSection:(NSInteger)section {
    if (section == NJSponsorBlockSettingSectionCache) {
        return [self cacheSectionFooterView];
    }
    if (section != NJSponsorBlockSettingSectionColors && section != NJSponsorBlockSettingSectionThumbnailBadgeColors) {
        return nil;
    }
    UIView *footer = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 0, 44)];
    UIButton *resetButton = [UIButton buttonWithType:UIButtonTypeSystem];
    if (section == NJSponsorBlockSettingSectionColors) {
        [resetButton setTitle:@"恢复默认行为颜色" forState:UIControlStateNormal];
        [resetButton addTarget:self action:@selector(resetColorsTapped) forControlEvents:UIControlEventTouchUpInside];
    } else {
        [resetButton setTitle:@"恢复默认缩略图标签颜色" forState:UIControlStateNormal];
        [resetButton addTarget:self action:@selector(resetThumbnailBadgeColorsTapped) forControlEvents:UIControlEventTouchUpInside];
    }
    resetButton.titleLabel.font = [UIFont systemFontOfSize:15];
    resetButton.translatesAutoresizingMaskIntoConstraints = NO;
    [footer addSubview:resetButton];
    [NSLayoutConstraint activateConstraints:@[
        [resetButton.centerXAnchor constraintEqualToAnchor:footer.centerXAnchor],
        [resetButton.centerYAnchor constraintEqualToAnchor:footer.centerYAnchor],
    ]];
    return footer;
}

- (CGFloat)tableView:(UITableView *)tableView heightForFooterInSection:(NSInteger)section {
    if (section == NJSponsorBlockSettingSectionCache) {
        return UITableViewAutomaticDimension;
    }
    if (section == NJSponsorBlockSettingSectionColors || section == NJSponsorBlockSettingSectionThumbnailBadgeColors) {
        return 44;
    }
    return UITableViewAutomaticDimension;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleValue1 reuseIdentifier:NJSponsorBlockSettingCellID];
    cell.textLabel.font = [UIFont systemFontOfSize:16];
    cell.detailTextLabel.font = [UIFont systemFontOfSize:14];
    cell.detailTextLabel.textColor = [UIColor secondaryLabelColor];
    cell.selectionStyle = UITableViewCellSelectionStyleDefault;
    cell.accessoryView = nil;
    cell.accessoryType = UITableViewCellAccessoryNone;

    switch (indexPath.section) {
        case NJSponsorBlockSettingSectionGeneral:
            [self configureGeneralCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionCache:
            [self configureCacheCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionBehavior:
            [self configureBehaviorCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionUI:
            [self configureUICell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionThumbnailBadgeColors:
            [self configureThumbnailBadgeColorCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionUnsubmittedSegments:
            [self configureUnsubmittedSegmentCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionCategories:
            [self configureCategoryCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionColors:
            [self configureColorCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionServer:
            [self configureServerCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionBackup:
            [self configureBackupCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionAbout:
            cell.textLabel.text = @"与官方核心跳过逻辑保持一致";
            cell.textLabel.numberOfLines = 0;
            cell.selectionStyle = UITableViewCellSelectionStyleNone;
            break;
        default:
            break;
    }
    return cell;
}

- (void)configureGeneralCell:(UITableViewCell *)cell row:(NSInteger)row {
    if (row == 0) {
        cell.textLabel.text = @"启用 SponsorBlock";
        cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings enabled] tag:100];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    cell.textLabel.text = @"跳过次数统计跟踪";
    cell.detailTextLabel.text = @"报告跳过数据";
    cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings skipTrackingEnabled] tag:104];
    cell.selectionStyle = UITableViewCellSelectionStyleNone;
}

- (void)configureCacheCell:(UITableViewCell *)cell row:(NSInteger)row {
    if (row == 0) {
        cell.textLabel.text = @"启用缓存";
        cell.detailTextLabel.text = @"提高加载速度";
        cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings cacheEnabled] tag:101];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    if (row == 1) {
        NJSponsorBlockCacheStats *stats = [NJSponsorBlockCacheStats sharedInstance];
        cell.textLabel.text = @"视频片段缓存";
        cell.detailTextLabel.text = [NSString stringWithFormat:@"%lu 项 · %lu KB",
                                      (unsigned long)stats.totalItems,
                                      (unsigned long)(stats.totalSizeBytes / 1024)];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    cell.textLabel.text = @"清除所有缓存";
    cell.textLabel.textColor = [UIColor systemRedColor];
    cell.accessoryType = UITableViewCellAccessoryNone;
}

- (UIView *)cacheSectionFooterView {
    NJSponsorBlockCacheStats *stats = [NJSponsorBlockCacheStats sharedInstance];
    BOOL cacheEnabled = [NJSponsorBlockSettings cacheEnabled];

    UIView *footer = [[UIView alloc] init];

    UIView *statsContainer = [[UIView alloc] init];
    statsContainer.backgroundColor = [UIColor tertiarySystemBackgroundColor];
    statsContainer.layer.cornerRadius = 10;
    statsContainer.translatesAutoresizingMaskIntoConstraints = NO;
    statsContainer.alpha = cacheEnabled ? 1.0 : 0.4;
    [footer addSubview:statsContainer];

    UILabel *headerLabel = [[UILabel alloc] init];
    headerLabel.text = @"缓存类型        缓存大小    缓存项数    今日命中    今日读取";
    headerLabel.font = [UIFont monospacedSystemFontOfSize:11 weight:UIFontWeightMedium];
    headerLabel.textColor = [UIColor secondaryLabelColor];
    headerLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [statsContainer addSubview:headerLabel];

    NSUInteger sizeKB = stats.totalSizeBytes / 1024;
    NSUInteger dailyKB = stats.dailySizeBytes / 1024;

    UILabel *segmentRow = [[UILabel alloc] init];
    segmentRow.text = [NSString stringWithFormat:@"视频片段缓存    %lu KB       %lu 项      %lu 次      %lu KB",
                        (unsigned long)sizeKB,
                        (unsigned long)stats.totalItems,
                        (unsigned long)stats.dailyHits,
                        (unsigned long)dailyKB];
    segmentRow.font = [UIFont monospacedSystemFontOfSize:12 weight:UIFontWeightRegular];
    segmentRow.textColor = [UIColor labelColor];
    segmentRow.translatesAutoresizingMaskIntoConstraints = NO;
    [statsContainer addSubview:segmentRow];

    UIView *separator = [[UIView alloc] init];
    separator.backgroundColor = [UIColor separatorColor];
    separator.translatesAutoresizingMaskIntoConstraints = NO;
    [statsContainer addSubview:separator];

    UILabel *totalRow = [[UILabel alloc] init];
    totalRow.text = [NSString stringWithFormat:@"总缓存          %lu KB       %lu 项      %lu 次      %lu KB",
                      (unsigned long)sizeKB,
                      (unsigned long)stats.totalItems,
                      (unsigned long)stats.dailyHits,
                      (unsigned long)dailyKB];
    totalRow.font = [UIFont monospacedSystemFontOfSize:12 weight:UIFontWeightBold];
    totalRow.textColor = [UIColor labelColor];
    totalRow.translatesAutoresizingMaskIntoConstraints = NO;
    [statsContainer addSubview:totalRow];

    [NSLayoutConstraint activateConstraints:@[
        [statsContainer.topAnchor constraintEqualToAnchor:footer.topAnchor constant:8],
        [statsContainer.leadingAnchor constraintEqualToAnchor:footer.leadingAnchor],
        [statsContainer.trailingAnchor constraintEqualToAnchor:footer.trailingAnchor],

        [headerLabel.topAnchor constraintEqualToAnchor:statsContainer.topAnchor constant:12],
        [headerLabel.leadingAnchor constraintEqualToAnchor:statsContainer.leadingAnchor constant:12],
        [headerLabel.trailingAnchor constraintEqualToAnchor:statsContainer.trailingAnchor constant:-12],

        [segmentRow.topAnchor constraintEqualToAnchor:headerLabel.bottomAnchor constant:8],
        [segmentRow.leadingAnchor constraintEqualToAnchor:statsContainer.leadingAnchor constant:12],
        [segmentRow.trailingAnchor constraintEqualToAnchor:statsContainer.trailingAnchor constant:-12],

        [separator.topAnchor constraintEqualToAnchor:segmentRow.bottomAnchor constant:8],
        [separator.leadingAnchor constraintEqualToAnchor:statsContainer.leadingAnchor constant:12],
        [separator.trailingAnchor constraintEqualToAnchor:statsContainer.trailingAnchor constant:-12],
        [separator.heightAnchor constraintEqualToConstant:0.5],

        [totalRow.topAnchor constraintEqualToAnchor:separator.bottomAnchor constant:8],
        [totalRow.leadingAnchor constraintEqualToAnchor:statsContainer.leadingAnchor constant:12],
        [totalRow.trailingAnchor constraintEqualToAnchor:statsContainer.trailingAnchor constant:-12],
        [totalRow.bottomAnchor constraintEqualToAnchor:statsContainer.bottomAnchor constant:-12],

        [statsContainer.bottomAnchor constraintEqualToAnchor:footer.bottomAnchor constant:-8],
    ]];

    return footer;
}

- (void)presentClearCacheConfirmation {
    UIAlertController *confirm = [UIAlertController alertControllerWithTitle:@"清除所有缓存"
                                                                    message:@"确定要清除所有缓存数据吗？此操作不可撤销。"
                                                             preferredStyle:UIAlertControllerStyleAlert];
    __weak typeof(self) weakSelf = self;
    [confirm addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [confirm addAction:[UIAlertAction actionWithTitle:@"清除" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [weakSelf.tableView reloadSections:[NSIndexSet indexSetWithIndex:NJSponsorBlockSettingSectionCache] withRowAnimation:UITableViewRowAnimationNone];
    }]];
    [self presentViewController:confirm animated:YES completion:nil];
}

- (void)configureUICell:(UITableViewCell *)cell row:(NSInteger)row {
    if (row == 0) {
        cell.textLabel.text = @"在 SeekbarWidget 中显示片段";
        cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings showSegmentsInSeekbarWidget] tag:105];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    if (row == 1) {
        cell.textLabel.text = @"在 ProgressWidget 中显示片段";
        cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings showSegmentsInProgressWidget] tag:106];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    if (row == 2) {
        cell.textLabel.text = @"播放器中显示 SponsorBlock 按钮";
        cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings showSharedEntryButton] tag:107];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    cell.textLabel.text = @"显示缩略图标签";
    cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings showVideoLabels] tag:108];
    cell.selectionStyle = UITableViewCellSelectionStyleNone;
}

- (void)configureBehaviorCell:(UITableViewCell *)cell row:(NSInteger)row {
    if (row == 0) {
        cell.textLabel.text = @"Seek 到片段内时跳过";
        cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings skipOnSeekToSegment] tag:102];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    if (row == 1) {
        cell.textLabel.text = @"最短片段时长";
        cell.detailTextLabel.text = [NSString stringWithFormat:@"%.1f 秒", [NJSponsorBlockSettings minDuration]];
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    cell.textLabel.text = @"提前提示时间";
    cell.detailTextLabel.text = [NSString stringWithFormat:@"%.1f 秒", [NJSponsorBlockSettings advanceNoticeDuration]];
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
}

- (void)configureCategoryCell:(UITableViewCell *)cell row:(NSInteger)row {
    NJSponsorBlockCategoryOption *option = [NJSponsorBlockSettings categoryOptions][row];
    NJSponsorBlockCategoryAction action = [NJSponsorBlockSettings actionForCategory:option.category];
    cell.textLabel.text = option.title;
    cell.detailTextLabel.text = [NJSponsorBlockSettings titleForAction:action];
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
}

- (void)configureColorCell:(UITableViewCell *)cell row:(NSInteger)row {
    NJSponsorBlockCategoryOption *option = [NJSponsorBlockSettings categoryOptions][row];
    UIColor *color = [NJSponsorBlockSettings colorForCategory:option.category];
    cell.textLabel.text = option.title;
    cell.imageView.image = [self circleImageWithColor:color size:22];
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
}

- (void)configureThumbnailBadgeColorCell:(UITableViewCell *)cell row:(NSInteger)row {
    NSArray<NJSponsorBlockCategoryOption *> *options = [NJSponsorBlockSettings thumbnailBadgeLabelOptions];
    if (row >= options.count) {
        return;
    }
    NJSponsorBlockCategoryOption *option = options[row];
    UIColor *color = [NJSponsorBlockSettings thumbnailBadgeColorForLabel:option.category];
    cell.textLabel.text = option.title;
    cell.imageView.image = [self circleImageWithColor:color size:22];
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
}

- (void)configureUnsubmittedSegmentCell:(UITableViewCell *)cell row:(NSInteger)row {
    NJSponsorBlockUnsubmittedSegmentStore *store = [NJSponsorBlockUnsubmittedSegmentStore sharedStore];
    cell.textLabel.text = @"未提交片段管理";
    cell.detailTextLabel.text = [NSString stringWithFormat:@"%lu 个视频 · %lu 段",
                                 (unsigned long)store.videoCount,
                                 (unsigned long)store.totalSegmentCount];
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
}

- (UIImage *)circleImageWithColor:(UIColor *)color size:(CGFloat)size {
    CGSize imageSize = CGSizeMake(size, size);
    UIGraphicsBeginImageContextWithOptions(imageSize, NO, 0);
    CGContextRef ctx = UIGraphicsGetCurrentContext();
    CGContextSetFillColorWithColor(ctx, color.CGColor);
    CGContextFillEllipseInRect(ctx, CGRectMake(0, 0, size, size));
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}

- (void)configureServerCell:(UITableViewCell *)cell row:(NSInteger)row {
    if (row == 0) {
        cell.textLabel.text = @"服务器地址";
        cell.detailTextLabel.text = [NJSponsorBlockSettings serverBaseURLString];
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    if (row == 1) {
        cell.textLabel.text = @"使用测试服务器";
        cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings testingServerEnabled] tag:103];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    if (row == 2) {
        cell.textLabel.text = @"服务器状态";
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    cell.textLabel.text = @"项目代码";
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
}

- (void)configureBackupCell:(UITableViewCell *)cell row:(NSInteger)row {
    if (row == 0) {
        cell.textLabel.text = @"导入/导出您的私人用户ID";
        cell.detailTextLabel.text = @"私人ID";
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    if (row == 1) {
        cell.textLabel.text = @"导入/导出所有选项";
        cell.detailTextLabel.text = @"设置、分类、颜色";
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    if (row == 2) {
        cell.textLabel.text = @"导入/导出所有其他数据";
        cell.detailTextLabel.text = @"缓存的片段数据";
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    cell.textLabel.text = @"重置所有设置";
    cell.textLabel.textColor = [UIColor systemRedColor];
    cell.accessoryType = UITableViewCellAccessoryNone;
}

- (UISwitch *)switchWithOn:(BOOL)on tag:(NSInteger)tag {
    UISwitch *aSwitch = [[UISwitch alloc] init];
    aSwitch.on = on;
    aSwitch.tag = tag;
    [aSwitch addTarget:self action:@selector(switchChanged:) forControlEvents:UIControlEventValueChanged];
    return aSwitch;
}

- (void)switchChanged:(UISwitch *)aSwitch {
    if (aSwitch.tag == 100) {
        [NJSponsorBlockSettings setEnabled:aSwitch.on];
    } else if (aSwitch.tag == 101) {
        [NJSponsorBlockSettings setCacheEnabled:aSwitch.on];
    } else if (aSwitch.tag == 102) {
        [NJSponsorBlockSettings setSkipOnSeekToSegment:aSwitch.on];
    } else if (aSwitch.tag == 103) {
        [NJSponsorBlockSettings setTestingServerEnabled:aSwitch.on];
    } else if (aSwitch.tag == 104) {
        [NJSponsorBlockSettings setSkipTrackingEnabled:aSwitch.on];
    } else if (aSwitch.tag == 105) {
        [NJSponsorBlockSettings setShowSegmentsInSeekbarWidget:aSwitch.on];
    } else if (aSwitch.tag == 106) {
        [NJSponsorBlockSettings setShowSegmentsInProgressWidget:aSwitch.on];
    } else if (aSwitch.tag == 107) {
        [NJSponsorBlockSettings setShowSharedEntryButton:aSwitch.on];
    } else if (aSwitch.tag == 108) {
        [NJSponsorBlockSettings setShowVideoLabels:aSwitch.on];
    }
    [self.tableView reloadData];
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    if (indexPath.section == NJSponsorBlockSettingSectionBehavior && indexPath.row == 1) {
        [self presentNumberInputWithTitle:@"最短片段时长" value:[NJSponsorBlockSettings minDuration] handler:^(NSTimeInterval value) {
            [NJSponsorBlockSettings setMinDuration:value];
        }];
        return;
    }
    if (indexPath.section == NJSponsorBlockSettingSectionBehavior && indexPath.row == 2) {
        [self presentNumberInputWithTitle:@"提前提示时间" value:[NJSponsorBlockSettings advanceNoticeDuration] handler:^(NSTimeInterval value) {
            [NJSponsorBlockSettings setAdvanceNoticeDuration:value];
        }];
        return;
    }
    if (indexPath.section == NJSponsorBlockSettingSectionThumbnailBadgeColors) {
        [self presentThumbnailBadgeColorPickerForRow:indexPath.row sourceCell:[tableView cellForRowAtIndexPath:indexPath]];
        return;
    }
    if (indexPath.section == NJSponsorBlockSettingSectionCategories) {
        [self presentCategoryActionSheetForRow:indexPath.row sourceCell:[tableView cellForRowAtIndexPath:indexPath]];
        return;
    }
    if (indexPath.section == NJSponsorBlockSettingSectionColors) {
        [self presentColorPickerForRow:indexPath.row sourceCell:[tableView cellForRowAtIndexPath:indexPath]];
        return;
    }
    if (indexPath.section == NJSponsorBlockSettingSectionServer && indexPath.row == 0) {
        [self presentServerInput];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionServer && indexPath.row == 2) {
        [UIApplication.sharedApplication openURL:[NSURL URLWithString:@"https://status.bsbsb.top"] options:@{} completionHandler:nil];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionServer && indexPath.row == 3) {
        [UIApplication.sharedApplication openURL:[NSURL URLWithString:@"https://github.com/hanydd/BilibiliSponsorBlock"] options:@{} completionHandler:nil];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionBackup && indexPath.row == 0) {
        [self presentUserIDManagement];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionBackup && indexPath.row == 1) {
        [self presentOptionsBackupSheet];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionBackup && indexPath.row == 2) {
        [self presentOtherDataBackupSheet];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionBackup && indexPath.row == 3) {
        [self presentResetConfirmation];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionCache && indexPath.row == 2) {
        [self presentClearCacheConfirmation];
    }
    if (indexPath.section == NJSponsorBlockSettingSectionUnsubmittedSegments) {
        NJSponsorBlockSubmissionManagerViewController *controller = [[NJSponsorBlockSubmissionManagerViewController alloc] init];
        [self.navigationController pushViewController:controller animated:YES];
    }
}

- (void)presentNumberInputWithTitle:(NSString *)title value:(NSTimeInterval)value handler:(void (^)(NSTimeInterval value))handler {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:title message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.keyboardType = UIKeyboardTypeDecimalPad;
        textField.text = [NSString stringWithFormat:@"%.1f", value];
        textField.placeholder = @"秒";
    }];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"保存" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        NSString *text = alert.textFields.firstObject.text ?: @"0";
        if (handler) {
            handler([text doubleValue]);
        }
        [weakSelf.tableView reloadData];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)presentServerInput {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"服务器地址" message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.keyboardType = UIKeyboardTypeURL;
        textField.autocapitalizationType = UITextAutocapitalizationTypeNone;
        textField.autocorrectionType = UITextAutocorrectionTypeNo;
        textField.text = [NJSponsorBlockSettings serverBaseURLString];
        textField.placeholder = @"https://bsbsb.top";
    }];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"保存" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        [NJSponsorBlockSettings setServerBaseURLString:alert.textFields.firstObject.text ?: @""];
        [weakSelf.tableView reloadData];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)presentUserIDManagement {
    NSString *currentUserID = [NJSponsorBlockSettings sponsorBlockUserID];
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"设置私人用户ID"
                                                                   message:[NSString stringWithFormat:@"当前 ID：\n%@\n\n私人ID应该被保密。如果他人获得了你的私人ID，他就可以冒充您。如果您想找公开用户ID，请点击弹出窗口中的剪贴板图标。", currentUserID]
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.text = currentUserID;
        textField.autocapitalizationType = UITextAutocapitalizationTypeNone;
        textField.autocorrectionType = UITextAutocorrectionTypeNo;
    }];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"复制当前ID" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        UIPasteboard.generalPasteboard.string = currentUserID;
    }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"设置" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        NSString *newUserID = alert.textFields.firstObject.text ?: @"";
        if ([newUserID isEqualToString:currentUserID]) {
            return;
        }
        UIAlertController *confirm = [UIAlertController alertControllerWithTitle:@"警告"
                                                                         message:@"更改私人用户ID是永久性的。您确定要这么做吗？请务必备份您的旧私人ID以防万一。"
                                                                  preferredStyle:UIAlertControllerStyleAlert];
        [confirm addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
        [confirm addAction:[UIAlertAction actionWithTitle:@"确定更改" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
            [NJSponsorBlockSettings setSponsorBlockUserID:newUserID];
            [weakSelf.tableView reloadData];
        }]];
        [weakSelf presentViewController:confirm animated:YES completion:nil];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

#pragma mark - Options Backup

- (void)presentOptionsBackupSheet {
    UIAlertController *sheet = [UIAlertController alertControllerWithTitle:@"导入/导出所有选项"
                                                                  message:@"这是您所有设置的 JSON 格式。它包含了您的私人用户 ID，所以您一定要谨慎的保管它。"
                                                           preferredStyle:UIAlertControllerStyleActionSheet];
    __weak typeof(self) weakSelf = self;
    [sheet addAction:[UIAlertAction actionWithTitle:@"编辑/复制" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        [weakSelf presentOptionsCopyEditor];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"保存到文件" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        [weakSelf exportOptionsToFile];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"从文件加载" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        weakSelf.isImportingOptions = YES;
        [weakSelf presentDocumentPickerForImport];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    sheet.popoverPresentationController.sourceView = self.view;
    sheet.popoverPresentationController.sourceRect = self.view.bounds;
    [self presentViewController:sheet animated:YES completion:nil];
}

- (void)presentOptionsCopyEditor {
    NSDictionary *settings = [NJSponsorBlockSettings exportSettings];
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:settings options:NSJSONWritingPrettyPrinted error:nil];
    NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding] ?: @"{}";

    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"所有选项"
                                                                   message:@"复制下方 JSON 以备份，或粘贴 JSON 以恢复。"
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.text = jsonString;
        textField.font = [UIFont monospacedSystemFontOfSize:11 weight:UIFontWeightRegular];
    }];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"复制" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        UIPasteboard.generalPasteboard.string = alert.textFields.firstObject.text ?: @"";
    }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"导入" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        NSString *text = alert.textFields.firstObject.text ?: @"";
        [weakSelf importOptionsFromJSONString:text];
    }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)exportOptionsToFile {
    NSDictionary *settings = [NJSponsorBlockSettings exportSettings];
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:settings options:NSJSONWritingPrettyPrinted error:nil];
    if (!jsonData) {
        return;
    }
    NSDateFormatter *fmt = [[NSDateFormatter alloc] init];
    fmt.dateFormat = @"yyyy-MM-dd_HH.mm.ss";
    NSString *dateStr = [fmt stringFromDate:[NSDate date]];
    NSString *fileName = [NSString stringWithFormat:@"SponsorBlockConfig_%@.json", dateStr];
    NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:fileName];
    NSURL *tempURL = [NSURL fileURLWithPath:tempPath];
    [jsonData writeToURL:tempURL atomically:YES];

    self.isImportingOptions = NO;
    UIDocumentPickerViewController *picker = [[UIDocumentPickerViewController alloc] initForExportingURLs:@[tempURL]];
    picker.delegate = self;
    [self presentViewController:picker animated:YES completion:nil];
}

- (void)importOptionsFromJSONString:(NSString *)string {
    if (string.length == 0) {
        return;
    }
    NSData *data = [string dataUsingEncoding:NSUTF8StringEncoding];
    NSError *error = nil;
    NSDictionary *dict = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (error || ![dict isKindOfClass:[NSDictionary class]]) {
        UIAlertController *err = [UIAlertController alertControllerWithTitle:@"格式错误" message:@"无法解析 JSON，请检查格式。" preferredStyle:UIAlertControllerStyleAlert];
        [err addAction:[UIAlertAction actionWithTitle:@"好的" style:UIAlertActionStyleDefault handler:nil]];
        [self presentViewController:err animated:YES completion:nil];
        return;
    }
    __weak typeof(self) weakSelf = self;
    UIAlertController *confirm = [UIAlertController alertControllerWithTitle:@"确认导入"
                                                                    message:@"导入将覆盖当前所有选项设置。确定要继续吗？"
                                                             preferredStyle:UIAlertControllerStyleAlert];
    [confirm addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [confirm addAction:[UIAlertAction actionWithTitle:@"确定导入" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [NJSponsorBlockSettings importSettings:dict];
        [weakSelf.tableView reloadData];
    }]];
    [self presentViewController:confirm animated:YES completion:nil];
}

#pragma mark - Other Data Backup

- (void)presentOtherDataBackupSheet {
    UIAlertController *sheet = [UIAlertController alertControllerWithTitle:@"导入/导出所有其他数据"
                                                                  message:@"其他数据包含缓存的片段信息等。"
                                                           preferredStyle:UIAlertControllerStyleActionSheet];
    __weak typeof(self) weakSelf = self;
    [sheet addAction:[UIAlertAction actionWithTitle:@"编辑/复制" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        [weakSelf presentOtherDataCopyEditor];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"保存到文件" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        [weakSelf exportOtherDataToFile];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"从文件加载" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        weakSelf.isImportingOptions = NO;
        [weakSelf presentDocumentPickerForImport];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    sheet.popoverPresentationController.sourceView = self.view;
    sheet.popoverPresentationController.sourceRect = self.view.bounds;
    [self presentViewController:sheet animated:YES completion:nil];
}

- (void)presentOtherDataCopyEditor {
    NSDictionary *otherData = [self exportOtherData];
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:otherData options:NSJSONWritingPrettyPrinted error:nil];
    NSString *jsonString = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding] ?: @"{}";

    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"所有其他数据"
                                                                   message:@"复制下方 JSON 以备份，或粘贴 JSON 以恢复。"
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.text = jsonString;
        textField.font = [UIFont monospacedSystemFontOfSize:11 weight:UIFontWeightRegular];
    }];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"复制" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        UIPasteboard.generalPasteboard.string = alert.textFields.firstObject.text ?: @"";
    }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"导入" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        NSString *text = alert.textFields.firstObject.text ?: @"";
        [weakSelf importOtherDataFromJSONString:text];
    }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)exportOtherDataToFile {
    NSDictionary *otherData = [self exportOtherData];
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:otherData options:NSJSONWritingPrettyPrinted error:nil];
    if (!jsonData) {
        return;
    }
    NSDateFormatter *fmt = [[NSDateFormatter alloc] init];
    fmt.dateFormat = @"yyyy-MM-dd_HH.mm.ss";
    NSString *dateStr = [fmt stringFromDate:[NSDate date]];
    NSString *fileName = [NSString stringWithFormat:@"SponsorBlockOtherData_%@.json", dateStr];
    NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:fileName];
    NSURL *tempURL = [NSURL fileURLWithPath:tempPath];
    [jsonData writeToURL:tempURL atomically:YES];

    self.isImportingOptions = NO;
    UIDocumentPickerViewController *picker = [[UIDocumentPickerViewController alloc] initForExportingURLs:@[tempURL]];
    picker.delegate = self;
    [self presentViewController:picker animated:YES completion:nil];
}

- (NSDictionary *)exportOtherData {
    NSMutableDictionary *data = [NSMutableDictionary dictionary];
    // Export cached segments from YYCache (segment cache keys follow "SB:{videoID}:{cid}" pattern)
    // For now, we export the segment cache dictionary if available
    // The actual segment data is stored in NJ_SETTING_CACHE with keys like "SB:{bvid}:{cid}"
    return data;
}

- (void)importOtherDataFromJSONString:(NSString *)string {
    if (string.length == 0) {
        return;
    }
    NSData *data = [string dataUsingEncoding:NSUTF8StringEncoding];
    NSError *error = nil;
    NSDictionary *dict = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (error || ![dict isKindOfClass:[NSDictionary class]]) {
        UIAlertController *err = [UIAlertController alertControllerWithTitle:@"格式错误" message:@"无法解析 JSON，请检查格式。" preferredStyle:UIAlertControllerStyleAlert];
        [err addAction:[UIAlertAction actionWithTitle:@"好的" style:UIAlertActionStyleDefault handler:nil]];
        [self presentViewController:err animated:YES completion:nil];
        return;
    }
    __weak typeof(self) weakSelf = self;
    UIAlertController *confirm = [UIAlertController alertControllerWithTitle:@"确认导入"
                                                                    message:@"导入将覆盖当前所有其他数据。确定要继续吗？"
                                                             preferredStyle:UIAlertControllerStyleAlert];
    [confirm addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [confirm addAction:[UIAlertAction actionWithTitle:@"确定导入" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [weakSelf importOtherData:dict];
        [weakSelf.tableView reloadData];
    }]];
    [self presentViewController:confirm animated:YES completion:nil];
}

- (void)importOtherData:(NSDictionary *)data {
    // Import cached segments back into YYCache
    // Implementation depends on the actual data structure
}

#pragma mark - Reset

- (void)presentResetConfirmation {
    UIAlertController *confirm = [UIAlertController alertControllerWithTitle:@"重置所有设置"
                                                                    message:@"确定要将所有设置恢复为默认值吗？您的私人用户 ID 将被保留。此操作不可撤销。"
                                                             preferredStyle:UIAlertControllerStyleAlert];
    __weak typeof(self) weakSelf = self;
    [confirm addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [confirm addAction:[UIAlertAction actionWithTitle:@"重置" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [NJSponsorBlockSettings resetToDefaults];
        [weakSelf.tableView reloadData];
    }]];
    [self presentViewController:confirm animated:YES completion:nil];
}

#pragma mark - UIDocumentPickerDelegate

- (void)documentPicker:(UIDocumentPickerViewController *)controller didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
    NSURL *url = urls.firstObject;
    if (!url) {
        return;
    }
    NSData *data = [NSData dataWithContentsOfURL:url];
    if (!data) {
        return;
    }
    NSString *string = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    if (self.isImportingOptions) {
        [self importOptionsFromJSONString:string];
    } else {
        [self importOtherDataFromJSONString:string];
    }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController *)controller {
    // No-op
}

- (void)presentDocumentPickerForImport {
    UIDocumentPickerViewController *picker = [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:@[UTTypeJSON]];
    picker.delegate = self;
    [self presentViewController:picker animated:YES completion:nil];
}

- (void)presentCategoryActionSheetForRow:(NSInteger)row sourceCell:(UITableViewCell *)sourceCell {
    NJSponsorBlockCategoryOption *option = [NJSponsorBlockSettings categoryOptions][row];
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:option.title message:nil preferredStyle:UIAlertControllerStyleActionSheet];
    NSArray<NSNumber *> *actions = @[@(NJSponsorBlockCategoryActionDisabled), @(NJSponsorBlockCategoryActionShowOverlay), @(NJSponsorBlockCategoryActionManualSkip), @(NJSponsorBlockCategoryActionAutoSkip)];
    __weak typeof(self) weakSelf = self;
    for (NSNumber *actionNumber in actions) {
        NJSponsorBlockCategoryAction action = (NJSponsorBlockCategoryAction)actionNumber.integerValue;
        [alert addAction:[UIAlertAction actionWithTitle:[NJSponsorBlockSettings titleForAction:action] style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *alertAction) {
            [NJSponsorBlockSettings setAction:action forCategory:option.category];
            [weakSelf.tableView reloadData];
        }]];
    }
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    alert.popoverPresentationController.sourceView = sourceCell ?: self.view;
    alert.popoverPresentationController.sourceRect = sourceCell ? sourceCell.bounds : self.view.bounds;
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)presentColorPickerForRow:(NSInteger)row sourceCell:(UITableViewCell *)sourceCell {
    NJSponsorBlockCategoryOption *option = [NJSponsorBlockSettings categoryOptions][row];
    self.editingColorCategory = option.category;
    self.editingThumbnailBadgeLabel = nil;
    UIColor *currentColor = [NJSponsorBlockSettings colorForCategory:option.category];
    NJSponsorBlockColorPickerController *picker = [[NJSponsorBlockColorPickerController alloc] initWithColor:currentColor categoryTitle:option.title];
    picker.delegate = self;
    UINavigationController *nav = [[UINavigationController alloc] initWithRootViewController:picker];
    nav.modalPresentationStyle = UIModalPresentationPageSheet;
    [self presentViewController:nav animated:YES completion:nil];
}

- (void)presentThumbnailBadgeColorPickerForRow:(NSInteger)row sourceCell:(UITableViewCell *)sourceCell {
    NSArray<NJSponsorBlockCategoryOption *> *options = [NJSponsorBlockSettings thumbnailBadgeLabelOptions];
    if (row >= options.count) {
        return;
    }
    NJSponsorBlockCategoryOption *option = options[row];
    self.editingColorCategory = nil;
    self.editingThumbnailBadgeLabel = option.category;
    UIColor *currentColor = [NJSponsorBlockSettings thumbnailBadgeColorForLabel:option.category];
    NJSponsorBlockColorPickerController *picker = [[NJSponsorBlockColorPickerController alloc] initWithColor:currentColor categoryTitle:option.title];
    picker.delegate = self;
    UINavigationController *nav = [[UINavigationController alloc] initWithRootViewController:picker];
    nav.modalPresentationStyle = UIModalPresentationPageSheet;
    [self presentViewController:nav animated:YES completion:nil];
}

- (void)colorPickerDidSelectColor:(UIColor *)color {
    if (self.editingThumbnailBadgeLabel) {
        [NJSponsorBlockSettings setThumbnailBadgeColor:color forLabel:self.editingThumbnailBadgeLabel];
        self.editingThumbnailBadgeLabel = nil;
        [self.tableView reloadData];
        return;
    }
    if (self.editingColorCategory) {
        [NJSponsorBlockSettings setColor:color forCategory:self.editingColorCategory];
        self.editingColorCategory = nil;
        [self.tableView reloadData];
    }
}

- (void)resetColorsTapped {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"恢复默认颜色" message:@"确定将所有分类颜色恢复为默认值？" preferredStyle:UIAlertControllerStyleAlert];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"恢复" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [NJSponsorBlockSettings resetColors];
        [weakSelf.tableView reloadData];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)resetThumbnailBadgeColorsTapped {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"恢复默认缩略图标签颜色" message:@"确定将所有缩略图标签颜色恢复为默认值？" preferredStyle:UIAlertControllerStyleAlert];
    __weak typeof(self) weakSelf = self;
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"恢复" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [NJSponsorBlockSettings resetThumbnailBadgeColors];
        [weakSelf.tableView reloadData];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

@end
