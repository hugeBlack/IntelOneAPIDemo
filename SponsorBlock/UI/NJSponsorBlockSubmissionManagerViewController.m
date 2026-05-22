//
//  NJSponsorBlockSubmissionManagerViewController.m
//  SponsorBlock
//

#import "NJSponsorBlockSubmissionManagerViewController.h"
#import "../Models/NJSponsorBlockSegment.h"
#import "../Settings/NJSponsorBlockSettings.h"
#import "../Services/NJSponsorBlockUnsubmittedSegmentStore.h"
#import <math.h>

static NSString * const NJSponsorBlockSubmissionCellID = @"NJSponsorBlockSubmissionCellID";

@interface NJSponsorBlockSubmissionManagerViewController ()

@property (nonatomic, strong) NSArray<NSString *> *videoKeys;
@property (nonatomic, assign) BOOL submissionInFlight;

@property NJSponsorBlockManager* manager;

- (void)reloadDrafts;
- (void)closeTapped;
- (NSArray<NJSponsorBlockSegment *> *)segmentsForSection:(NSInteger)section;
- (NSString *)videoKeyForSection:(NSInteger)section;
- (NSString *)videoIDFromKey:(NSString *)key;
- (NSInteger)cidFromKey:(NSString *)key;
- (void)configureSummaryCell:(UITableViewCell *)cell row:(NSInteger)row;
- (void)configureGroupCell:(UITableViewCell *)cell indexPath:(NSIndexPath *)indexPath;
- (void)openVideoForKey:(NSString *)key;
- (void)submitVideoForKey:(NSString *)key;
- (void)confirmClearVideoForKey:(NSString *)key;
- (void)confirmClearAll;
- (void)presentSegmentActions:(NJSponsorBlockSegment *)segment;
- (void)presentEditForSegment:(NJSponsorBlockSegment *)segment;
- (void)confirmDeleteSegment:(NJSponsorBlockSegment *)segment;
- (BOOL)validateSegment:(NJSponsorBlockSegment *)segment start:(NSTimeInterval)start end:(NSTimeInterval)end;
- (void)showValidationError:(NSString *)message;
- (NSString *)stringFromTime:(NSTimeInterval)time;
- (void)postDraftsChangedNotification;

@end

@implementation NJSponsorBlockSubmissionManagerViewController

- (instancetype)initWithManager:(NJSponsorBlockManager*)manager {
    _manager = manager;
    return [super initWithStyle:UITableViewStyleGrouped];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"未提交片段";
    self.tableView.rowHeight = 52;
    self.tableView.backgroundColor = [UIColor systemBackgroundColor];
    [self.tableView registerClass:[UITableViewCell class] forCellReuseIdentifier:NJSponsorBlockSubmissionCellID];
    self.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:@"清空"
                                                                              style:UIBarButtonItemStylePlain
                                                                             target:self
                                                                             action:@selector(clearAllTapped)];
    [self reloadDrafts];
}

- (void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    if (self.navigationController.presentingViewController && self.navigationController.viewControllers.firstObject == self) {
        self.navigationItem.leftBarButtonItem = [[UIBarButtonItem alloc] initWithTitle:@"关闭"
                                                                                  style:UIBarButtonItemStylePlain
                                                                                 target:self
                                                                                 action:@selector(closeTapped)];
    }
    [self reloadDrafts];
    [self.tableView reloadData];
}

- (void)closeTapped {
    [self dismissViewControllerAnimated:YES completion:nil];
}

- (void)reloadDrafts {
    self.videoKeys = [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] videoKeys];
    self.navigationItem.rightBarButtonItem.enabled = [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] totalSegmentCount] > 0;
}

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return self.videoKeys.count + 1;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    if (section == 0) {
        return [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] totalSegmentCount] > 0 ? 2 : 1;
    }
    return [self segmentsForSection:section].count + 3;
}

- (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section {
    if (section == 0) {
        return @"总览";
    }
    NSString *key = [self videoKeyForSection:section];
    NSArray<NJSponsorBlockSegment *> *segments = [self segmentsForSection:section];
    return [NSString stringWithFormat:@"%@ · %lu 段", key, (unsigned long)segments.count];
}

- (NSString *)tableView:(UITableView *)tableView titleForFooterInSection:(NSInteger)section {
    if (section == 0) {
        return @"片段录制完成后会先保存在这里，只有点击提交才会发送到 SponsorBlock 服务器。";
    }
    return nil;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleValue1 reuseIdentifier:NJSponsorBlockSubmissionCellID];
    cell.textLabel.font = [UIFont systemFontOfSize:16];
    cell.detailTextLabel.font = [UIFont systemFontOfSize:13];
    cell.detailTextLabel.textColor = [UIColor secondaryLabelColor];
    cell.selectionStyle = UITableViewCellSelectionStyleDefault;
    cell.accessoryType = UITableViewCellAccessoryNone;

    if (indexPath.section == 0) {
        [self configureSummaryCell:cell row:indexPath.row];
    } else {
        [self configureGroupCell:cell indexPath:indexPath];
    }
    return cell;
}

- (void)configureSummaryCell:(UITableViewCell *)cell row:(NSInteger)row {
    NJSponsorBlockUnsubmittedSegmentStore *store = [NJSponsorBlockUnsubmittedSegmentStore sharedStore];
    if (row == 0) {
        cell.textLabel.text = @"本地未提交片段";
        cell.detailTextLabel.text = [NSString stringWithFormat:@"%lu 个视频 · %lu 段",
                                     (unsigned long)store.videoCount,
                                     (unsigned long)store.totalSegmentCount];
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        return;
    }
    cell.textLabel.text = @"清除全部未提交片段";
    cell.textLabel.textColor = [UIColor systemRedColor];
}

- (void)configureGroupCell:(UITableViewCell *)cell indexPath:(NSIndexPath *)indexPath {
    NSArray<NJSponsorBlockSegment *> *segments = [self segmentsForSection:indexPath.section];
    if (indexPath.row == 0) {
        cell.textLabel.text = @"打开原视频";
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    if (indexPath.row == 1) {
        cell.textLabel.text = @"提交此视频";
        cell.detailTextLabel.text = self.submissionInFlight ? @"提交中" : nil;
        cell.textLabel.textColor = [UIColor systemBlueColor];
        return;
    }
    if (indexPath.row == 2) {
        cell.textLabel.text = @"清除此视频片段";
        cell.textLabel.textColor = [UIColor systemRedColor];
        return;
    }

    NJSponsorBlockSegment *segment = segments[indexPath.row - 3];
    cell.textLabel.text = [NSString stringWithFormat:@"%@ · %@", [NJSponsorBlockSettings titleForCategory:segment.category], segment.actionType.length > 0 ? segment.actionType : @"skip"];
    if ([segment.actionType isEqualToString:@"poi"]) {
        cell.detailTextLabel.text = [NSString stringWithFormat:@"%@", [self stringFromTime:segment.startTime]];
    } else {
        cell.detailTextLabel.text = [NSString stringWithFormat:@"%@-%@", [self stringFromTime:segment.startTime], [self stringFromTime:segment.endTime]];
    }
    cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    if (indexPath.section == 0) {
        if (indexPath.row == 1) {
            [self confirmClearAll];
        }
        return;
    }

    NSString *key = [self videoKeyForSection:indexPath.section];
    if (indexPath.row == 0) {
        [self openVideoForKey:key];
        return;
    }
    if (indexPath.row == 1) {
        [self submitVideoForKey:key];
        return;
    }
    if (indexPath.row == 2) {
        [self confirmClearVideoForKey:key];
        return;
    }

    NSArray<NJSponsorBlockSegment *> *segments = [self segmentsForSection:indexPath.section];
    [self presentSegmentActions:segments[indexPath.row - 3]];
}

- (NSArray<NJSponsorBlockSegment *> *)segmentsForSection:(NSInteger)section {
    NSString *key = [self videoKeyForSection:section];
    return [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] segmentsForVideoKey:key];
}

- (NSString *)videoKeyForSection:(NSInteger)section {
    NSInteger index = section - 1;
    if (index < 0 || index >= self.videoKeys.count) {
        return @"";
    }
    return self.videoKeys[index];
}

- (NSString *)videoIDFromKey:(NSString *)key {
    NSRange range = [key rangeOfString:@":" options:NSBackwardsSearch];
    return range.location == NSNotFound ? key : [key substringToIndex:range.location];
}

- (NSInteger)cidFromKey:(NSString *)key {
    NSRange range = [key rangeOfString:@":" options:NSBackwardsSearch];
    if (range.location == NSNotFound || NSMaxRange(range) >= key.length) {
        return 0;
    }
    return [[key substringFromIndex:NSMaxRange(range)] integerValue];
}

- (void)openVideoForKey:(NSString *)key {
    NSString *videoID = [self videoIDFromKey:key];
    if (videoID.length == 0) {
        return;
    }
    NSString *appURLString = [NSString stringWithFormat:@"bilibili://video/%@", videoID];
    NSString *webURLString = [NSString stringWithFormat:@"https://www.bilibili.com/video/%@", videoID];
    NSURL *appURL = [NSURL URLWithString:appURLString];
    NSURL *webURL = [NSURL URLWithString:webURLString];
    [UIApplication.sharedApplication openURL:appURL options:@{} completionHandler:^(BOOL success) {
        if (!success) {
            [UIApplication.sharedApplication openURL:webURL options:@{} completionHandler:nil];
        }
    }];
}

- (void)submitVideoForKey:(NSString *)key {
    if (self.submissionInFlight) {
        return;
    }
    NSString *videoID = [self videoIDFromKey:key];
    NSInteger cid = [self cidFromKey:key];
    NJSponsorBlockManager *manager = _manager;
    if (![manager.videoID isEqualToString:videoID] || manager.cid != cid) {
        UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"请先打开原视频"
                                                                       message:@"只能提交当前播放器正在播放的视频片段。"
                                                                preferredStyle:UIAlertControllerStyleAlert];
        [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
        [alert addAction:[UIAlertAction actionWithTitle:@"打开原视频" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
            [self openVideoForKey:key];
        }]];
        [self presentViewController:alert animated:YES completion:nil];
        return;
    }

    self.submissionInFlight = YES;
    [self.tableView reloadData];
    __weak typeof(self) weakSelf = self;
    [manager.submissionController submitSegmentsForCurrentVideoWithCompletion:^(BOOL success, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            strongSelf.submissionInFlight = NO;
            [strongSelf reloadDrafts];
            [strongSelf.tableView reloadData];
            NSString *title = success ? @"提交成功" : @"提交失败";
            NSString *message = success ? @"本视频未提交片段已清除。" : (error.localizedDescription ?: @"请稍后重试");
            UIAlertController *alert = [UIAlertController alertControllerWithTitle:title message:message preferredStyle:UIAlertControllerStyleAlert];
            [alert addAction:[UIAlertAction actionWithTitle:@"好的" style:UIAlertActionStyleDefault handler:nil]];
            [strongSelf presentViewController:alert animated:YES completion:nil];
        });
    }];
}

- (void)confirmClearVideoForKey:(NSString *)key {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"清除此视频片段"
                                                                   message:@"确定清除此视频的所有未提交片段吗？"
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"清除" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        NSString *videoID = [self videoIDFromKey:key];
        NSInteger cid = [self cidFromKey:key];
        NJSponsorBlockManager *manager = self->_manager;
        if ([manager.videoID isEqualToString:videoID] && manager.cid == cid) {
            [manager.submissionController clearSegmentsForCurrentVideo];
        } else {
            [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] removeSegmentsForVideoID:videoID cid:cid];
            [self postDraftsChangedNotification];
        }
        [self reloadDrafts];
        [self.tableView reloadData];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)clearAllTapped {
    [self confirmClearAll];
}

- (void)confirmClearAll {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"清除全部未提交片段"
                                                                   message:@"确定清除所有本地未提交片段吗？此操作不可撤销。"
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"清除" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [self.manager.submissionController clearAllSegments];
        [self reloadDrafts];
        [self.tableView reloadData];
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)presentSegmentActions:(NJSponsorBlockSegment *)segment {
    UIAlertController *sheet = [UIAlertController alertControllerWithTitle:[NJSponsorBlockSettings titleForCategory:segment.category]
                                                                   message:[NSString stringWithFormat:@"%@-%@", [self stringFromTime:segment.startTime], [self stringFromTime:segment.endTime]]
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [sheet addAction:[UIAlertAction actionWithTitle:@"编辑时间" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        [self presentEditForSegment:segment];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"删除此片段" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        [self confirmDeleteSegment:segment];
    }]];
    [sheet addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    sheet.popoverPresentationController.sourceView = self.view;
    sheet.popoverPresentationController.sourceRect = self.view.bounds;
    [self presentViewController:sheet animated:YES completion:nil];
}

- (void)presentEditForSegment:(NJSponsorBlockSegment *)segment {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"编辑片段时间" message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.keyboardType = UIKeyboardTypeDecimalPad;
        textField.text = [NSString stringWithFormat:@"%.3f", segment.startTime];
        textField.placeholder = @"开始秒数";
    }];
    [alert addTextFieldWithConfigurationHandler:^(UITextField *textField) {
        textField.keyboardType = UIKeyboardTypeDecimalPad;
        textField.text = [NSString stringWithFormat:@"%.3f", segment.endTime];
        textField.placeholder = @"结束秒数";
        textField.enabled = ![segment.actionType isEqualToString:@"poi"];
    }];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"保存" style:UIAlertActionStyleDefault handler:^(__unused UIAlertAction *action) {
        NSTimeInterval start = [alert.textFields.firstObject.text doubleValue];
        NSTimeInterval end = [segment.actionType isEqualToString:@"poi"] ? start : [alert.textFields.lastObject.text doubleValue];
        if (![self validateSegment:segment start:start end:end]) {
            return;
        }
        BOOL updated = [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] updateSegmentWithUUID:segment.uuid
                                                                                          videoID:segment.videoID
                                                                                              cid:segment.cid
                                                                                        startTime:start
                                                                                          endTime:end];
        if (updated) {
            [self postDraftsChangedNotification];
            [self reloadDrafts];
            [self.tableView reloadData];
        }
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (BOOL)validateSegment:(NJSponsorBlockSegment *)segment start:(NSTimeInterval)start end:(NSTimeInterval)end {
    if (!isfinite(start) || !isfinite(end) || start < 0 || end < 0) {
        [self showValidationError:@"请输入有效秒数"];
        return NO;
    }
    if (![segment.actionType isEqualToString:@"poi"] && end <= start) {
        [self showValidationError:@"结束时间必须大于开始时间"];
        return NO;
    }
    if (segment.videoDuration > 0 && (start > segment.videoDuration || end > segment.videoDuration)) {
        [self showValidationError:@"时间不能超过视频时长"];
        return NO;
    }
    NSTimeInterval minDuration = MAX([NJSponsorBlockSettings minDuration], 0.5);
    if (![segment.actionType isEqualToString:@"poi"] && end - start < minDuration) {
        [self showValidationError:[NSString stringWithFormat:@"片段至少需要 %@", [self stringFromTime:minDuration]]];
        return NO;
    }
    return YES;
}

- (void)showValidationError:(NSString *)message {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"保存失败" message:message preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"好的" style:UIAlertActionStyleDefault handler:nil]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (void)confirmDeleteSegment:(NJSponsorBlockSegment *)segment {
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"删除片段"
                                                                   message:@"确定删除这个未提交片段吗？"
                                                            preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"删除" style:UIAlertActionStyleDestructive handler:^(__unused UIAlertAction *action) {
        BOOL removed = [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] removeSegmentWithUUID:segment.uuid videoID:segment.videoID cid:segment.cid];
        if (removed) {
            [self postDraftsChangedNotification];
            [self reloadDrafts];
            [self.tableView reloadData];
        }
    }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (NSString *)stringFromTime:(NSTimeInterval)time {
    NSInteger total = MAX(0, (NSInteger)round(time));
    NSInteger seconds = total % 60;
    NSInteger minutes = (total / 60) % 60;
    NSInteger hours = total / 3600;
    if (hours > 0) {
        return [NSString stringWithFormat:@"%ld:%02ld:%02ld", (long)hours, (long)minutes, (long)seconds];
    }
    return [NSString stringWithFormat:@"%ld:%02ld", (long)minutes, (long)seconds];
}

- (void)postDraftsChangedNotification {
    [[NSNotificationCenter defaultCenter] postNotificationName:NJSponsorBlockStateDidChangeNotification object:_manager];
}

@end
