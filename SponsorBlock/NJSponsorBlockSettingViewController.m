//
//  NJSponsorBlockSettingViewController.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockSettingViewController.h"
#import "NJSponsorBlockSettings.h"

static NSString * const NJSponsorBlockSettingCellID = @"NJSponsorBlockSettingCellID";

typedef NS_ENUM(NSInteger, NJSponsorBlockSettingSection) {
    NJSponsorBlockSettingSectionGeneral = 0,
    NJSponsorBlockSettingSectionBehavior,
    NJSponsorBlockSettingSectionCategories,
    NJSponsorBlockSettingSectionServer,
    NJSponsorBlockSettingSectionAbout,
    NJSponsorBlockSettingSectionCount,
};

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

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return NJSponsorBlockSettingSectionCount;
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    switch (section) {
        case NJSponsorBlockSettingSectionGeneral:
            return 2;
        case NJSponsorBlockSettingSectionBehavior:
            return 3;
        case NJSponsorBlockSettingSectionCategories:
            return [NJSponsorBlockSettings categoryOptions].count;
        case NJSponsorBlockSettingSectionServer:
            return 2;
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
        case NJSponsorBlockSettingSectionBehavior:
            return @"跳过行为";
        case NJSponsorBlockSettingSectionCategories:
            return @"分类行为";
        case NJSponsorBlockSettingSectionServer:
            return @"服务器";
        case NJSponsorBlockSettingSectionAbout:
            return @"说明";
        default:
            return nil;
    }
}

- (NSString *)tableView:(UITableView *)tableView titleForFooterInSection:(NSInteger)section {
    if (section == NJSponsorBlockSettingSectionAbout) {
        return @"当前 iOS 客户端支持官方插件的播放跳过、分类行为、服务器、缓存配置和基础片段投稿。投稿为移动端简化流程，不包含浏览器扩展的完整编辑器、快捷键、动态/评论屏蔽、缩略图标签等功能。";
    }
    return nil;
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
        case NJSponsorBlockSettingSectionBehavior:
            [self configureBehaviorCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionCategories:
            [self configureCategoryCell:cell row:indexPath.row];
            break;
        case NJSponsorBlockSettingSectionServer:
            [self configureServerCell:cell row:indexPath.row];
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
    cell.textLabel.text = @"缓存请求结果";
    cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings cacheEnabled] tag:101];
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

- (void)configureServerCell:(UITableViewCell *)cell row:(NSInteger)row {
    if (row == 0) {
        cell.textLabel.text = @"服务器地址";
        cell.detailTextLabel.text = [NJSponsorBlockSettings serverBaseURLString];
        cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
        return;
    }
    cell.textLabel.text = @"使用测试服务器";
    cell.accessoryView = [self switchWithOn:[NJSponsorBlockSettings testingServerEnabled] tag:103];
    cell.selectionStyle = UITableViewCellSelectionStyleNone;
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
    if (indexPath.section == NJSponsorBlockSettingSectionCategories) {
        [self presentCategoryActionSheetForRow:indexPath.row sourceCell:[tableView cellForRowAtIndexPath:indexPath]];
        return;
    }
    if (indexPath.section == NJSponsorBlockSettingSectionServer && indexPath.row == 0) {
        [self presentServerInput];
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

@end
