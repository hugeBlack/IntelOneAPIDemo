#include "Tweaks.h"
#include "../NJSponsorBlockThumbnailLabelService.h"
#include "../NJSponsorBlockSettings.h"
#import <objc/runtime.h>

static const NSInteger kNJSBThumbnailBadgeTag = 0x53424247;
static const void *kNJSBThumbnailBadgeRequestBVIDKey = &kNJSBThumbnailBadgeRequestBVIDKey;

static void NJSBSetThumbnailBadgeRequestBVID(UIView *cardView, NSString *bvid) {
    if (!cardView) {
        return;
    }
    objc_setAssociatedObject(cardView, kNJSBThumbnailBadgeRequestBVIDKey, bvid, OBJC_ASSOCIATION_COPY_NONATOMIC);
}

static NSString *NJSBThumbnailBadgeRequestBVID(UIView *cardView) {
    if (!cardView) {
        return nil;
    }
    id value = objc_getAssociatedObject(cardView, kNJSBThumbnailBadgeRequestBVIDKey);
    return [value isKindOfClass:[NSString class]] ? value : nil;
}

@interface NJSBThumbnailVideoIdentity : NSObject

@property (nonatomic, copy) NSString *bvid;
@property (nonatomic, strong) NSNumber *aid;
@property (nonatomic, strong) NSNumber *cid;

@end

@implementation NJSBThumbnailVideoIdentity
@end

static id NJSBSafeValueForKey(id object, NSString *key) {
    if (!object || key.length == 0) {
        return nil;
    }

    @try {
        return [object valueForKey:key];
    } @catch (__unused NSException *exception) {
        if ([key hasPrefix:@"_"]) {
            return nil;
        }
        @try {
            return [object valueForKey:[@"_" stringByAppendingString:key]];
        } @catch (__unused NSException *innerException) {
            return nil;
        }
    }
}

static NSString *NJSBNormalizedIvarName(Ivar ivar) {
    const char *name = ivar_getName(ivar);
    NSString *ivarName = name ? [NSString stringWithUTF8String:name] : @"";
    if ([ivarName hasPrefix:@"_"]) {
        ivarName = [ivarName substringFromIndex:1];
    }
    NSString *lazyPrefix = @"$__lazy_storage_$_";
    if ([ivarName hasPrefix:lazyPrefix]) {
        ivarName = [ivarName substringFromIndex:lazyPrefix.length];
    }
    return ivarName;
}

static BOOL NJSBKeyIsNumericVideoIdentityField(NSString *key) {
    NSString *lowerKey = key.lowercaseString ?: @"";
    return [lowerKey isEqualToString:@"avid"] ||
           [lowerKey isEqualToString:@"aid"] ||
           [lowerKey isEqualToString:@"id"] ||
           [lowerKey isEqualToString:@"id_p"] ||
           [lowerKey isEqualToString:@"cid"];
}

static Ivar NJSBIvarForKey(id object, NSString *key) {
    if (!object || key.length == 0) {
        return NULL;
    }

    for (Class class = [object class]; class; class = class_getSuperclass(class)) {
        unsigned int ivarCount = 0;
        Ivar *ivars = class_copyIvarList(class, &ivarCount);
        for (unsigned int i = 0; i < ivarCount; i++) {
            Ivar ivar = ivars[i];
            if ([NJSBNormalizedIvarName(ivar) isEqualToString:key]) {
                free(ivars);
                return ivar;
            }
        }
        free(ivars);
    }
    return NULL;
}

static NSNumber *NJSBSearchVideoViewModelAID(id object) {
    Ivar ivar = NJSBIvarForKey(object, @"avID");
    if (!ivar) {
        return nil;
    }

    ptrdiff_t offset = ivar_getOffset(ivar);
    uint8_t *bytes = (__bridge void *)object;
    if (!bytes || offset < 0) {
        return nil;
    }

    uint64_t rawValue = 0;
    memcpy(&rawValue, bytes + offset, sizeof(rawValue));
    return rawValue > 0 ? @(rawValue) : nil;
}

static id NJSBSafeIvarValueForKey(id object, NSString *key) {
    Ivar ivar = NJSBIvarForKey(object, key);
    if (!ivar) {
        return nil;
    }

    const char *type = ivar_getTypeEncoding(ivar);
    if (type && type[0] == '@') {
        return object_getIvar(object, ivar);
    }

    ptrdiff_t offset = ivar_getOffset(ivar);
    uint8_t *bytes = (__bridge void *)object;
    if (!bytes || offset < 0) {
        return nil;
    }

    if (!type || type[0] == '\0') {
        return nil;
    }

    void *address = bytes + offset;
    if (type[0] == '?' && NJSBKeyIsNumericVideoIdentityField(key)) {
        return @(*(uint64_t *)address);
    }

    switch (type[0]) {
        case 'q': return @(*(int64_t *)address);
        case 'Q': return @(*(uint64_t *)address);
        case 'i': return @(*(int *)address);
        case 'I': return @(*(unsigned int *)address);
        case 'l': return @(*(long *)address);
        case 'L': return @(*(unsigned long *)address);
        case 's': return @(*(short *)address);
        case 'S': return @(*(unsigned short *)address);
        default: return nil;
    }
}

static UIView *NJSBThumbnailBadgeCoverViewForCard(id cardView) {
    id coverView = NJSBSafeValueForKey(cardView, @"coverImageView") ?: NJSBSafeIvarValueForKey(cardView, @"coverImageView");
    return [coverView isKindOfClass:[UIView class]] ? coverView : nil;
}

static NSString *NJSBThumbnailBVIDFromString(NSString *string) {
    if (![string isKindOfClass:[NSString class]] || string.length == 0) {
        return nil;
    }

    static NSRegularExpression *expression = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        expression = [NSRegularExpression regularExpressionWithPattern:@"\\bBV1[0-9A-Za-z]{9}\\b" options:0 error:nil];
    });

    NSTextCheckingResult *match = [expression firstMatchInString:string options:0 range:NSMakeRange(0, string.length)];
    return match ? [string substringWithRange:match.range] : nil;
}

static NSNumber *NJSBThumbnailNumberFromValue(id value) {
    if ([value isKindOfClass:[NSNumber class]]) {
        NSInteger integerValue = [value integerValue];
        return integerValue > 0 ? @(integerValue) : nil;
    }
    if ([value isKindOfClass:[NSString class]]) {
        NSString *string = [(NSString *)value stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if (string.length == 0) {
            return nil;
        }
        NSScanner *scanner = [NSScanner scannerWithString:string];
        long long rawValue = 0;
        if ([scanner scanLongLong:&rawValue] && scanner.isAtEnd && rawValue > 0) {
            return @(rawValue);
        }
    }
    return nil;
}

static NSNumber *NJSBThumbnailAIDFromString(NSString *string) {
    if (![string isKindOfClass:[NSString class]] || string.length == 0) {
        return nil;
    }

    static NSArray<NSRegularExpression *> *expressions = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        expressions = @[
            [NSRegularExpression regularExpressionWithPattern:@"(?:^|[/?&=#])(?:aid|avid)=([0-9]+)(?:$|[^0-9])" options:NSRegularExpressionCaseInsensitive error:nil],
            [NSRegularExpression regularExpressionWithPattern:@"(?:^|[^A-Za-z0-9])av([0-9]+)(?:$|[^0-9])" options:NSRegularExpressionCaseInsensitive error:nil],
            [NSRegularExpression regularExpressionWithPattern:@"(?:^|/)video/([0-9]+)(?:$|[^0-9])" options:NSRegularExpressionCaseInsensitive error:nil],
        ];
    });
    for (NSRegularExpression *expression in expressions) {
        NSTextCheckingResult *match = [expression firstMatchInString:string options:0 range:NSMakeRange(0, string.length)];
        if (match.numberOfRanges > 1) {
            NSString *rawValue = [string substringWithRange:[match rangeAtIndex:1]];
            NSNumber *number = NJSBThumbnailNumberFromValue(rawValue);
            if (number.integerValue > 0) {
                return number;
            }
        }
    }
    return nil;
}

static NSString *NJSBBVIDFromAID(uint64_t aid) {
    static NSString *alphabet = @"FcwAPNKTMug3GV5Lj7EJnHpWsx4tb8haYeviqBz6rkCy12mUSDQX9RdoZf";
    static const uint64_t xorCode = 23442827791579ULL;
    static const uint64_t maxAID = 1ULL << 51;
    static const NSUInteger encodeMap[] = {8, 7, 0, 5, 1, 3, 2, 4, 6};

    if (aid == 0 || aid >= maxAID) {
        return nil;
    }

    uint64_t value = (maxAID | aid) ^ xorCode;
    NSMutableArray<NSString *> *characters = [NSMutableArray arrayWithCapacity:9];
    for (NSUInteger i = 0; i < 9; i++) {
        [characters addObject:@""];
    }

    NSUInteger base = alphabet.length;
    for (NSUInteger i = 0; i < 9; i++) {
        NSUInteger index = (NSUInteger)(value % base);
        unichar character = [alphabet characterAtIndex:index];
        characters[encodeMap[i]] = [NSString stringWithCharacters:&character length:1];
        value /= base;
    }

    return [@"BV1" stringByAppendingString:[characters componentsJoinedByString:@""]];
}

static id NJSBThumbnailCandidateValueForObject(id object, NSString *key) {
    return NJSBSafeValueForKey(object, key) ?: NJSBSafeIvarValueForKey(object, key);
}

static void NJSBThumbnailApplyCandidateValue(id value, NSString *key, NJSBThumbnailVideoIdentity *identity) {
    if (!value) {
        return;
    }

    if ([value isKindOfClass:[NSString class]]) {
        NSString *string = value;
        NSString *bvid = identity.bvid.length == 0 ? NJSBThumbnailBVIDFromString(string) : nil;
        if (bvid.length > 0) {
            identity.bvid = bvid;
            return;
        }
        if (!identity.aid) {
            NSNumber *aid = NJSBThumbnailAIDFromString(string);
            if (aid.integerValue > 0) {
                identity.aid = aid;
            }
        }
    } else if ([value isKindOfClass:[NSURL class]]) {
        NJSBThumbnailApplyCandidateValue([(NSURL *)value absoluteString], key, identity);
        return;
    }

    NSString *lowerKey = key.lowercaseString ?: @"";
    if (!identity.aid && ([lowerKey isEqualToString:@"aid"] ||
                          [lowerKey isEqualToString:@"avid"] ||
                          [lowerKey isEqualToString:@"id"] ||
                          [lowerKey isEqualToString:@"id_p"])) {
        NSNumber *aid = NJSBThumbnailNumberFromValue(value);
        if (aid.unsignedLongLongValue > 0) {
            identity.aid = aid;
        }
    }
    if (!identity.cid && [lowerKey isEqualToString:@"cid"]) {
        identity.cid = NJSBThumbnailNumberFromValue(value);
    }
}

static void NJSBScanThumbnailIdentityFields(id object, NJSBThumbnailVideoIdentity *identity) {
    if (!object || identity.bvid.length > 0) {
        return;
    }

    static NSArray<NSString *> *keys = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        keys = @[
            @"bvid",
            @"bvidStr",
            @"bvidString",
            @"bvID",
            @"bvId",
            @"uri",
            @"url",
            @"param",
            @"link",
            @"coverURL",
            @"avID",
            @"aid",
            @"avid",
            @"id_p",
            @"id",
            @"cid",
        ];
    });
    for (NSString *key in keys) {
        id value = NJSBThumbnailCandidateValueForObject(object, key);
        NJSBThumbnailApplyCandidateValue(value, key, identity);
        if (identity.bvid.length > 0) {
            return;
        }
    }
}

NJSBThumbnailVideoIdentity *NJSBExtractThumbnailVideoIdentity(id object) {
    NJSBThumbnailVideoIdentity *identity = [[NJSBThumbnailVideoIdentity alloc] init];

    NSString *objectClassName = object ? NSStringFromClass([object class]) : @"";
    BOOL searchVideoViewModel = [objectClassName rangeOfString:@"BBSearchCardsProvider"].location != NSNotFound &&
                                [objectClassName rangeOfString:@"VideoViewModel"].location != NSNotFound;
    if (searchVideoViewModel) {
        NJSBThumbnailApplyCandidateValue(NJSBSearchVideoViewModelAID(object), @"avID", identity);
    } else {
        if ([objectClassName rangeOfString:@"VideoViewModel"].location != NSNotFound) {
            NJSBThumbnailApplyCandidateValue(NJSBThumbnailCandidateValueForObject(object, @"avID"), @"avID", identity);
        }
        NJSBScanThumbnailIdentityFields(object, identity);
        id basicInfo = NJSBThumbnailCandidateValueForObject(object, @"basicInfo");
        NJSBScanThumbnailIdentityFields(basicInfo, identity);
        if (!identity.cid) {
            NJSBThumbnailApplyCandidateValue(NJSBThumbnailCandidateValueForObject(object, @"cid"), @"cid", identity);
        }
    }

    if (identity.bvid.length == 0 && identity.aid.unsignedLongLongValue > 0) {
        identity.bvid = NJSBBVIDFromAID(identity.aid.unsignedLongLongValue);
    }

    return identity;
}

static void NJSBHideThumbnailBadgeFromCoverView(UIView *coverView);

static void NJSBSetThumbnailBadgeText(UIView *coverView, NSString *badgeText, NSString *category) {
    UIView *target = coverView.superview ?: coverView;
    if (!coverView || !target || badgeText.length == 0 || category.length == 0) {
        NJSBHideThumbnailBadgeFromCoverView(coverView);
        return;
    }

    UIView *existingBadge = [target viewWithTag:kNJSBThumbnailBadgeTag];
    UILabel *badge = [existingBadge isKindOfClass:[UILabel class]] ? (UILabel *)existingBadge : nil;
    if (existingBadge && !badge) {
        [existingBadge removeFromSuperview];
    }

    if (!badge) {
        badge = [[UILabel alloc] initWithFrame:CGRectZero];
        badge.tag = kNJSBThumbnailBadgeTag;
        badge.textColor = [UIColor whiteColor];
        badge.font = [UIFont systemFontOfSize:11 weight:UIFontWeightMedium];
        badge.textAlignment = NSTextAlignmentCenter;
        badge.layer.cornerRadius = 3;
        badge.clipsToBounds = YES;
        [target addSubview:badge];
    }

    badge.backgroundColor = [NJSponsorBlockSettings thumbnailBadgeColorForLabel:category];
    badge.hidden = NO;
    badge.text = badgeText;
    badge.frame = CGRectMake(4, 4, 34, 18);
}

static void NJSBHideThumbnailBadgeFromCoverView(UIView *coverView) {
    UIView *target = coverView.superview ?: coverView;
    UIView *badge = [target viewWithTag:kNJSBThumbnailBadgeTag];
    if ([badge isKindOfClass:[UILabel class]]) {
        badge.hidden = YES;
    }
}

static void NJSBUpdateThumbnailBadgeForVideoCard(UIView *cardView, UIView *coverView, id object);

static BOOL NJSBSearchModelIsVideoViewModel(id model) {
    NSString *className = model ? NSStringFromClass([model class]) : @"";
    return [className rangeOfString:@"BBSearchCardsProvider"].location != NSNotFound &&
           [className rangeOfString:@"VideoViewModel"].location != NSNotFound;
}

static UIImageView *NJSBLargestImageViewInView(UIView *view) {
    if (!view) {
        return nil;
    }

    UIImageView *largestImageView = [view isKindOfClass:[UIImageView class]] ? (UIImageView *)view : nil;
    CGFloat largestArea = largestImageView ? CGRectGetWidth(largestImageView.bounds) * CGRectGetHeight(largestImageView.bounds) : 0;
    for (UIView *subview in view.subviews) {
        UIImageView *imageView = NJSBLargestImageViewInView(subview);
        CGFloat area = imageView ? CGRectGetWidth(imageView.bounds) * CGRectGetHeight(imageView.bounds) : 0;
        if (imageView && (!largestImageView || area > largestArea)) {
            largestArea = area;
            largestImageView = imageView;
        }
    }
    return largestImageView;
}

static UIView *NJSBSearchCoverViewForCell(id cell) {
    return NJSBLargestImageViewInView([cell isKindOfClass:[UIView class]] ? (UIView *)cell : nil);
}

static void NJSBApplySearchThumbnailBadge(id cell, id model) {
    UIView *coverView = NJSBSearchCoverViewForCell(cell);
    if (!NJSBSearchModelIsVideoViewModel(model)) {
        NJSBSetThumbnailBadgeRequestBVID((UIView *)cell, nil);
        NJSBHideThumbnailBadgeFromCoverView(coverView);
        return;
    }

    if (!coverView) {
        NJSBSetThumbnailBadgeRequestBVID((UIView *)cell, nil);
        return;
    }

    NJSBUpdateThumbnailBadgeForVideoCard((UIView *)cell, coverView, model);
}

static void NJSBUpdateThumbnailBadgeForVideoCard(UIView *cardView, UIView *coverView, id object) {
    UIView *target = coverView.superview ?: coverView;
    if (![NJSponsorBlockSettings showVideoLabels]) {
        NJSBSetThumbnailBadgeRequestBVID(cardView, nil);
        NJSBHideThumbnailBadgeFromCoverView(coverView);
        return;
    }

    NJSBThumbnailVideoIdentity *identity = NJSBExtractThumbnailVideoIdentity(object);
    NSString *bvid = identity.bvid;

    if (!coverView || !target) {
        NJSBSetThumbnailBadgeRequestBVID(cardView, nil);
        NJSBHideThumbnailBadgeFromCoverView(coverView);
        return;
    }

    if (bvid.length == 0) {
        NJSBSetThumbnailBadgeRequestBVID(cardView, nil);
        NJSBHideThumbnailBadgeFromCoverView(coverView);
        return;
    }

    NJSBHideThumbnailBadgeFromCoverView(coverView);
    NJSBSetThumbnailBadgeRequestBVID(cardView, bvid);

    __weak id weakCardView = cardView;
    __weak id weakCoverView = coverView;
    NSString *requestedBVID = [bvid copy];
    [[NJSponsorBlockThumbnailLabelService sharedService] fetchLabelForBVID:requestedBVID completion:^(NSString *category, NSString *text) {
        id strongCardView = weakCardView;
        id strongCoverView = weakCoverView;
        if (!strongCardView || !strongCoverView) {
            return;
        }

        NSString *currentRequestBVID = NJSBThumbnailBadgeRequestBVID((UIView *)strongCardView);
        if (![currentRequestBVID isEqualToString:requestedBVID]) {
            return;
        }

        NJSBSetThumbnailBadgeText((UIView *)strongCoverView, text, category);
    }];
}

static void (*orig_BBMediaUniteRelateCell_installCellWithObject)(id self, SEL sel, id object) = nil;
static void hook_BBMediaUniteRelateCell_installCellWithObject(id self, SEL sel, id object) {
    orig_BBMediaUniteRelateCell_installCellWithObject(self, sel, object);

    UIView *cover = NJSBThumbnailBadgeCoverViewForCard(self);
    NJSBUpdateThumbnailBadgeForVideoCard((UIView *)self, cover, object);
}

static void (*orig_BBMediaUniteRelateCell_prepareForReuse)(id self, SEL sel) = nil;
static void hook_BBMediaUniteRelateCell_prepareForReuse(id self, SEL sel) {
    orig_BBMediaUniteRelateCell_prepareForReuse(self, sel);
    NJSBHideThumbnailBadgeFromCoverView(NJSBThumbnailBadgeCoverViewForCard(self));
    NJSBSetThumbnailBadgeRequestBVID((UIView *)self, nil);
}

static void NJSBApplySearchThumbnailBadgeAfterConfig(id cell, id model) {
    __weak id weakCell = cell;
    id capturedModel = model;
    dispatch_async(dispatch_get_main_queue(), ^{
        id strongCell = weakCell;
        if (strongCell) {
            NJSBApplySearchThumbnailBadge(strongCell, capturedModel);
        }
    });
}

static void (*orig_SearchVideoCell_config)(id self, SEL sel, id model) = nil;
static void hook_SearchVideoCell_config(id self, SEL sel, id model) {
    orig_SearchVideoCell_config(self, sel, model);
    NJSBApplySearchThumbnailBadgeAfterConfig(self, model);
}

static void (*orig_SearchImageTextCell_config)(id self, SEL sel, id model) = nil;
static void hook_SearchImageTextCell_config(id self, SEL sel, id model) {
    orig_SearchImageTextCell_config(self, sel, model);
    NJSBApplySearchThumbnailBadgeAfterConfig(self, model);
}

static UIView *NJSBUserSpaceCoverViewForCell(id cell) {
    id coverView = NJSBSafeValueForKey(cell, @"coverView") ?: NJSBSafeIvarValueForKey(cell, @"coverView");
    return [coverView isKindOfClass:[UIView class]] ? coverView : nil;
}

static id NJSBUserSpaceItemModelForCell(id cell, id object) {
    if (object) {
        return object;
    }
    return NJSBSafeValueForKey(cell, @"itemModel") ?: NJSBSafeIvarValueForKey(cell, @"itemModel");
}

static void NJSBClearUserSpaceThumbnailBadge(id cell) {
    if (![cell isKindOfClass:[UIView class]]) {
        return;
    }

    UIView *coverView = NJSBUserSpaceCoverViewForCell(cell);
    NJSBSetThumbnailBadgeRequestBVID((UIView *)cell, nil);
    NJSBHideThumbnailBadgeFromCoverView(coverView);
}

static void NJSBApplyUserSpaceThumbnailBadge(id cell, id object) {
    if (![cell isKindOfClass:[UIView class]]) {
        return;
    }

    UIView *coverView = NJSBUserSpaceCoverViewForCell(cell);
    id model = NJSBUserSpaceItemModelForCell(cell, object);
    if (!coverView) {
        NJSBSetThumbnailBadgeRequestBVID((UIView *)cell, nil);
        return;
    }

    NJSBUpdateThumbnailBadgeForVideoCard((UIView *)cell, coverView, model);
}

static void (*orig_BBPhoneUserSpaceUploadVideoCell_installObj)(id self, SEL sel, id object) = nil;
static void hook_BBPhoneUserSpaceUploadVideoCell_installObj(id self, SEL sel, id object) {
    NJSBClearUserSpaceThumbnailBadge(self);
    orig_BBPhoneUserSpaceUploadVideoCell_installObj(self, sel, object);
    NJSBApplyUserSpaceThumbnailBadge(self, object);
}

static void (*orig_BBPhoneUserSpaceUploadVideoCell_prepareForReuse)(id self, SEL sel) = nil;
static void hook_BBPhoneUserSpaceUploadVideoCell_prepareForReuse(id self, SEL sel) {
    orig_BBPhoneUserSpaceUploadVideoCell_prepareForReuse(self, sel);
    NJSBClearUserSpaceThumbnailBadge(self);
}

static void (*orig_BBPhoneUserSpaceHomeVideoCell_installObj)(id self, SEL sel, id object) = nil;
static void hook_BBPhoneUserSpaceHomeVideoCell_installObj(id self, SEL sel, id object) {
    NJSBClearUserSpaceThumbnailBadge(self);
    orig_BBPhoneUserSpaceHomeVideoCell_installObj(self, sel, object);
    NJSBApplyUserSpaceThumbnailBadge(self, object);
}

void initThumbnailBadgeHooks(void) {
    Class relateCellClass = objc_getClass("BBMediaUniteRelateCell");
    if (relateCellClass) {
        JRSwizzleInstanceMethod(relateCellClass,
                                NSSelectorFromString(@"installCellWithObject:"),
                                (IMP)hook_BBMediaUniteRelateCell_installCellWithObject,
                                (IMP *)&orig_BBMediaUniteRelateCell_installCellWithObject);

        JRSwizzleInstanceMethod(relateCellClass,
                                @selector(prepareForReuse),
                                (IMP)hook_BBMediaUniteRelateCell_prepareForReuse,
                                (IMP *)&orig_BBMediaUniteRelateCell_prepareForReuse);
    }

    Class searchVideoCellClass = objc_getClass("_TtCO21BBSearchCardsProvider6Result9VideoCell");
    if (searchVideoCellClass) {
        JRSwizzleInstanceMethod(searchVideoCellClass,
                                NSSelectorFromString(@"config:"),
                                (IMP)hook_SearchVideoCell_config,
                                (IMP *)&orig_SearchVideoCell_config);
    }

    Class searchImageTextCellClass = objc_getClass("_TtCO21BBSearchCardsProvider6Result13ImageTextCell");
    if (searchImageTextCellClass) {
        JRSwizzleInstanceMethod(searchImageTextCellClass,
                                NSSelectorFromString(@"config:"),
                                (IMP)hook_SearchImageTextCell_config,
                                (IMP *)&orig_SearchImageTextCell_config);
    }

    Class userSpaceUploadVideoCellClass = objc_getClass("BBPhoneUserSpaceUploadVideoCell");
    if (userSpaceUploadVideoCellClass) {
        JRSwizzleInstanceMethod(userSpaceUploadVideoCellClass,
                                NSSelectorFromString(@"installObj:"),
                                (IMP)hook_BBPhoneUserSpaceUploadVideoCell_installObj,
                                (IMP *)&orig_BBPhoneUserSpaceUploadVideoCell_installObj);

        JRSwizzleInstanceMethod(userSpaceUploadVideoCellClass,
                                @selector(prepareForReuse),
                                (IMP)hook_BBPhoneUserSpaceUploadVideoCell_prepareForReuse,
                                (IMP *)&orig_BBPhoneUserSpaceUploadVideoCell_prepareForReuse);
    }

    Class userSpaceHomeVideoCellClass = objc_getClass("BBPhoneUserSpaceHomeVideoCell");
    if (userSpaceHomeVideoCellClass) {
        JRSwizzleInstanceMethod(userSpaceHomeVideoCellClass,
                                NSSelectorFromString(@"installObj:"),
                                (IMP)hook_BBPhoneUserSpaceHomeVideoCell_installObj,
                                (IMP *)&orig_BBPhoneUserSpaceHomeVideoCell_installObj);
    }
}
