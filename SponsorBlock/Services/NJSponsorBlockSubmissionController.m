//
//  NJSponsorBlockSubmissionController.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockSubmissionController.h"
#import "../Models/NJSponsorBlockSegment.h"
#import "../Settings/NJSponsorBlockSettings.h"
#import "NJSponsorBlockUnsubmittedSegmentStore.h"
#import "../Tweaks/RuntimeClasses/SponsorBlockHintToast.h"
#import <math.h>

// ── Time formatting helpers ──────────────────────────────────────────────

NSString *NJSBFormatTime(NSTimeInterval time) {
    if (time <= 0 || !isfinite(time)) return @"0:00.000";
    NSInteger mins = (NSInteger)(time / 60.0);
    double secs = time - mins * 60.0;
    return [NSString stringWithFormat:@"%ld:%06.3f", (long)mins, secs];
}

NSString *NJSBFormatCompact(NSTimeInterval time) {
    if (time <= 0 || !isfinite(time)) return @"0s";
    NSInteger secs = (NSInteger)round(time);
    if (secs < 60) return [NSString stringWithFormat:@"%lds", (long)secs];
    return [NSString stringWithFormat:@"%ld:%02ld", (long)(secs / 60), (long)(secs % 60)];
}

@interface NJSponsorBlockSubmissionController ()

@property (nonatomic, strong) NJSponsorBlockService *service;
@property (nonatomic, weak) id<NJSponsorBlockSubmissionDelegate> delegate;

// Draft state
@property (nonatomic, assign, readwrite) BOOL draftInProgress;
@property (nonatomic, copy) NSString *draftCategory;
@property (nonatomic, copy) NSString *draftVideoID;
@property (nonatomic, assign) NSInteger draftCID;
@property (nonatomic, assign) NSTimeInterval draftVideoDuration;
@property (nonatomic, assign) NSTimeInterval draftStartTime;

// Submission in-flight
@property (nonatomic, assign, readwrite) BOOL submissionInFlight;

- (NSError *)submissionErrorWithCode:(NSInteger)code message:(NSString *)message;
- (void)showDraftToast;
- (void)finishDraft;
- (void)submitStoredSegment:(NJSponsorBlockSegment *)segment
                    videoID:(NSString *)videoID
                        cid:(NSInteger)cid
                   duration:(NSTimeInterval)duration
                 completion:(void (^)(BOOL success, NSError *_Nullable error))completion;
- (void)submitSegments:(NSArray<NJSponsorBlockSegment *> *)segments
               videoID:(NSString *)videoID
                   cid:(NSInteger)cid
              duration:(NSTimeInterval)duration
                 index:(NSUInteger)index
            completion:(nullable NJSponsorBlockSubmitCompletion)completion;

@end

@implementation NJSponsorBlockSubmissionController

- (instancetype)initWithService:(NJSponsorBlockService *)service
                       delegate:(id<NJSponsorBlockSubmissionDelegate>)delegate {
    self = [super init];
    if (self) {
        _service = service;
        _delegate = delegate;
    }
    return self;
}

// ── Draft recording ──────────────────────────────────────────────────────

- (void)beginDraftWithCategory:(NSString *)category {
    id<NJSponsorBlockSubmissionDelegate> ctx = self.delegate;
    BBPlayerContext *playerContext = ctx.playerContext;

    NSTimeInterval currentTime = ctx.currentPlaybackTime;
    if (currentTime < 0 || !isfinite(currentTime)) {
        [playerContext.toastWidgetService showToastContainerWithText:@"无法提交, 无法获取当前播放时间"];
        return;
    }
    if (ctx.videoID.length == 0 || ctx.cid <= 0) {
        [playerContext.toastWidgetService showToastContainerWithText:@"无法提交, 尚未识别当前视频"];
        return;
    }
    if (ctx.estimatedVideoDuration <= 0 || !isfinite(ctx.estimatedVideoDuration)) {
        [playerContext.toastWidgetService showToastContainerWithText:@"无法提交, 暂未获取视频时长，稍后再试"];
        return;
    }

    self.draftCategory      = category;
    self.draftVideoID       = ctx.videoID;
    self.draftCID           = ctx.cid;
    self.draftVideoDuration = ctx.estimatedVideoDuration;
    self.draftStartTime     = currentTime;
    self.draftInProgress    = YES;

    if ([category isEqualToString:@"poi_highlight"]) {
        [self finishDraft];
        return;
    }

    [self showDraftToast];
}

- (void)showDraftToast {
    id<NJSponsorBlockSubmissionDelegate> ctx = self.delegate;
    BBPlayerContext *playerContext = ctx.playerContext;
    if (!playerContext) return;

    NSString *detail = [NSString stringWithFormat:@"%@ · 起点 %@",
                        [NJSponsorBlockSettings titleForCategory:self.draftCategory],
                        NJSBFormatTime(self.draftStartTime)];

    __weak typeof(self) weakSelf = self;
    id toast = NJSponsorBlockCreateHintToast(
        playerContext,
        @"已记录起点",
        detail,
        @"终点提交",
        @"取消",
        ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) return;
            [strongSelf finishDraft];
        },
        ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) return;
            [strongSelf cancelDraft];
        },
        nil,
        300.0,  // Effectively indefinite (5 min)
        NO
    );
    if (toast) {
        [playerContext.toastWidgetService presentCustomToast:toast];
    }
}

- (void)finishDraft {
    if (!self.draftInProgress) return;

    id<NJSponsorBlockSubmissionDelegate> ctx = self.delegate;
    BBPlayerContext *playerContext = ctx.playerContext;

    NSTimeInterval currentTime = ctx.currentPlaybackTime;
    if (currentTime < 0 || !isfinite(currentTime)) {
        [playerContext.toastWidgetService showToastContainerWithText:@"无法提交, 无法获取当前播放时间"];
        return;
    }

    if (![ctx.videoID isEqualToString:self.draftVideoID] || ctx.cid != self.draftCID) {
        [self cancelDraft];
        [playerContext.toastWidgetService showToastContainerWithText:@"已取消提交, 当前视频已切换，请重新记录片段"];
        return;
    }

    NSTimeInterval duration = self.draftVideoDuration;
    if (duration <= 0 || !isfinite(duration)) {
        [playerContext.toastWidgetService showToastContainerWithText:@"无法提交，暂未获取视频时长，稍后再试"];
        return;
    }

    NSTimeInterval rawStart = self.draftStartTime;
    NSTimeInterval start = round((MIN(rawStart, currentTime) + DBL_EPSILON) * 1000.0) / 1000.0;
    NSTimeInterval end   = round((MAX(rawStart, currentTime) + DBL_EPSILON) * 1000.0) / 1000.0;
    if (end > duration) end = duration;

    NSString *category = self.draftCategory;
    NSTimeInterval minDuration = [category isEqualToString:@"poi_highlight"] ? 0 : MAX([NJSponsorBlockSettings minDuration], 0.5);
    if (end - start < minDuration) {
        [playerContext.toastWidgetService showToastContainerWithText:
            [NSString stringWithFormat:@"片段太短, 至少需要 %@", NJSBFormatCompact(minDuration)]];
        return;
    }

    NSString *actionType = [category isEqualToString:@"poi_highlight"] ? @"poi" : @"skip";

    NJSponsorBlockSegment *localSegment = [self addSegmentWithCategory:category
                                                            actionType:actionType
                                                               segment:@[@(start), @(end)]];
    // Clear draft state
    self.draftInProgress    = NO;
    self.draftCategory      = nil;
    self.draftVideoID       = nil;
    self.draftCID           = 0;
    self.draftVideoDuration = 0;
    self.draftStartTime     = 0;

    if (!localSegment) {
        [playerContext.toastWidgetService showToastContainerWithText:@"保存失败, 无法保存本地未提交片段"];
        return;
    }

    [playerContext.toastWidgetService showToastContainerWithText:@"已保存草稿, 可在提交菜单中提交或管理"];
}

- (void)cancelDraft {
    self.draftInProgress    = NO;
    self.draftCategory      = nil;
    self.draftVideoID       = nil;
    self.draftCID           = 0;
    self.draftVideoDuration = 0;
    self.draftStartTime     = 0;
}

// ── Unsubmitted segment store access ────────────────────────────────────

- (nullable NJSponsorBlockSegment *)addSegmentWithCategory:(NSString *)category
                                                actionType:(NSString *)actionType
                                                   segment:(NSArray<NSNumber *> *)segment {
    id<NJSponsorBlockSubmissionDelegate> ctx = self.delegate;
    NJSponsorBlockSegment *localSegment =
        [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] addSegmentForVideoID:ctx.videoID
                                                                              cid:ctx.cid
                                                                         category:category
                                                                       actionType:actionType
                                                                          segment:segment
                                                                    videoDuration:ctx.estimatedVideoDuration];
    if (localSegment) {
        [ctx submissionControllerDidChangeState:self];
    }
    return localSegment;
}

- (NSArray<NJSponsorBlockSegment *> *)segmentsForCurrentVideo {
    id<NJSponsorBlockSubmissionDelegate> ctx = self.delegate;
    return [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] segmentsForVideoID:ctx.videoID cid:ctx.cid];
}

- (void)clearSegmentsForCurrentVideo {
    id<NJSponsorBlockSubmissionDelegate> ctx = self.delegate;
    NSArray<NJSponsorBlockSegment *> *segments = [self segmentsForCurrentVideo];
    [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] removeSegmentsForVideoID:ctx.videoID cid:ctx.cid];
    [ctx submissionController:self didRemoveUnsubmittedSegments:segments];
}

- (void)clearAllSegments {
    NSArray<NJSponsorBlockSegment *> *segments =
        [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] allSegments];
    [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] clearAllSegments];
    [self.delegate submissionController:self didRemoveUnsubmittedSegments:segments];
}

// ── Server submission ────────────────────────────────────────────────────

- (void)submitSegmentsForCurrentVideoWithCompletion:(nullable NJSponsorBlockSubmitCompletion)completion {
    id<NJSponsorBlockSubmissionDelegate> ctx = self.delegate;

    if (ctx.videoID.length == 0 || ctx.cid <= 0) {
        if (completion) {
            completion(NO, [self submissionErrorWithCode:-1 message:@"尚未识别当前视频"]);
        }
        return;
    }

    NSArray<NJSponsorBlockSegment *> *segments = [self segmentsForCurrentVideo];
    if (segments.count == 0) {
        if (completion) {
            completion(NO, [self submissionErrorWithCode:-4 message:@"当前视频没有未提交片段"]);
        }
        return;
    }

    NSTimeInterval duration = ctx.estimatedVideoDuration;
    if (duration <= 0 || isnan(duration) || isinf(duration)) {
        if (completion) {
            completion(NO, [self submissionErrorWithCode:-2 message:@"暂未获取视频时长，稍后再试"]);
        }
        return;
    }

    [self submitSegments:segments
                 videoID:ctx.videoID
                     cid:ctx.cid
                duration:duration
                   index:0
              completion:completion];
}

- (void)submitSegments:(NSArray<NJSponsorBlockSegment *> *)segments
               videoID:(NSString *)videoID
                   cid:(NSInteger)cid
              duration:(NSTimeInterval)duration
                 index:(NSUInteger)index
            completion:(nullable NJSponsorBlockSubmitCompletion)completion {
    if (index >= segments.count) {
        [[NJSponsorBlockUnsubmittedSegmentStore sharedStore] removeSegmentsForVideoID:videoID cid:cid];
        [self.delegate submissionController:self
            didCompleteServerSubmissionForVideoID:videoID
                                              cid:cid
                                         segments:segments];
        if (completion) {
            completion(YES, nil);
        }
        return;
    }

    NJSponsorBlockSegment *segment = segments[index];
    __weak typeof(self) weakSelf = self;
    [self submitStoredSegment:segment videoID:videoID cid:cid duration:duration completion:^(BOOL success, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) {
                if (completion) {
                    NSError *staleError = [NSError errorWithDomain:@"NJSponsorBlockSubmissionController"
                                                              code:-5
                                                          userInfo:@{NSLocalizedDescriptionKey: @"提交状态已失效"}];
                    completion(NO, staleError);
                }
                return;
            }
            if (!success) {
                if (completion) completion(NO, error);
                return;
            }
            [strongSelf submitSegments:segments
                               videoID:videoID
                                   cid:cid
                              duration:duration
                                 index:index + 1
                            completion:completion];
        });
    }];
}

- (void)submitStoredSegment:(NJSponsorBlockSegment *)segment
                    videoID:(NSString *)videoID
                        cid:(NSInteger)cid
                   duration:(NSTimeInterval)duration
                 completion:(void (^)(BOOL success, NSError *_Nullable error))completion {
    NSArray<NSNumber *> *values = [segment.actionType isEqualToString:@"poi"]
        ? @[@(segment.startTime)]
        : @[@(segment.startTime), @(segment.endTime)];
    [self.service submitSegmentWithVideoID:videoID
                                       cid:cid
                                  category:segment.category
                                actionType:segment.actionType
                                   segment:values
                             videoDuration:duration
                                completion:completion];
}

- (void)submitCurrentVideoSegmentsShowingToasts {
    if (self.submissionInFlight) return;
    self.submissionInFlight = YES;
    [self.delegate submissionControllerDidChangeState:self];

    BBPlayerContext *playerContext = self.delegate.playerContext;
    [playerContext.toastWidgetService showToastContainerWithText:@"正在提交, 请稍候"];

    __weak typeof(self) weakSelf = self;
    [self submitSegmentsForCurrentVideoWithCompletion:^(BOOL success, NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf) return;

            strongSelf.submissionInFlight = NO;
            [strongSelf.delegate submissionControllerDidChangeState:strongSelf];

            BBPlayerContext *pc = strongSelf.delegate.playerContext;
            NSString *title  = success ? @"提交成功" : @"提交失败";
            NSString *detail = success ? @"当前视频草稿已提交" : (error.localizedDescription ?: @"请稍后重试");
            id resultToast = NJSponsorBlockCreateHintToast(pc, title, detail, nil, nil, nil, nil, nil, 3.0, NO);
            if (resultToast) [pc.toastWidgetService presentCustomToast:resultToast];
        });
    }];
}

// ── Info toast ───────────────────────────────────────────────────────────

- (void)showInfoToast:(NSString *)title detail:(NSString *)detail {
    BBPlayerContext *playerContext = self.delegate.playerContext;
    if (!playerContext) return;
    id toast = NJSponsorBlockCreateHintToast(playerContext, title, detail, nil, nil, nil, nil, nil, 3.0, NO);
    if (toast) [playerContext.toastWidgetService presentCustomToast:toast];
}

// ── Private helpers ──────────────────────────────────────────────────────

- (NSError *)submissionErrorWithCode:(NSInteger)code message:(NSString *)message {
    return [NSError errorWithDomain:@"NJSponsorBlockSubmissionController"
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: message ?: @"SponsorBlock submission failed"}];
}

@end
