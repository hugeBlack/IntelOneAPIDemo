//
//  NJSponsorBlockPlaybackHook.mm
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
extern "C"
{
#include "../Tweaks/Tweaks.h"
#include "../Settings/NJCommonDefine.h"
#include "../Settings/NJSettingCache.h"
#include "../Services/NJSponsorBlockManager.h"
#include "../UI/NJSponsorBlockPanelView.h"
#include "../Models/NJSponsorBlockSegment.h"
}

@interface IJKFFMoviePlayerControllerFFPlay: NSObject
@property (nonatomic) NSTimeInterval currentPlaybackTime;
-(void)prepareToPlay;
@end

static __weak IJKFFMoviePlayerControllerFFPlay* NJSponsorBlockCurrentIJKPlayer;
static NSTimer *NJSponsorBlockPlaybackPollTimer;
static NSTimeInterval NJSponsorBlockLastPlaybackPosition = -1;
static void NJSponsorBlockCaptureIJKPlayer(id player);
static void NJSponsorBlockInstallRuntimeHooks(void);
static BOOL NJSponsorBlockSeekCurrentPlayerToTime(NSTimeInterval time);
static void NJSponsorBlockHandlePlaybackTime(NSTimeInterval position);
static void NJSponsorBlockStartPlaybackPolling(void);
static void NJSponsorBlockStopPlaybackPolling(void);


static void (*NJSBOrigIJKPrepareToPlay)(id self, SEL _cmd);
static void NJSBHookIJKPrepareToPlay(id self, SEL _cmd) {
    NJSponsorBlockCaptureIJKPlayer(self);
    if (NJSBOrigIJKPrepareToPlay) {
        NJSBOrigIJKPrepareToPlay(self, _cmd);
    }
}

static void NJSponsorBlockCaptureIJKPlayer(id player) {
    if (!player) {
        return;
    }
    if (NJSponsorBlockCurrentIJKPlayer == player) {
        return;
    }
    NJSponsorBlockCurrentIJKPlayer = player;
    NJSponsorBlockStartPlaybackPolling();
    NSLog(@"[NJSponsorBlock] captured IJK player: %@", player);
}

static BOOL NJSponsorBlockReadPlaybackTime(IJKFFMoviePlayerControllerFFPlay* player, NSTimeInterval *time) {
    if (!player || !time) {
        return NO;
    }
    
    *time = player.currentPlaybackTime;
    return YES;
}

static void NJSponsorBlockStartPlaybackPolling(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (NJSponsorBlockPlaybackPollTimer) {
            return;
        }
        
        __block NSInteger failedReadCount = 0;
        NJSponsorBlockPlaybackPollTimer = [NSTimer scheduledTimerWithTimeInterval:0.25 repeats:YES block:^(__unused NSTimer *timer) {
            id player = NJSponsorBlockCurrentIJKPlayer;
            NSTimeInterval time = 0;
            if (!NJSponsorBlockReadPlaybackTime(player, &time)) {
                failedReadCount++;
                if (failedReadCount >= 12) {
                    NSLog(@"[NJSponsorBlock] playback polling stopped after stale player");
                    NJSponsorBlockStopPlaybackPolling();
                }
                return;
            }
            failedReadCount = 0;
            NJSponsorBlockHandlePlaybackTime(time);
        }];
        NSLog(@"[NJSponsorBlock] playback polling started");
    });
}

static void NJSponsorBlockStopPlaybackPolling(void) {
    [NJSponsorBlockPlaybackPollTimer invalidate];
    NJSponsorBlockPlaybackPollTimer = nil;
    NJSponsorBlockCurrentIJKPlayer = nil;
}

static BOOL NJSponsorBlockSeekCurrentPlayerToTime(NSTimeInterval time) {
    if (!NJSponsorBlockCurrentIJKPlayer) {
        NSLog(@"[NJSponsorBlock] skip requested but seek object is nil");
        return NO;
    }
    
    NJSponsorBlockCurrentIJKPlayer.currentPlaybackTime = time;
    return YES;
}

static void NJSponsorBlockHandlePlaybackTime(NSTimeInterval position) {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    BOOL movedBySeek = NJSponsorBlockLastPlaybackPosition >= 0 && fabs(position - NJSponsorBlockLastPlaybackPosition) > 2.0;
    NJSponsorBlockLastPlaybackPosition = position;

    [manager handlePlaybackTimeForProbe:position];

    if ([manager isInCooldown]) {
        return;
    }

    NSArray<NJSponsorBlockSegment *> *segments = [manager autoSkipSegmentsAtPlaybackTime:position];
    NJSponsorBlockSegment *segment = segments.lastObject;
    if (!segment) {
        return;
    }

    if (movedBySeek && ![manager skipOnSeekToSegment]) {
        NSLog(@"[NJSponsorBlock] ignore seek into segment %@ %.2f-%.2f", segment.uuid, segment.startTime, segment.endTime);
        for (NJSponsorBlockSegment *skippedSegment in segments) {
            [manager markSegmentSkipped:skippedSegment];
        }
        return;
    }

    NSTimeInterval targetTime = segment.endTime;
    if (segment.videoDuration > 0 && targetTime > segment.videoDuration - 2.0) {
        NSLog(@"[NJSponsorBlock] ignore segment near video end %@ %.2f-%.2f", segment.uuid, segment.startTime, segment.endTime);
        for (NJSponsorBlockSegment *skippedSegment in segments) {
            [manager markSegmentSkipped:skippedSegment];
        }
        return;
    }

    if (NJSponsorBlockSeekCurrentPlayerToTime(targetTime)) {
        for (NJSponsorBlockSegment *skippedSegment in segments) {
            [manager markSegmentSkipped:skippedSegment];
            [manager reportSegmentSkipped:skippedSegment];
            [manager recordLastSkippedSegment:skippedSegment];
        }
        [manager enterCooldown];
        NSLog(@"[NJSponsorBlock] skipped %@ %.2f-%.2f target=%.2f", segment.uuid, segment.startTime, segment.endTime, targetTime);
    }
}

static void NJSponsorBlockHandleManualSkipRequest(NSNotification *notification) {
    NJSponsorBlockSegment *segment = [notification.object isKindOfClass:[NJSponsorBlockSegment class]] ? notification.object : nil;
    if (!segment) {
        return;
    }

    NSTimeInterval targetTime = [segment.actionType isEqualToString:@"poi"] ? segment.startTime : segment.endTime;
    if (NJSponsorBlockSeekCurrentPlayerToTime(targetTime)) {
        NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
        [manager markSegmentSkipped:segment];
        [manager reportSegmentSkipped:segment];
        [manager recordLastSkippedSegment:segment];
        [manager enterCooldown];
        NSLog(@"[NJSponsorBlock] manually skipped %@ %.2f-%.2f target=%.2f", segment.uuid, segment.startTime, segment.endTime, targetTime);
    }
}

static void NJSponsorBlockHandleSeekRequest(NSNotification *notification) {
    NSNumber *timeNumber = [notification.object isKindOfClass:[NSNumber class]] ? notification.object : nil;
    if (!timeNumber) {
        return;
    }

    NSTimeInterval targetTime = timeNumber.doubleValue;
    if (targetTime < 0 || isnan(targetTime) || isinf(targetTime)) {
        return;
    }

    if (NJSponsorBlockSeekCurrentPlayerToTime(targetTime)) {
        [[NJSponsorBlockManager sharedInstance] enterCooldown];
        NSLog(@"[NJSponsorBlock] seek requested target=%.2f", targetTime);
    }
}

static void NJSponsorBlockInstallRuntimeHooks(void) {
    if (!NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    
    Class ijkClass = objc_getClass("IJKFFMoviePlayerControllerFFPlay");
    if (ijkClass) {
        JRSwizzleInstanceMethod(ijkClass,
                                   NSSelectorFromString(@"prepareToPlay"),
                                   (IMP)NJSBHookIJKPrepareToPlay,
                                   (IMP *)&NJSBOrigIJKPrepareToPlay);
    } else if (!ijkClass) {
        NSLog(@"[NJSponsorBlock] IJKFFMoviePlayerControllerFFPlay not found yet");
    }
}

void NJSponsorBlockPlaybackHookInit(void) {
    if (!NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    
    NSLog(@"[NJSponsorBlock] bootstrap");
    NJSponsorBlockInstallRuntimeHooks();
    [[NSNotificationCenter defaultCenter] addObserverForName:NJSponsorBlockManualSkipRequestNotification
                                                      object:nil
                                                       queue:[NSOperationQueue mainQueue]
                                                  usingBlock:^(NSNotification *note) {
        NJSponsorBlockHandleManualSkipRequest(note);
    }];
    [[NSNotificationCenter defaultCenter] addObserverForName:NJSponsorBlockSeekRequestNotification
                                                      object:nil
                                                       queue:[NSOperationQueue mainQueue]
                                                  usingBlock:^(NSNotification *note) {
        NJSponsorBlockHandleSeekRequest(note);
    }];
}
