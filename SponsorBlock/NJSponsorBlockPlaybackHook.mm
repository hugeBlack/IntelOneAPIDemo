//
//  NJSponsorBlockPlaybackHook.mm
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <substrate.h>
#import "NJCommonDefine.h"
#import "NJSponsorBlockManager.h"
#import "NJSponsorBlockPanelView.h"
#import "NJSponsorBlockSegment.h"

static __weak id NJSponsorBlockCurrentIJKPlayer;
static __weak id NJSponsorBlockCurrentSeekObject;
static NSTimer *NJSponsorBlockPlaybackPollTimer;
static BOOL NJSponsorBlockClockHooksInstalled;
static BOOL NJSponsorBlockPlaybackHookInstalled;
static BOOL NJSponsorBlockIJKHooksInstalled;
static BOOL NJSponsorBlockPlayerItemHookInstalled;
static void NJSponsorBlockCaptureIJKPlayer(id player);
static void NJSponsorBlockCaptureSeekObject(id object);
static void NJSponsorBlockCaptureIJKPlayerFromObject(id object);
static void NJSponsorBlockInstallOverlayIfNeeded(void);
static void NJSponsorBlockInstallRuntimeHooks(void);
static void NJSponsorBlockScheduleInstallRetries(void);
static BOOL NJSponsorBlockSeekCurrentPlayerToTime(NSTimeInterval time);
static void NJSponsorBlockHandlePlaybackTime(NSTimeInterval position);
static void NJSponsorBlockStartPlaybackPolling(void);
static void NJSponsorBlockStopPlaybackPolling(void);

static void (*NJSBOrigUpdateClockCurPosition)(id self, SEL _cmd, void *clock, double position);
static void NJSponsorBlockLogRuntimeProbe(id object);

static void NJSBHookUpdateClockCurPosition(id self, SEL _cmd, void *clock, double position) {
    if (NJSBOrigUpdateClockCurPosition) {
        NJSBOrigUpdateClockCurPosition(self, _cmd, clock, position);
    }
    NJSponsorBlockCaptureIJKPlayerFromObject(self);
    NJSponsorBlockCaptureSeekObject(self);
    NJSponsorBlockInstallOverlayIfNeeded();
    NJSponsorBlockLogRuntimeProbe(self);
    NJSponsorBlockHandlePlaybackTime(position);
}

static void (*NJSBOrigUpdateClockCurPositionTimestamp)(id self, SEL _cmd, void *clock, double position, double timestamp);
static void NJSBHookUpdateClockCurPositionTimestamp(id self, SEL _cmd, void *clock, double position, double timestamp) {
    if (NJSBOrigUpdateClockCurPositionTimestamp) {
        NJSBOrigUpdateClockCurPositionTimestamp(self, _cmd, clock, position, timestamp);
    }
    NJSponsorBlockCaptureIJKPlayerFromObject(self);
    NJSponsorBlockCaptureSeekObject(self);
    NJSponsorBlockInstallOverlayIfNeeded();
    NJSponsorBlockLogRuntimeProbe(self);
    NJSponsorBlockHandlePlaybackTime(position);
}

static void (*NJSBOrigPlaybackClockChanged)(id self, SEL _cmd, void *clock, double position, double timestamp);
static void NJSBHookPlaybackClockChanged(id self, SEL _cmd, void *clock, double position, double timestamp) {
    if (NJSBOrigPlaybackClockChanged) {
        NJSBOrigPlaybackClockChanged(self, _cmd, clock, position, timestamp);
    }
    NJSponsorBlockCaptureSeekObject(self);
    NJSponsorBlockCaptureIJKPlayerFromObject(self);
    NJSponsorBlockInstallOverlayIfNeeded();
    NJSponsorBlockHandlePlaybackTime(position);
}

static void (*NJSBOrigNetworkPlayerItemServiceSetMP)(id self, SEL _cmd, id player);
static void NJSBHookNetworkPlayerItemServiceSetMP(id self, SEL _cmd, id player) {
    NJSponsorBlockCaptureIJKPlayer(player);
    NJSponsorBlockCaptureSeekObject(player);
    if (NJSBOrigNetworkPlayerItemServiceSetMP) {
        NJSBOrigNetworkPlayerItemServiceSetMP(self, _cmd, player);
    }
}

static void (*NJSBOrigIJKSetPlaybackRate)(id self, SEL _cmd, float rate);
static void NJSBHookIJKSetPlaybackRate(id self, SEL _cmd, float rate) {
    NJSponsorBlockCaptureIJKPlayer(self);
    if (NJSBOrigIJKSetPlaybackRate) {
        NJSBOrigIJKSetPlaybackRate(self, _cmd, rate);
    }
}

static void (*NJSBOrigIJKPlay)(id self, SEL _cmd);
static void NJSBHookIJKPlay(id self, SEL _cmd) {
    NJSponsorBlockCaptureIJKPlayer(self);
    if (NJSBOrigIJKPlay) {
        NJSBOrigIJKPlay(self, _cmd);
    }
}

static void (*NJSBOrigIJKPrepareToPlay)(id self, SEL _cmd);
static void NJSBHookIJKPrepareToPlay(id self, SEL _cmd) {
    NJSponsorBlockCaptureIJKPlayer(self);
    if (NJSBOrigIJKPrepareToPlay) {
        NJSBOrigIJKPrepareToPlay(self, _cmd);
    }
}

static void NJSponsorBlockHookSelector(Class cls, SEL selector, IMP replacement, IMP *original) {
    if (!cls || !selector || !replacement || !original) {
        return;
    }
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) {
        NSLog(@"[NJSponsorBlock] selector not found: %@ %@", NSStringFromClass(cls), NSStringFromSelector(selector));
        return;
    }
    const char *encoding = method_getTypeEncoding(method);
    NSLog(@"[NJSponsorBlock] hook encoding %@ %@: %s", NSStringFromClass(cls), NSStringFromSelector(selector), encoding ?: "");
    MSHookMessageEx(cls, selector, replacement, original);
    NSLog(@"[NJSponsorBlock] hooked: %@ %@", NSStringFromClass(cls), NSStringFromSelector(selector));
}

static BOOL NJSponsorBlockMethodArgumentIsFloatingPoint(Method method, unsigned int index) {
    char *type = method_copyArgumentType(method, index);
    if (!type) {
        return NO;
    }
    BOOL result = type[0] == @encode(double)[0] || type[0] == @encode(float)[0];
    free(type);
    return result;
}

static void NJSponsorBlockHookClockSelector(Class cls,
                                            SEL selector,
                                            IMP replacement,
                                            IMP *original,
                                            BOOL requiresTimestamp) {
    Method method = class_getInstanceMethod(cls, selector);
    if (!method) {
        NSLog(@"[NJSponsorBlock] selector not found: %@ %@", NSStringFromClass(cls), NSStringFromSelector(selector));
        return;
    }
    
    if (!NJSponsorBlockMethodArgumentIsFloatingPoint(method, 3) ||
        (requiresTimestamp && !NJSponsorBlockMethodArgumentIsFloatingPoint(method, 4))) {
        NSLog(@"[NJSponsorBlock] skip unsafe clock hook %@ %@: %s", NSStringFromClass(cls), NSStringFromSelector(selector), method_getTypeEncoding(method) ?: "");
        return;
    }
    
    NJSponsorBlockHookSelector(cls, selector, replacement, original);
}

static void NJSponsorBlockCaptureIJKPlayer(id player) {
    if (!player) {
        return;
    }
    if (NJSponsorBlockCurrentIJKPlayer == player) {
        return;
    }
    NJSponsorBlockCurrentIJKPlayer = player;
    NJSponsorBlockCaptureSeekObject(player);
    NJSponsorBlockStartPlaybackPolling();
    NSLog(@"[NJSponsorBlock] captured IJK player: %@", player);
}

static BOOL NJSponsorBlockObjectCanSeek(id object) {
    if (!object) {
        return NO;
    }
    return [object respondsToSelector:NSSelectorFromString(@"setCurrentPlaybackTime:accurate:")] ||
           [object respondsToSelector:NSSelectorFromString(@"setCurrentPlaybackTime:")] ||
           [object respondsToSelector:NSSelectorFromString(@"seekToTime:")] ||
           [object respondsToSelector:NSSelectorFromString(@"seekToTime:completionHandler:")];
}

static void NJSponsorBlockCaptureSeekObject(id object) {
    if (!NJSponsorBlockObjectCanSeek(object) || NJSponsorBlockCurrentSeekObject == object) {
        return;
    }
    NJSponsorBlockCurrentSeekObject = object;
    NSLog(@"[NJSponsorBlock] captured seek object: %@", object);
}

static BOOL NJSponsorBlockObjectIsIJKPlayer(id object) {
    if (!object) {
        return NO;
    }
    NSString *className = NSStringFromClass([object class]);
    return [className isEqualToString:@"IJKFFMoviePlayerControllerFFPlay"] ||
           [className containsString:@"IJKFFMoviePlayerController"];
}

static BOOL NJSponsorBlockShouldInspectObject(id object) {
    if (!object || NJSponsorBlockObjectIsIJKPlayer(object)) {
        return NO;
    }
    NSString *className = NSStringFromClass([object class]);
    return [className hasPrefix:@"BBPlayer"] ||
           [className hasPrefix:@"BFCPlayer"] ||
           [className hasPrefix:@"IJK"];
}

static id NJSponsorBlockSeekObjectInObject(id object, NSInteger depth) {
    if (NJSponsorBlockObjectCanSeek(object)) {
        return object;
    }
    if (depth <= 0 || !NJSponsorBlockShouldInspectObject(object)) {
        return nil;
    }
    
    Class cls = object_getClass(object);
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    id foundObject = nil;
    for (unsigned int i = 0; i < count && !foundObject; i++) {
        Ivar ivar = ivars[i];
        const char *type = ivar_getTypeEncoding(ivar);
        if (!type || type[0] != '@') {
            continue;
        }
        id value = object_getIvar(object, ivar);
        if (NJSponsorBlockObjectCanSeek(value)) {
            foundObject = value;
            break;
        }
        if (NJSponsorBlockShouldInspectObject(value)) {
            foundObject = NJSponsorBlockSeekObjectInObject(value, depth - 1);
        }
    }
    free(ivars);
    return foundObject;
}

static id NJSponsorBlockIJKPlayerInObject(id object, NSInteger depth) {
    if (NJSponsorBlockObjectIsIJKPlayer(object)) {
        return object;
    }
    if (depth <= 0 || !NJSponsorBlockShouldInspectObject(object)) {
        return nil;
    }
    
    Class cls = object_getClass(object);
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    id foundPlayer = nil;
    for (unsigned int i = 0; i < count && !foundPlayer; i++) {
        Ivar ivar = ivars[i];
        const char *type = ivar_getTypeEncoding(ivar);
        if (!type || type[0] != '@') {
            continue;
        }
        
        id value = object_getIvar(object, ivar);
        if (NJSponsorBlockObjectIsIJKPlayer(value)) {
            foundPlayer = value;
            break;
        }
        if (NJSponsorBlockShouldInspectObject(value)) {
            foundPlayer = NJSponsorBlockIJKPlayerInObject(value, depth - 1);
        }
    }
    free(ivars);
    return foundPlayer;
}

static void NJSponsorBlockCaptureIJKPlayerFromObject(id object) {
    if (NJSponsorBlockCurrentIJKPlayer) {
        return;
    }
    
    static CFAbsoluteTime lastScanTime = 0;
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (now - lastScanTime < 1.0) {
        return;
    }
    lastScanTime = now;
    
    id player = NJSponsorBlockIJKPlayerInObject(object, 2);
    if (player) {
        NJSponsorBlockCaptureIJKPlayer(player);
    }
    id seekObject = NJSponsorBlockSeekObjectInObject(object, 2);
    if (seekObject) {
        NJSponsorBlockCaptureSeekObject(seekObject);
    }
}

static void NJSponsorBlockInstallOverlayIfNeeded(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [NJSponsorBlockPanelView markPlaybackActive];
        [NJSponsorBlockPanelView refresh];
    });
}

static BOOL NJSponsorBlockInvokeTimeSelector(id player, SEL selector, NSTimeInterval time, BOOL allowsAccurateArgument) {
    if (!player || ![player respondsToSelector:selector]) {
        return NO;
    }
    
    NSMethodSignature *signature = [player methodSignatureForSelector:selector];
    if (!signature || signature.numberOfArguments < 3) {
        return NO;
    }
    
    NSInvocation *invocation = [NSInvocation invocationWithMethodSignature:signature];
    invocation.target = player;
    invocation.selector = selector;
    
    const char *timeType = [signature getArgumentTypeAtIndex:2];
    if (timeType && timeType[0] == @encode(float)[0]) {
        float targetTime = (float)time;
        [invocation setArgument:&targetTime atIndex:2];
    } else if (timeType && timeType[0] == @encode(double)[0]) {
        double targetTime = time;
        [invocation setArgument:&targetTime atIndex:2];
    } else {
        NSLog(@"[NJSponsorBlock] skip incompatible seek selector %@ %@ arg=%s",
              NSStringFromClass([player class]),
              NSStringFromSelector(selector),
              timeType ?: "");
        return NO;
    }
    
    if (allowsAccurateArgument && signature.numberOfArguments > 3) {
        BOOL accurate = YES;
        [invocation setArgument:&accurate atIndex:3];
    }
    [invocation invoke];
    return YES;
}

static BOOL NJSponsorBlockReadPlaybackTime(id player, NSTimeInterval *time) {
    if (!player || !time) {
        return NO;
    }
    
    SEL selector = NSSelectorFromString(@"currentPlaybackTime");
    if (!player || ![player respondsToSelector:selector]) {
        return NO;
    }
    
    NSMethodSignature *signature = [player methodSignatureForSelector:selector];
    if (!signature) {
        return NO;
    }
    
    NSInvocation *invocation = [NSInvocation invocationWithMethodSignature:signature];
    invocation.target = player;
    invocation.selector = selector;
    [invocation invoke];
    
    const char *returnType = signature.methodReturnType;
    if (returnType && returnType[0] == @encode(float)[0]) {
        float value = 0;
        [invocation getReturnValue:&value];
        *time = value;
        return YES;
    }
    if (returnType && returnType[0] == @encode(double)[0]) {
        double value = 0;
        [invocation getReturnValue:&value];
        *time = value;
        return YES;
    }
    return NO;
}

static void NJSponsorBlockStartPlaybackPolling(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (NJSponsorBlockPlaybackPollTimer) {
            return;
        }
        
        __block NSInteger failedReadCount = 0;
        NJSponsorBlockPlaybackPollTimer = [NSTimer scheduledTimerWithTimeInterval:0.25 repeats:YES block:^(__unused NSTimer *timer) {
            id player = NJSponsorBlockCurrentIJKPlayer ?: NJSponsorBlockCurrentSeekObject;
            NSTimeInterval time = 0;
            if (!NJSponsorBlockReadPlaybackTime(player, &time)) {
                failedReadCount++;
                if (failedReadCount >= 12) {
                    NSLog(@"[NJSponsorBlock] playback polling stopped after stale player");
                    NJSponsorBlockStopPlaybackPolling();
                    [NJSponsorBlockPanelView hideOverlay];
                }
                return;
            }
            failedReadCount = 0;
            NJSponsorBlockInstallOverlayIfNeeded();
            NJSponsorBlockHandlePlaybackTime(time);
        }];
        NSLog(@"[NJSponsorBlock] playback polling started");
    });
}

static void NJSponsorBlockStopPlaybackPolling(void) {
    [NJSponsorBlockPlaybackPollTimer invalidate];
    NJSponsorBlockPlaybackPollTimer = nil;
    NJSponsorBlockCurrentIJKPlayer = nil;
    NJSponsorBlockCurrentSeekObject = nil;
}

static BOOL NJSponsorBlockSeekObjectToTime(id player, NSTimeInterval time) {
    if (!player) {
        return NO;
    }
    
    SEL accurateSelector = NSSelectorFromString(@"setCurrentPlaybackTime:accurate:");
    if (NJSponsorBlockInvokeTimeSelector(player, accurateSelector, time, YES)) {
        NSLog(@"[NJSponsorBlock] seek %@ with %@ to %.2f",
              NSStringFromClass([player class]),
              NSStringFromSelector(accurateSelector),
              time);
        return YES;
    }
    
    SEL normalSelector = NSSelectorFromString(@"setCurrentPlaybackTime:");
    if (NJSponsorBlockInvokeTimeSelector(player, normalSelector, time, NO)) {
        NSLog(@"[NJSponsorBlock] seek %@ with %@ to %.2f",
              NSStringFromClass([player class]),
              NSStringFromSelector(normalSelector),
              time);
        return YES;
    }
    
    SEL seekSelector = NSSelectorFromString(@"seekToTime:");
    if (NJSponsorBlockInvokeTimeSelector(player, seekSelector, time, NO)) {
        NSLog(@"[NJSponsorBlock] seek %@ with %@ to %.2f",
              NSStringFromClass([player class]),
              NSStringFromSelector(seekSelector),
              time);
        return YES;
    }
    
    return NO;
}

static BOOL NJSponsorBlockSeekCurrentPlayerToTime(NSTimeInterval time) {
    NSMutableArray *candidates = [NSMutableArray array];
    if (NJSponsorBlockCurrentSeekObject) {
        [candidates addObject:NJSponsorBlockCurrentSeekObject];
    }
    if (NJSponsorBlockCurrentIJKPlayer && NJSponsorBlockCurrentIJKPlayer != NJSponsorBlockCurrentSeekObject) {
        [candidates addObject:NJSponsorBlockCurrentIJKPlayer];
    }
    
    if (candidates.count == 0) {
        NSLog(@"[NJSponsorBlock] skip requested but seek object is nil");
        return NO;
    }
    
    for (id player in candidates) {
        if (NJSponsorBlockSeekObjectToTime(player, time)) {
            return YES;
        }
    }
    
    NSLog(@"[NJSponsorBlock] all seek candidates failed: %@", candidates);
    return NO;
}

static void NJSponsorBlockHandlePlaybackTime(NSTimeInterval position) {
    NJSponsorBlockManager *manager = [NJSponsorBlockManager sharedInstance];
    [manager handlePlaybackTimeForProbe:position];
    
    if ([manager isInCooldown]) {
        return;
    }
    
    NJSponsorBlockSegment *segment = [manager activeSegmentAtPlaybackTime:position];
    if (!segment || [manager hasSkippedSegment:segment]) {
        return;
    }
    
    NSTimeInterval targetTime = segment.endTime + 0.15;
    if (segment.videoDuration > 0 && targetTime > segment.videoDuration - 2.0) {
        NSLog(@"[NJSponsorBlock] ignore segment near video end %@ %.2f-%.2f", segment.uuid, segment.startTime, segment.endTime);
        [manager markSegmentSkipped:segment];
        return;
    }
    
    if (NJSponsorBlockSeekCurrentPlayerToTime(targetTime)) {
        [manager markSegmentSkipped:segment];
        [manager enterCooldown];
        NSLog(@"[NJSponsorBlock] skipped %@ %.2f-%.2f target=%.2f", segment.uuid, segment.startTime, segment.endTime, targetTime);
    }
}

static BOOL NJSponsorBlockSelectorLooksUseful(NSString *selectorName) {
    NSString *name = selectorName.lowercaseString;
    return [name containsString:@"seek"] ||
           [name containsString:@"time"] ||
           [name containsString:@"playback"] ||
           [name containsString:@"current"];
}

static void NJSponsorBlockLogMethodsForClassName(NSString *className) {
    Class cls = NSClassFromString(className);
    if (!cls) {
        NSLog(@"[NJSponsorBlock] class not found for method probe: %@", className);
        return;
    }
    
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (unsigned int i = 0; i < count; i++) {
        NSString *name = NSStringFromSelector(method_getName(methods[i]));
        if (NJSponsorBlockSelectorLooksUseful(name)) {
            [names addObject:name];
        }
    }
    free(methods);
    NSLog(@"[NJSponsorBlock] methods %@: %@", className, names);
}

static void NJSponsorBlockLogIvarsForObject(id object) {
    Class cls = object_getClass(object);
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    NSMutableArray<NSString *> *items = [NSMutableArray array];
    for (unsigned int i = 0; i < count; i++) {
        Ivar ivar = ivars[i];
        const char *name = ivar_getName(ivar);
        const char *type = ivar_getTypeEncoding(ivar);
        NSString *item = [NSString stringWithFormat:@"%s %s", name ?: "", type ?: ""];
        if (type && type[0] == '@') {
            id value = object_getIvar(object, ivar);
            item = [item stringByAppendingFormat:@" => %@", value ? NSStringFromClass([value class]) : @"nil"];
        }
        [items addObject:item];
    }
    free(ivars);
    NSLog(@"[NJSponsorBlock] ivars %@: %@", NSStringFromClass(cls), items);
}

static void NJSponsorBlockLogRuntimeProbe(id object) {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NJSponsorBlockLogIvarsForObject(object);
        NJSponsorBlockLogMethodsForClassName(@"BBPlayerClockService");
        NJSponsorBlockLogMethodsForClassName(@"BBPlayerPlayback");
        NJSponsorBlockLogMethodsForClassName(@"IJKFFMoviePlayerControllerFFPlay");
        NJSponsorBlockLogMethodsForClassName(@"BBPlayerInteractiveBizService");
        NJSponsorBlockLogMethodsForClassName(@"BBPlayerNetworkPlayerItemService");
    });
}

static void NJSponsorBlockInstallRuntimeHooks(void) {
    if (!NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    
    Class clockServiceClass = objc_getClass("BBPlayerClockService");
    if (clockServiceClass && !NJSponsorBlockClockHooksInstalled) {
        NJSponsorBlockHookClockSelector(clockServiceClass,
                                        NSSelectorFromString(@"updateClock:curPosition:"),
                                        (IMP)NJSBHookUpdateClockCurPosition,
                                        (IMP *)&NJSBOrigUpdateClockCurPosition,
                                        NO);
        
        NJSponsorBlockHookClockSelector(clockServiceClass,
                                        NSSelectorFromString(@"updateClock:curPosition:timestamp:"),
                                        (IMP)NJSBHookUpdateClockCurPositionTimestamp,
                                        (IMP *)&NJSBOrigUpdateClockCurPositionTimestamp,
                                        YES);
        NJSponsorBlockClockHooksInstalled = NJSBOrigUpdateClockCurPosition || NJSBOrigUpdateClockCurPositionTimestamp;
    } else if (!clockServiceClass) {
        NSLog(@"[NJSponsorBlock] BBPlayerClockService not found yet");
    }
    
    Class playbackClass = objc_getClass("BBPlayerPlayback");
    if (playbackClass && !NJSponsorBlockPlaybackHookInstalled) {
        NJSponsorBlockHookClockSelector(playbackClass,
                                        NSSelectorFromString(@"clockChanged:curPosition:timestamp:"),
                                        (IMP)NJSBHookPlaybackClockChanged,
                                        (IMP *)&NJSBOrigPlaybackClockChanged,
                                        YES);
        NJSponsorBlockPlaybackHookInstalled = NJSBOrigPlaybackClockChanged != NULL;
    } else {
        if (!playbackClass) {
            NSLog(@"[NJSponsorBlock] BBPlayerPlayback not found yet");
        }
    }
    
    Class ijkClass = objc_getClass("IJKFFMoviePlayerControllerFFPlay");
    if (ijkClass && !NJSponsorBlockIJKHooksInstalled) {
        NJSponsorBlockHookSelector(ijkClass,
                                   NSSelectorFromString(@"setPlaybackRate:"),
                                   (IMP)NJSBHookIJKSetPlaybackRate,
                                   (IMP *)&NJSBOrigIJKSetPlaybackRate);
        NJSponsorBlockHookSelector(ijkClass,
                                   NSSelectorFromString(@"play"),
                                   (IMP)NJSBHookIJKPlay,
                                   (IMP *)&NJSBOrigIJKPlay);
        NJSponsorBlockHookSelector(ijkClass,
                                   NSSelectorFromString(@"prepareToPlay"),
                                   (IMP)NJSBHookIJKPrepareToPlay,
                                   (IMP *)&NJSBOrigIJKPrepareToPlay);
        NJSponsorBlockIJKHooksInstalled = NJSBOrigIJKSetPlaybackRate || NJSBOrigIJKPlay || NJSBOrigIJKPrepareToPlay;
    } else if (!ijkClass) {
        NSLog(@"[NJSponsorBlock] IJKFFMoviePlayerControllerFFPlay not found yet");
    }
    
    Class playerItemServiceClass = objc_getClass("BBPlayerNetworkPlayerItemService");
    if (playerItemServiceClass && !NJSponsorBlockPlayerItemHookInstalled) {
        NJSponsorBlockHookSelector(playerItemServiceClass,
                                   NSSelectorFromString(@"setMp:"),
                                   (IMP)NJSBHookNetworkPlayerItemServiceSetMP,
                                   (IMP *)&NJSBOrigNetworkPlayerItemServiceSetMP);
        NJSponsorBlockPlayerItemHookInstalled = NJSBOrigNetworkPlayerItemServiceSetMP != NULL;
    } else if (!playerItemServiceClass) {
        NSLog(@"[NJSponsorBlock] BBPlayerNetworkPlayerItemService not found yet");
    }
    
    // Overlay installation is driven by player layout and playback clock callbacks.
}

static void NJSponsorBlockScheduleInstallRetries(void) {
    for (NSInteger i = 0; i < 20; i++) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(i * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
            NJSponsorBlockInstallRuntimeHooks();
        });
    }
}

__attribute__((constructor)) static void NJSponsorBlockPlaybackHookInit(void) {
    if (!NJ_MASTER_SWITCH_VALUE) {
        return;
    }
    
    NSLog(@"[NJSponsorBlock] bootstrap");
    NJSponsorBlockInstallRuntimeHooks();
    NJSponsorBlockScheduleInstallRetries();
    [[NSNotificationCenter defaultCenter] addObserverForName:UIApplicationDidBecomeActiveNotification
                                                      object:nil
                                                       queue:[NSOperationQueue mainQueue]
                                                  usingBlock:^(__unused NSNotification *note) {
        NJSponsorBlockInstallRuntimeHooks();
    }];
}
