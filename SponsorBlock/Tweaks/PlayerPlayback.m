//
//  PlayerPlayback.m
//  SponsorBlock
//
//  Created by s s on 2026/5/16.
//

#include "Tweaks.h"
#include "../Services/NJSponsorBlockManager.h"

void (*orig_BBPlayerPlayback_setCurrentItem)(id self, SEL sel, id newItem) = nil;
static void hook_BBPlayerPlayback_setCurrentItem(BBPlayerPlayback* self, SEL _cmd, id newItem) {
    id currentItem = [self currentItem];
    orig_BBPlayerPlayback_setCurrentItem(self, _cmd, newItem);
    // 有的时候会复用播放器，导致使用同一个BBPlayerContext，因此加入-[BBPlayerPlayback setCurrentItem:]的hook，在切换视频切复用播放器时可以及时更新manager的状态，
    
    BBPlayerContext* context = [self context];
    NJSponsorBlockManager* manager = objc_getAssociatedObject(context, sponsorBlockManagerKey);
    if(!manager) {
        manager = [[NJSponsorBlockManager alloc] initWithContext:[self context]];
        objc_setAssociatedObject([self context], sponsorBlockManagerKey, manager, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    } else if (currentItem) {
        [manager reset];
    }
    [manager startListeningForVideoInfoWithCID:[newItem cid]];
}


void initPlayerPlaybackHooks(void) {
    JRSwizzleInstanceMethod(PrivClass(BBPlayerPlayback),
                            @selector(setCurrentItem:),
                            (IMP)hook_BBPlayerPlayback_setCurrentItem,
                            (IMP*)&orig_BBPlayerPlayback_setCurrentItem);
}
