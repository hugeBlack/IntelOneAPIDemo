//
//  Tweak.m
//  nmsl
//
//  Created by s s on 2026/5/2.
//
@import UIKit;
@import ObjectiveC;
#include "Tweaks/Tweaks.h"

__attribute__((constructor)) void TweakInit(void) {
    NSLog(@"SposorBlock loaded.");
    
    initSettingsHooks();
    initSeekbarHooks();
    initPlayerWidgetButtonHooks();
    initViewReplyHooks();
    initThumbnailBadgeHooks();
    initPlayerContextHooks();
    initPlayerPlaybackHooks();
}
