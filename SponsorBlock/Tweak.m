//
//  Tweak.m
//  nmsl
//
//  Created by s s on 2026/5/2.
//
@import UIKit;
@import ObjectiveC;
#include "Tweaks/Tweaks.h"

void registerOpenPanelButtonWidget(void);
void registerSponsorBlockPanelWidget(void);
void registerSponsorBlockHintToast(void);

__attribute__((constructor)) void TweakInit(void) {
    NSLog(@"SposorBlock loaded.");
    
    registerOpenPanelButtonWidget();
    registerSponsorBlockPanelWidget();
    registerSponsorBlockHintToast();
    
    initSettingsHooks();
    initSeekbarHooks();
    initPlayerWidgetButtonHooks();
    initViewReplyHooks();
    initThumbnailBadgeHooks();
    initPlayerContextHooks();
    initPlayerPlaybackHooks();
}
