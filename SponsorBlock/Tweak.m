//
//  Tweak.m
//  SponsorBlock
//
//  Created by s s on 2026/5/2.
//
@import UIKit;
@import ObjectiveC;
#include "Tweaks/Tweaks.h"
#include "Settings/NJCommonDefine.h"

void registerOpenPanelButtonWidget(void);
void registerSponsorBlockPanelWidget(void);
void registerSponsorBlockHintToast(void);

__attribute__((constructor)) void TweakInit(void) {
    NSLog(@"SposorBlock loaded.");
    
    initSettingsHooks();
    
    if(!NJ_SPONSOR_BLOCK_VALUE) {
        NSLog(@"SposorBlock is disabled.");
        return;
    }
    
    registerOpenPanelButtonWidget();
    registerSponsorBlockPanelWidget();
    registerSponsorBlockHintToast();
    
    initSeekbarHooks();
    initPlayerWidgetButtonHooks();
    initViewReplyHooks();
    initThumbnailBadgeHooks();
    initPlayerContextHooks();
    initPlayerPlaybackHooks();
    
    initExtraHooks();
}
