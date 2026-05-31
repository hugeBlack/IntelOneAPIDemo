//
//  Extra.m
//  SponsorBlock
//
//  Created by s s on 2026/5/23.
//
#include "Tweaks.h"
@import AVFoundation;

@implementation AVAudioSession(aaa)

-(BOOL)hook_setCategory:(AVAudioSessionCategory)category
                   mode:(AVAudioSessionMode)mode
     routeSharingPolicy:(AVAudioSessionRouteSharingPolicy)policy
                options:(AVAudioSessionCategoryOptions)options
                  error:(NSError**)error {
    if([category isEqualToString:AVAudioSessionCategoryAmbient]) {
        category = AVAudioSessionCategoryPlayback;
    }
    return [self hook_setCategory:category mode:mode routeSharingPolicy:policy options:options error:error];
}

@end

void initExtraHooks(void) {
    // fix app not displayed in NowPlaying center due to PhotosUI switching AVAudioSession's category to AVAudioSessionCategoryAmbient
    swizzle(AVAudioSession.class, @selector(setCategory:mode:routeSharingPolicy:options:error:), @selector(hook_setCategory:mode:routeSharingPolicy:options:error:));
}
