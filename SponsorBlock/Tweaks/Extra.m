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

@interface BAPIAdV1SourceContentDto : GPBMessage
@property (nonatomic) BOOL hasAdContent;
@end

id (*orig_BAPIAdV1SourceContentDto_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAdV1SourceContentDto_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    BAPIAdV1SourceContentDto* ret = orig_BAPIAdV1SourceContentDto_initWithData_extensionRegistry_error(self, sel, nil, registry, error);
    return ret;
}

void initExtraHooks(void) {
    // fix app not displayed in NowPlaying center due to PhotosUI switching AVAudioSession's category to AVAudioSessionCategoryAmbient
    swizzle(AVAudioSession.class, @selector(setCategory:mode:routeSharingPolicy:options:error:), @selector(hook_setCategory:mode:routeSharingPolicy:options:error:));
    
    // remove iPad playlist ad
    JRSwizzleInstanceMethod(objc_getClass("BAPIAdV1SourceContentDto"), @selector(initWithData:extensionRegistry:error:),
                            (IMP)hook_BAPIAdV1SourceContentDto_initWithData_extensionRegistry_error,
                            (IMP*)&orig_BAPIAdV1SourceContentDto_initWithData_extensionRegistry_error);
}
