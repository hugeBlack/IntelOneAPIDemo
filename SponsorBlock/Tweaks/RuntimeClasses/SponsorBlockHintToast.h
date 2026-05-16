//
//  SponsorBlockHintToast.h
//  SponsorBlock
//

#pragma once
#import "../Tweaks.h"

/**
 * Factory function – create and configure a SponsorBlockHintToast instance.
 *
 * @param context         Player context
 * @param title           Bold title text
 * @param detail          Secondary detail text
 * @param actionTitle     Primary button label (nil = hidden)
 * @param secondaryTitle  Secondary button label (nil = hidden)
 * @param actionHandler   Called when primary button is tapped (nil = no-op)
 * @param secondaryHandler Called when secondary button is tapped (nil = no-op)
 * @param closeHandler    Called when × button is tapped (nil = no-op); always dismisses the toast
 * @param duration        Auto-dismiss seconds; 0 or very large = effectively indefinite
 * @param showCloseButton Whether to show the × close button
 * @return A configured BBPlayerToastWidget subclass instance ready for presentCustomToast:
 */
id NJSponsorBlockCreateHintToast(
    BBPlayerContext *context,
    NSString *title,
    NSString *detail,
    NSString * _Nullable actionTitle,
    NSString * _Nullable secondaryTitle,
    void (^ _Nullable actionHandler)(void),
    void (^ _Nullable secondaryHandler)(void),
    void (^ _Nullable closeHandler)(void),
    NSTimeInterval duration,
    BOOL showCloseButton
);
