//
//  ViewReply.m
//  SponsorBlock
//
//  Created by s s on 2026/5/4.
//
#include "Tweaks.h"
#include "../Settings/NJSettingDefine.h"
#include "../Services/NJSponsorBlockManager.h"

NSMutableDictionary* cachedCidVideoInfoDict = nil;

@interface ViewReplyHelper : NSObject

@end

@implementation ViewReplyHelper

+ (instancetype)sharedInstance {
    static ViewReplyHelper *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[ViewReplyHelper alloc] init];
    });
    return instance;
}

- (NSDictionary *)videoIdentityInObject:(id)object {
    if ([object isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = (NSDictionary *)object;
        NSString *videoID = [self videoIDInDictionary:dictionary];
        NSNumber *cid = [self cidInDictionary:dictionary];
        if (videoID.length > 0 && cid.integerValue > 0) {
            return @{@"videoID": videoID, @"cid": cid};
        }
        
        for (id value in dictionary.allValues) {
            NSDictionary *identity = [self videoIdentityInObject:value];
            if (identity) {
                return identity;
            }
        }
        return nil;
    }
    
    if ([object isKindOfClass:[NSArray class]]) {
        for (id value in (NSArray *)object) {
            NSDictionary *identity = [self videoIdentityInObject:value];
            if (identity) {
                return identity;
            }
        }
    }
    return nil;
}

- (NSString *)videoIDInDictionary:(NSDictionary *)dictionary {
    for (NSString *key in dictionary) {
        if (![key isKindOfClass:[NSString class]]) {
            continue;
        }
        NSString *lowerKey = key.lowercaseString;
        if (![lowerKey isEqualToString:@"bvid"] &&
            ![lowerKey isEqualToString:@"bvidstr"] &&
            ![lowerKey isEqualToString:@"bvid_str"] &&
            ![lowerKey isEqualToString:@"bv_id"]) {
            continue;
        }
        id value = dictionary[key];
        if ([value isKindOfClass:[NSString class]] && [value hasPrefix:@"BV"]) {
            return value;
        }
    }
    return nil;
}

- (NSNumber *)cidInDictionary:(NSDictionary *)dictionary {
    for (NSString *key in dictionary) {
        if (![key isKindOfClass:[NSString class]] || ![key.lowercaseString isEqualToString:@"cid"]) {
            continue;
        }
        id value = dictionary[key];
        if ([value respondsToSelector:@selector(integerValue)] && [value integerValue] > 0) {
            return @([value integerValue]);
        }
    }
    return nil;
}

- (NSTimeInterval)videoDurationInObject:(id)object {
    if ([object isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = (NSDictionary *)object;
        NSTimeInterval duration = [self durationInDictionary:dictionary];
        if (duration > 0) {
            return duration;
        }
        for (id value in dictionary.allValues) {
            duration = [self videoDurationInObject:value];
            if (duration > 0) {
                return duration;
            }
        }
        return 0;
    }

    if ([object isKindOfClass:[NSArray class]]) {
        for (id value in (NSArray *)object) {
            NSTimeInterval duration = [self videoDurationInObject:value];
            if (duration > 0) {
                return duration;
            }
        }
    }
    return 0;
}

- (NSTimeInterval)durationInDictionary:(NSDictionary *)dictionary {
    for (NSString *key in dictionary) {
        if (![key isKindOfClass:[NSString class]]) {
            continue;
        }
        NSTimeInterval duration = [self durationFromKey:key value:dictionary[key]];
        if (duration > 0) {
            return duration;
        }
    }
    return 0;
}

- (NSTimeInterval)durationFromKey:(NSString *)key value:(id)value {
    if (![value respondsToSelector:@selector(doubleValue)]) {
        return 0;
    }
    NSString *lowerKey = key.lowercaseString;
    NSTimeInterval rawValue = [value doubleValue];
    if (rawValue <= 0 || !isfinite(rawValue)) {
        return 0;
    }
    if ([lowerKey isEqualToString:@"timelength"] ||
        [lowerKey isEqualToString:@"time_length"] ||
        [lowerKey isEqualToString:@"duration_ms"]) {
        return rawValue / 1000.0;
    }
    if ([lowerKey isEqualToString:@"duration"] ||
        [lowerKey isEqualToString:@"video_duration"] ||
        [lowerKey isEqualToString:@"videoduration"]) {
        return rawValue > 86400 ? rawValue / 1000.0 : rawValue;
    }
    return 0;
}

- (NSTimeInterval)durationFromCandidateAccessorsOfObject:(id)object {
    NSArray<NSString *> *selectors = @[@"duration", @"timelength", @"timeLength", @"videoDuration"];
    for (NSString *selectorName in selectors) {
        id value = [self safeValueForSelectorName:selectorName object:object];
        NSTimeInterval duration = [self durationFromKey:selectorName value:value];
        if (duration > 0) {
            return duration;
        }
    }
    return 0;
}

- (void)collectVideoIdentityFromObject:(id)object
                                 depth:(NSInteger)depth
                               visited:(NSMutableSet<NSValue *> *)visited
                               videoID:(NSString **)videoID
                                   cid:(NSNumber **)cid
                              duration:(NSTimeInterval *)duration {
    if (!object || depth > 5 || ((*videoID).length > 0 && (*cid).integerValue > 0 && *duration > 0)) {
        return;
    }
    
    if ([object isKindOfClass:[NSDictionary class]]) {
        NSDictionary *dictionary = (NSDictionary *)object;
        NSString *foundVideoID = [self videoIDInDictionary:dictionary];
        NSNumber *foundCID = [self cidInDictionary:dictionary];
        if ((*videoID).length == 0 && foundVideoID.length > 0) {
            *videoID = foundVideoID;
        }
        if ((*cid).integerValue <= 0 && foundCID.integerValue > 0) {
            *cid = foundCID;
        }
        if (*duration <= 0) {
            *duration = [self durationInDictionary:dictionary];
        }
        for (id value in dictionary.allValues) {
            [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
        }
        return;
    }
    
    if ([object isKindOfClass:[NSArray class]] || [object isKindOfClass:[NSSet class]]) {
        for (id value in object) {
            [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
        }
        return;
    }
    
    if ([object isKindOfClass:[NSString class]] ||
        [object isKindOfClass:[NSNumber class]] ||
        [object isKindOfClass:[NSData class]] ||
        [object isKindOfClass:[NSDate class]]) {
        return;
    }
    
    NSValue *pointer = [NSValue valueWithNonretainedObject:object];
    if ([visited containsObject:pointer]) {
        return;
    }
    [visited addObject:pointer];
    
    NSString *className = NSStringFromClass([object class]);
    if (![className hasPrefix:@"BAPI"] && ![className hasPrefix:@"BBPlayer"] && ![className hasPrefix:@"BFCPlayer"]) {
        return;
    }
    
    [self collectVideoIdentityFromCandidateAccessorsOfObject:object videoID:videoID cid:cid];
    if (*duration <= 0) {
        *duration = [self durationFromCandidateAccessorsOfObject:object];
    }

    unsigned int propertyCount = 0;
    objc_property_t *properties = class_copyPropertyList([object class], &propertyCount);
    for (unsigned int i = 0; i < propertyCount; i++) {
        const char *name = property_getName(properties[i]);
        if (!name) {
            continue;
        }
        id value = [self safeValueForKey:[NSString stringWithUTF8String:name] object:object];
        [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
    }
    free(properties);
    
    unsigned int ivarCount = 0;
    Ivar *ivars = class_copyIvarList([object class], &ivarCount);
    for (unsigned int i = 0; i < ivarCount; i++) {
        Ivar ivar = ivars[i];
        const char *type = ivar_getTypeEncoding(ivar);
        const char *name = ivar_getName(ivar);
        if (!type || type[0] != '@' || !name) {
            continue;
        }
        id value = object_getIvar(object, ivar);
        [self collectVideoIdentityFromObject:value depth:depth + 1 visited:visited videoID:videoID cid:cid duration:duration];
    }
    free(ivars);
}

- (void)collectVideoIdentityFromCandidateAccessorsOfObject:(id)object
                                                   videoID:(NSString **)videoID
                                                       cid:(NSNumber **)cid {
    NSArray<NSString *> *videoSelectors = @[@"bvid", @"bvidStr", @"bvidString", @"bvID", @"bvId"];
    for (NSString *selectorName in videoSelectors) {
        id value = [self safeValueForSelectorName:selectorName object:object];
        if ((*videoID).length == 0 && [value isKindOfClass:[NSString class]] && [value hasPrefix:@"BV"]) {
            *videoID = value;
        }
    }
    
    id cidValue = [self safeValueForSelectorName:@"cid" object:object];
    if ((*cid).integerValue <= 0 && [cidValue respondsToSelector:@selector(integerValue)] && [cidValue integerValue] > 0) {
        *cid = @([cidValue integerValue]);
    }
}

- (id)safeValueForSelectorName:(NSString *)selectorName object:(id)object {
    SEL selector = NSSelectorFromString(selectorName);
    if (![object respondsToSelector:selector]) {
        return nil;
    }
    
    NSMethodSignature *signature = [object methodSignatureForSelector:selector];
    if (!signature || signature.numberOfArguments != 2) {
        return nil;
    }
    
    return [self safeValueForKey:selectorName object:object];
}

- (id)safeValueForKey:(NSString *)key object:(id)object {
    @try {
        return [object valueForKey:key];
    } @catch (__unused NSException *exception) {
        if ([key hasPrefix:@"_"]) {
            return nil;
        }
        @try {
            return [object valueForKey:[@"_" stringByAppendingString:key]];
        } @catch (__unused NSException *innerException) {
            return nil;
        }
    }
}

- (void)inspectResponseData:(NSData *)data response:(NSURLResponse *)response {
    if (data.length == 0) {
        return;
    }
    
    if (![self shouldInspectResponse:response data:data]) {
        return;
    }
    
    NSError *error = nil;
    id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (error || !json) {
        return;
    }
    
    NSDictionary *identity = [self videoIdentityInObject:json];
    NSString *videoID = identity[@"videoID"];
    NSNumber *cid = identity[@"cid"];
    if (videoID.length == 0 || cid.integerValue <= 0) {
        return;
    }
    NSLog(@"[NJSponsorBlock] found video identity %@:%ld from %@", videoID, (long)cid.integerValue, response.URL.absoluteString);
    NSDictionary* userInfo = @{
        @"videoID": videoID,
        @"cid": cid,
        @"duration": @([self videoDurationInObject:json])
    };
    cachedCidVideoInfoDict[cid] = userInfo;
    [NSNotificationCenter.defaultCenter postNotificationName:NJSponsorBlockVideoInfoRetrievedNotification object:self];
}

- (void)inspectModelObject:(id)object source:(NSString *)source {
    if (!object) {
        return;
    }
    
    NSMutableSet<NSValue *> *visited = [NSMutableSet set];
    NSString *__block videoID = nil;
    NSNumber *__block cid = nil;
    NSTimeInterval __block duration = 0;
    [self collectVideoIdentityFromObject:object
                                   depth:0
                                 visited:visited
                                 videoID:&videoID
                                     cid:&cid
                                duration:&duration];
    if (videoID.length == 0 || cid.integerValue <= 0) {
        NSLog(@"[NJSponsorBlock] model identity not found from %@ %@", source ?: @"unknown", NSStringFromClass([object class]));
        return;
    }
    
    NSLog(@"[NJSponsorBlock] found video identity %@:%ld from model %@ %@",
          videoID,
          (long)cid.integerValue,
          source ?: @"unknown",
          NSStringFromClass([object class]));

    NSDictionary* userInfo = @{
        @"videoID": videoID,
        @"cid": cid,
        @"duration": @(duration)
    };
    cachedCidVideoInfoDict[cid] = userInfo;
    [NSNotificationCenter.defaultCenter postNotificationName:NJSponsorBlockVideoInfoRetrievedNotification object:self];
}

- (BOOL)shouldInspectResponse:(NSURLResponse *)response data:(NSData *)data {
    NSString *url = response.URL.absoluteString.lowercaseString ?: @"";
    NSString *mimeType = response.MIMEType.lowercaseString ?: @"";
    if ([url containsString:@"view"] ||
        [url containsString:@"detail"] ||
        [url containsString:@"player"] ||
        [url containsString:@"playurl"] ||
        [url containsString:@"archive"]) {
        return YES;
    }
    
    if (![mimeType containsString:@"json"] || data.length > 2 * 1024 * 1024) {
        return NO;
    }
    
    NSData *bvidData = [@"bvid" dataUsingEncoding:NSUTF8StringEncoding];
    NSData *cidData = [@"cid" dataUsingEncoding:NSUTF8StringEncoding];
    NSRange range = NSMakeRange(0, data.length);
    return [data rangeOfData:bvidData options:0 range:range].location != NSNotFound &&
           [data rangeOfData:cidData options:0 range:range].location != NSNotFound;
}

@end


id (*orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    id ret = orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error(self, sel, data, registry, error);
    [[ViewReplyHelper sharedInstance] inspectModelObject:ret source:@"BAPIAppViewuniteV1ViewReply"];
    return ret;
}

id (*orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error)(id self, SEL, id data, id registry, id* error) = nil;
id hook_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error(id self, SEL sel, id data, id registry, id* error) {
    id ret = orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error(self, sel, data, registry, error);
    [[ViewReplyHelper sharedInstance] inspectModelObject:ret source:@"BAPIAppViewV1ViewReply"];
    return ret;
}

//@interface BBVDPlayerFlowConfig: NSObject
//@property NSInteger avid;
//@property (nonatomic, copy) NSString *bvid;
//@property NSInteger cid;
//@end
//
//void (*orig_BBVDPlayerVC_play)(id self, SEL sel, BBVDPlayerFlowConfig* config) = nil;
//void hook_BBVDPlayerVC_play(id self, SEL sel, BBVDPlayerFlowConfig* config) {
//    orig_BBVDPlayerVC_play(self, sel, config);
//
//    [[NJSponsorBlockManager sharedInstance] updateVideoID:config.bvid cid:config.cid];
//    [[NJSponsorBlockManager sharedInstance] updateNativeVideoDuration:123];
//}

void initViewReplyHooks(void) {
    cachedCidVideoInfoDict = [NSMutableDictionary new];
    //    JRSwizzleInstanceMethod(objc_getClass("BBVDPlayerVC"), @selector(play:),
    //                            (IMP)hook_BBVDPlayerVC_play,
    //                            (IMP*)&orig_BBVDPlayerVC_play);
        JRSwizzleInstanceMethod(objc_getClass("BAPIAppViewuniteV1ViewReply"), @selector(initWithData:extensionRegistry:error:),
                                (IMP)hook_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error,
                                (IMP*)&orig_BAPIAppViewuniteV1ViewReply_initWithData_extensionRegistry_error);
        
        JRSwizzleInstanceMethod(objc_getClass("BAPIAppViewV1ViewReply"), @selector(initWithData:extensionRegistry:error:),
                                (IMP)hook_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error,
                                (IMP*)&orig_BAPIAppViewV1ViewReply_initWithData_extensionRegistry_error);
}
