//
//  NJSponsorBlockUnsubmittedSegmentStore.m
//  SponsorBlock
//

#import "NJSponsorBlockUnsubmittedSegmentStore.h"
#import "NJSponsorBlockSegment.h"
#import "NJSettingCache.h"
#import <math.h>

static NSString * const NJSponsorBlockUnsubmittedSegmentsKey = @"NJSponsorBlockUnsubmittedSegmentsKey";

@implementation NJSponsorBlockUnsubmittedSegmentStore

+ (instancetype)sharedStore {
    static NJSponsorBlockUnsubmittedSegmentStore *store = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        store = [[NJSponsorBlockUnsubmittedSegmentStore alloc] init];
    });
    return store;
}

- (NSArray<NSString *> *)videoKeys {
    return [[self storedSegmentsByVideoKey].allKeys sortedArrayUsingSelector:@selector(localizedStandardCompare:)];
}

- (NSArray<NJSponsorBlockSegment *> *)segmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid {
    return [self segmentsForVideoKey:[self keyForVideoID:videoID cid:cid]];
}

- (NSArray<NJSponsorBlockSegment *> *)segmentsForVideoKey:(NSString *)videoKey {
    if (videoKey.length == 0) {
        return @[];
    }
    return [self storedSegmentsByVideoKey][videoKey] ?: @[];
}

- (NSArray<NJSponsorBlockSegment *> *)allSegments {
    NSMutableArray<NJSponsorBlockSegment *> *segments = [NSMutableArray array];
    for (NSArray<NJSponsorBlockSegment *> *videoSegments in [self storedSegmentsByVideoKey].allValues) {
        [segments addObjectsFromArray:videoSegments];
    }
    return [segments copy];
}

- (NJSponsorBlockSegment *)addSegmentForVideoID:(NSString *)videoID
                                            cid:(NSInteger)cid
                                       category:(NSString *)category
                                     actionType:(NSString *)actionType
                                        segment:(NSArray<NSNumber *> *)segment
                                  videoDuration:(NSTimeInterval)videoDuration {
    NSString *key = [self keyForVideoID:videoID cid:cid];
    if (key.length == 0 || segment.count == 0) {
        return nil;
    }

    NSNumber *startNumber = segment.firstObject;
    NSNumber *endNumber = segment.count > 1 ? segment[1] : segment.firstObject;
    if (![startNumber respondsToSelector:@selector(doubleValue)] || ![endNumber respondsToSelector:@selector(doubleValue)]) {
        return nil;
    }

    NSTimeInterval start = startNumber.doubleValue;
    NSTimeInterval end = endNumber.doubleValue;
    if (![actionType isEqualToString:@"poi"] && end <= start) {
        return nil;
    }

    NSMutableDictionary<NSString *, NSArray<NJSponsorBlockSegment *> *> *stored = [[self storedSegmentsByVideoKey] mutableCopy];
    NSMutableArray<NJSponsorBlockSegment *> *segments = [stored[key] mutableCopy] ?: [NSMutableArray array];
    for (NJSponsorBlockSegment *existing in segments) {
        if ([existing.category isEqualToString:category ?: @""] &&
            [existing.actionType isEqualToString:actionType ?: @""] &&
            fabs(existing.startTime - start) < 0.001 &&
            fabs(existing.endTime - end) < 0.001) {
            return existing;
        }
    }

    NJSponsorBlockSegment *localSegment = [[NJSponsorBlockSegment alloc] init];
    localSegment.startTime = start;
    localSegment.endTime = end;
    localSegment.videoID = videoID ?: @"";
    localSegment.cid = cid;
    localSegment.category = category ?: @"";
    localSegment.actionType = actionType ?: @"skip";
    localSegment.videoDuration = videoDuration;
    localSegment.unsubmitted = YES;
    localSegment.uuid = [NSString stringWithFormat:@"local:%@:%ld:%.3f:%.3f:%@", localSegment.videoID, (long)cid, start, end, localSegment.category];

    [segments addObject:localSegment];
    stored[key] = [segments copy];
    [self saveSegmentsByVideoKey:stored];
    return localSegment;
}

- (BOOL)updateSegmentWithUUID:(NSString *)uuid
                      videoID:(NSString *)videoID
                          cid:(NSInteger)cid
                    startTime:(NSTimeInterval)startTime
                      endTime:(NSTimeInterval)endTime {
    NSString *key = [self keyForVideoID:videoID cid:cid];
    if (key.length == 0 || uuid.length == 0) {
        return NO;
    }

    NSMutableDictionary<NSString *, NSArray<NJSponsorBlockSegment *> *> *stored = [[self storedSegmentsByVideoKey] mutableCopy];
    NSMutableArray<NJSponsorBlockSegment *> *segments = [stored[key] mutableCopy];
    if (segments.count == 0) {
        return NO;
    }
    for (NJSponsorBlockSegment *segment in segments) {
        if (![segment.uuid isEqualToString:uuid]) {
            continue;
        }
        if (![segment.actionType isEqualToString:@"poi"] && endTime <= startTime) {
            return NO;
        }
        segment.startTime = startTime;
        segment.endTime = endTime;
        [self saveSegmentsByVideoKey:stored];
        return YES;
    }
    return NO;
}

- (BOOL)removeSegmentWithUUID:(NSString *)uuid videoID:(NSString *)videoID cid:(NSInteger)cid {
    NSString *key = [self keyForVideoID:videoID cid:cid];
    if (key.length == 0 || uuid.length == 0) {
        return NO;
    }

    NSMutableDictionary<NSString *, NSArray<NJSponsorBlockSegment *> *> *stored = [[self storedSegmentsByVideoKey] mutableCopy];
    NSMutableArray<NJSponsorBlockSegment *> *segments = [stored[key] mutableCopy];
    if (segments.count == 0) {
        return NO;
    }
    NSUInteger index = [segments indexOfObjectPassingTest:^BOOL(NJSponsorBlockSegment *segment, __unused NSUInteger idx, __unused BOOL *stop) {
        return [segment.uuid isEqualToString:uuid];
    }];
    if (index == NSNotFound) {
        return NO;
    }

    [segments removeObjectAtIndex:index];
    if (segments.count > 0) {
        stored[key] = [segments copy];
    } else {
        [stored removeObjectForKey:key];
    }
    [self saveSegmentsByVideoKey:stored];
    return YES;
}

- (void)removeSegmentsForVideoID:(NSString *)videoID cid:(NSInteger)cid {
    NSString *key = [self keyForVideoID:videoID cid:cid];
    if (key.length == 0) {
        return;
    }
    NSMutableDictionary *stored = [[self storedSegmentsByVideoKey] mutableCopy];
    [stored removeObjectForKey:key];
    [self saveSegmentsByVideoKey:stored];
}

- (void)clearAllSegments {
    [NJ_SETTING_CACHE removeObjectForKey:NJSponsorBlockUnsubmittedSegmentsKey];
}

- (NSUInteger)totalSegmentCount {
    NSUInteger count = 0;
    for (NSArray<NJSponsorBlockSegment *> *segments in [self storedSegmentsByVideoKey].allValues) {
        count += segments.count;
    }
    return count;
}

- (NSUInteger)videoCount {
    return [self storedSegmentsByVideoKey].count;
}

- (NSDictionary<NSString *, NSArray<NJSponsorBlockSegment *> *> *)storedSegmentsByVideoKey {
    id object = [NJ_SETTING_CACHE objectForKey:NJSponsorBlockUnsubmittedSegmentsKey];
    if (![object isKindOfClass:[NSDictionary class]]) {
        return @{};
    }
    NSMutableDictionary<NSString *, NSArray<NJSponsorBlockSegment *> *> *result = [NSMutableDictionary dictionary];
    [(NSDictionary *)object enumerateKeysAndObjectsUsingBlock:^(id key, id value, __unused BOOL *stop) {
        if (![key isKindOfClass:[NSString class]] || ![value isKindOfClass:[NSArray class]]) {
            return;
        }
        NSMutableArray<NJSponsorBlockSegment *> *segments = [NSMutableArray array];
        for (id segment in (NSArray *)value) {
            if ([segment isKindOfClass:[NJSponsorBlockSegment class]]) {
                NJSponsorBlockSegment *localSegment = segment;
                localSegment.unsubmitted = YES;
                [segments addObject:localSegment];
            }
        }
        if (segments.count > 0) {
            result[key] = [segments copy];
        }
    }];
    return [result copy];
}

- (void)saveSegmentsByVideoKey:(NSDictionary<NSString *, NSArray<NJSponsorBlockSegment *> *> *)segmentsByVideoKey {
    [NJ_SETTING_CACHE setObject:segmentsByVideoKey ?: @{} forKey:NJSponsorBlockUnsubmittedSegmentsKey];
}

- (NSString *)keyForVideoID:(NSString *)videoID cid:(NSInteger)cid {
    if (videoID.length == 0 || cid <= 0) {
        return @"";
    }
    return [NSString stringWithFormat:@"%@:%ld", videoID, (long)cid];
}

@end
