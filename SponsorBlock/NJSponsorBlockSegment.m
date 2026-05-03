//
//  NJSponsorBlockSegment.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockSegment.h"

@implementation NJSponsorBlockSegment

+ (BOOL)supportsSecureCoding {
    return YES;
}

+ (instancetype)segmentWithDictionary:(NSDictionary *)dictionary {
    if (![dictionary isKindOfClass:[NSDictionary class]]) {
        return nil;
    }
    
    NSArray *segmentArray = dictionary[@"segment"];
    if (![segmentArray isKindOfClass:[NSArray class]] || segmentArray.count < 1) {
        return nil;
    }

    NSNumber *start = segmentArray[0];
    NSNumber *end = segmentArray.count > 1 ? segmentArray[1] : start;
    if (![start respondsToSelector:@selector(doubleValue)] || ![end respondsToSelector:@selector(doubleValue)]) {
        return nil;
    }

    NJSponsorBlockSegment *segment = [[NJSponsorBlockSegment alloc] init];
    segment.startTime = start.doubleValue;
    segment.endTime = end.doubleValue;
    segment.uuid = [dictionary[@"UUID"] isKindOfClass:[NSString class]] ? dictionary[@"UUID"] : @"";
    segment.videoID = [dictionary[@"videoID"] isKindOfClass:[NSString class]] ? dictionary[@"videoID"] : @"";
    segment.category = [dictionary[@"category"] isKindOfClass:[NSString class]] ? dictionary[@"category"] : @"";
    segment.actionType = [dictionary[@"actionType"] isKindOfClass:[NSString class]] ? dictionary[@"actionType"] : @"";
    segment.cid = [dictionary[@"cid"] respondsToSelector:@selector(integerValue)] ? [dictionary[@"cid"] integerValue] : 0;
    segment.videoDuration = [dictionary[@"videoDuration"] respondsToSelector:@selector(doubleValue)] ? [dictionary[@"videoDuration"] doubleValue] : 0;
    
    if ((![segment.actionType isEqualToString:@"poi"] && segment.endTime <= segment.startTime) || segment.uuid.length == 0) {
        return nil;
    }
    return segment;
}

- (BOOL)containsPlaybackTime:(NSTimeInterval)time {
    return time >= self.startTime && time < self.endTime;
}

- (void)encodeWithCoder:(NSCoder *)coder {
    [coder encodeDouble:self.startTime forKey:@"startTime"];
    [coder encodeDouble:self.endTime forKey:@"endTime"];
    [coder encodeObject:self.uuid forKey:@"uuid"];
    [coder encodeObject:self.videoID forKey:@"videoID"];
    [coder encodeObject:self.category forKey:@"category"];
    [coder encodeObject:self.actionType forKey:@"actionType"];
    [coder encodeInteger:self.cid forKey:@"cid"];
    [coder encodeDouble:self.videoDuration forKey:@"videoDuration"];
}

- (instancetype)initWithCoder:(NSCoder *)coder {
    self = [super init];
    if (self) {
        self.startTime = [coder decodeDoubleForKey:@"startTime"];
        self.endTime = [coder decodeDoubleForKey:@"endTime"];
        self.uuid = [coder decodeObjectOfClass:[NSString class] forKey:@"uuid"] ?: @"";
        self.videoID = [coder decodeObjectOfClass:[NSString class] forKey:@"videoID"] ?: @"";
        self.category = [coder decodeObjectOfClass:[NSString class] forKey:@"category"] ?: @"";
        self.actionType = [coder decodeObjectOfClass:[NSString class] forKey:@"actionType"] ?: @"";
        self.cid = [coder decodeIntegerForKey:@"cid"];
        self.videoDuration = [coder decodeDoubleForKey:@"videoDuration"];
    }
    return self;
}

@end
