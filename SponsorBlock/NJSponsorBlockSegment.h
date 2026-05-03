//
//  NJSponsorBlockSegment.h
//  BiliBiliMDDylib
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NJSponsorBlockSegment : NSObject <NSSecureCoding>

@property (nonatomic, assign) NSTimeInterval startTime;
@property (nonatomic, assign) NSTimeInterval endTime;
@property (nonatomic, copy) NSString *uuid;
@property (nonatomic, copy) NSString *videoID;
@property (nonatomic, copy) NSString *category;
@property (nonatomic, copy) NSString *actionType;
@property (nonatomic, assign) NSInteger cid;
@property (nonatomic, assign) NSTimeInterval videoDuration;

+ (nullable instancetype)segmentWithDictionary:(NSDictionary *)dictionary;
- (BOOL)containsPlaybackTime:(NSTimeInterval)time;

@end

NS_ASSUME_NONNULL_END
