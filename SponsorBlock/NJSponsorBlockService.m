//
//  NJSponsorBlockService.m
//  BiliBiliMDDylib
//

#import "NJSponsorBlockService.h"
#import "NJSponsorBlockSegment.h"

static NSString * const NJSponsorBlockAPIBaseURLString = @"https://bsbsb.top/api/skipSegments";

@implementation NJSponsorBlockService

- (void)fetchSegmentsWithVideoID:(NSString *)videoID
                             cid:(NSInteger)cid
                      categories:(NSArray<NSString *> *)categories
                      completion:(NJSponsorBlockSegmentsCompletion)completion {
    if (videoID.length == 0 || cid <= 0) {
        if (completion) {
            completion(@[], nil);
        }
        return;
    }
    
    NSURL *url = [self requestURLWithVideoID:videoID cid:cid categories:categories];
    if (!url) {
        if (completion) {
            NSError *error = [NSError errorWithDomain:@"NJSponsorBlockService"
                                                 code:-1
                                             userInfo:@{NSLocalizedDescriptionKey: @"Invalid SponsorBlock URL"}];
            completion(@[], error);
        }
        return;
    }
    
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    request.HTTPMethod = @"GET";
    request.timeoutInterval = 8;
    [request setValue:@"BiliBiliMApp/1.0" forHTTPHeaderField:@"User-Agent"];
    [request setValue:@"BiliBiliMApp" forHTTPHeaderField:@"Origin"];
    [request setValue:@"BiliBiliMApp" forHTTPHeaderField:@"x-ext-name"];
    [request setValue:@"1.0" forHTTPHeaderField:@"x-ext-version"];
    
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            if (completion) {
                completion(@[], error);
            }
            return;
        }
        
        if (data.length == 0) {
            if (completion) {
                completion(@[], nil);
            }
            return;
        }
        
        NSError *jsonError = nil;
        id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError) {
            if (completion) {
                completion(@[], jsonError);
            }
            return;
        }
        
        NSArray *items = [json isKindOfClass:[NSArray class]] ? json : @[];
        NSMutableArray<NJSponsorBlockSegment *> *segments = [NSMutableArray array];
        for (NSDictionary *item in items) {
            NJSponsorBlockSegment *segment = [NJSponsorBlockSegment segmentWithDictionary:item];
            if (!segment) {
                continue;
            }
            if (segment.actionType.length > 0 && ![segment.actionType isEqualToString:@"skip"]) {
                continue;
            }
            [segments addObject:segment];
        }
        
        [segments sortUsingComparator:^NSComparisonResult(NJSponsorBlockSegment *obj1, NJSponsorBlockSegment *obj2) {
            if (obj1.startTime < obj2.startTime) {
                return NSOrderedAscending;
            }
            if (obj1.startTime > obj2.startTime) {
                return NSOrderedDescending;
            }
            return NSOrderedSame;
        }];
        
        if (completion) {
            completion([segments copy], nil);
        }
    }];
    [task resume];
}

- (NSURL *)requestURLWithVideoID:(NSString *)videoID cid:(NSInteger)cid categories:(NSArray<NSString *> *)categories {
    NSArray<NSString *> *effectiveCategories = categories.count > 0 ? categories : @[@"sponsor"];
    NSData *categoryData = [NSJSONSerialization dataWithJSONObject:effectiveCategories options:0 error:nil];
    NSString *categoryString = [[NSString alloc] initWithData:categoryData encoding:NSUTF8StringEncoding] ?: @"[\"sponsor\"]";
    
    NSURLComponents *components = [NSURLComponents componentsWithString:NJSponsorBlockAPIBaseURLString];
    components.queryItems = @[
        [NSURLQueryItem queryItemWithName:@"videoID" value:videoID],
        [NSURLQueryItem queryItemWithName:@"cid" value:[NSString stringWithFormat:@"%ld", (long)cid]],
        [NSURLQueryItem queryItemWithName:@"categories" value:categoryString],
    ];
    return components.URL;
}

@end
