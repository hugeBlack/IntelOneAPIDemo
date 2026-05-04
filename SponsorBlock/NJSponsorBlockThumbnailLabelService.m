//
//  NJSponsorBlockThumbnailLabelService.m
//  SponsorBlock
//

#import "NJSponsorBlockThumbnailLabelService.h"
#import "NJSponsorBlockService.h"

static NSString * const NJSBThumbnailLabelNoLabel = @"noLabel";
static NSString * const NJSBThumbnailLabelError = @"error";
static NSString * const NJSBThumbnailLabelSponsor = @"sponsor";
static NSString * const NJSBThumbnailLabelExclusiveAccess = @"exclusive_access";

@interface NJSponsorBlockThumbnailLabelResult : NSObject

@property (nonatomic, copy) NSString *category;
@property (nonatomic, copy) NSString *text;

@end

@implementation NJSponsorBlockThumbnailLabelResult
@end

@interface NJSponsorBlockThumbnailLabelService ()

@property (nonatomic, strong) NSCache<NSString *, NJSponsorBlockThumbnailLabelResult *> *cache;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSMutableArray<NJSBThumbnailLabelCompletion> *> *inflightCompletions;
@property (nonatomic, strong) dispatch_queue_t stateQueue;

- (NJSponsorBlockThumbnailLabelResult *)noLabelResult;
- (NJSponsorBlockThumbnailLabelResult *)errorResult;

@end

@implementation NJSponsorBlockThumbnailLabelService

+ (instancetype)sharedService {
    static NJSponsorBlockThumbnailLabelService *service = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        service = [[NJSponsorBlockThumbnailLabelService alloc] init];
    });
    return service;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        self.cache = [[NSCache alloc] init];
        self.cache.countLimit = 500;
        self.inflightCompletions = [NSMutableDictionary dictionary];
        self.stateQueue = dispatch_queue_create("com.njsponsorblock.thumbnail-label", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (void)fetchLabelForBVID:(NSString *)bvid completion:(NJSBThumbnailLabelCompletion)completion {
    NSString *normalizedBVID = [bvid isKindOfClass:[NSString class]] ? [bvid copy] : @"";
    if (normalizedBVID.length == 0) {
        [self completeOnMainWithCompletion:completion result:[self noLabelResult]];
        return;
    }

    dispatch_async(self.stateQueue, ^{
        NJSponsorBlockThumbnailLabelResult *cached = [self.cache objectForKey:normalizedBVID];
        if (cached) {
            [self completeOnMainWithCompletion:completion result:cached];
            return;
        }

        NSMutableArray<NJSBThumbnailLabelCompletion> *completions = self.inflightCompletions[normalizedBVID];
        if (completions) {
            if (completion) {
                [completions addObject:[completion copy]];
            }
            return;
        }

        self.inflightCompletions[normalizedBVID] = completion ? [NSMutableArray arrayWithObject:[completion copy]] : [NSMutableArray array];
        [self startRequestForBVID:normalizedBVID];
    });
}

- (void)startRequestForBVID:(NSString *)bvid {
    NSString *prefix = [NJSponsorBlockService hashPrefixForVideoID:bvid];
    NSURL *url = [NJSponsorBlockService apiURLWithPath:@"videoLabels" hashPrefix:prefix];
    if (!url) {
        [self resolveBVID:bvid result:[self errorResult]];
        return;
    }

    NSMutableURLRequest *request = [NJSponsorBlockService sponsorBlockRequestWithURL:url method:@"GET" timeout:8];
    [request setValue:@"chrome-extension://eaoelafamejbnggahofapllmfhlhajdd" forHTTPHeaderField:@"Origin"];
    [request setValue:@"BiliBiliSponsorBlock" forHTTPHeaderField:@"x-ext-name"];
    [request setValue:@"0.5.0" forHTTPHeaderField:@"x-ext-version"];
    NSURLSessionDataTask *task = [[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error) {
            [self resolveBVID:bvid result:[self errorResult]];
            return;
        }

        NSHTTPURLResponse *httpResponse = [response isKindOfClass:[NSHTTPURLResponse class]] ? (NSHTTPURLResponse *)response : nil;
        if (httpResponse && httpResponse.statusCode == 404) {
            [self resolveBVID:bvid result:[self noLabelResult]];
            return;
        }
        if (httpResponse && (httpResponse.statusCode < 200 || httpResponse.statusCode >= 300)) {
            [self resolveBVID:bvid result:[self errorResult]];
            return;
        }

        if (data.length == 0) {
            [self resolveBVID:bvid result:[self errorResult]];
            return;
        }

        NSError *jsonError = nil;
        id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError];
        if (jsonError) {
            [self resolveBVID:bvid result:[self errorResult]];
            return;
        }
        if (![json isKindOfClass:[NSArray class]]) {
            [self resolveBVID:bvid result:[self errorResult]];
            return;
        }

        [self handleResponseArray:(NSArray *)json requestedBVID:bvid];
    }];
    [task resume];
}

- (void)handleResponseArray:(NSArray *)responseArray requestedBVID:(NSString *)requestedBVID {
    NSMutableDictionary<NSString *, NJSponsorBlockThumbnailLabelResult *> *results = [NSMutableDictionary dictionary];
    NSMutableSet<NSString *> *malformedVideoIDs = [NSMutableSet set];
    for (id item in responseArray) {
        if (![item isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary *dictionary = (NSDictionary *)item;
        id videoIDObject = dictionary[@"videoID"];
        NSString *videoID = [videoIDObject isKindOfClass:[NSString class]] ? videoIDObject : @"";
        if (videoID.length == 0) {
            continue;
        }

        id segmentsObject = dictionary[@"segments"];
        if (![segmentsObject isKindOfClass:[NSArray class]]) {
            [malformedVideoIDs addObject:videoID];
            continue;
        }

        BOOL malformedSegments = NO;
        NJSponsorBlockThumbnailLabelResult *result = [self resultForSegments:segmentsObject malformed:&malformedSegments];
        if (malformedSegments) {
            [malformedVideoIDs addObject:videoID];
            continue;
        }

        results[videoID] = result;
    }

    NJSponsorBlockThumbnailLabelResult *requestedResult = results[requestedBVID];
    dispatch_async(self.stateQueue, ^{
        [results enumerateKeysAndObjectsUsingBlock:^(NSString *videoID, NJSponsorBlockThumbnailLabelResult *result, BOOL *stop) {
            [self.cache setObject:result forKey:videoID];
        }];

        if (requestedResult) {
            [self finishBVID:requestedBVID result:requestedResult];
            return;
        }

        NJSponsorBlockThumbnailLabelResult *noLabelResult = [self noLabelResult];
        if ([malformedVideoIDs containsObject:requestedBVID]) {
            [self finishBVID:requestedBVID result:[self errorResult]];
            return;
        }

        [self.cache setObject:noLabelResult forKey:requestedBVID];
        [self finishBVID:requestedBVID result:noLabelResult];
    });
}

- (NJSponsorBlockThumbnailLabelResult *)resultForSegments:(NSArray *)segmentsObject malformed:(BOOL *)malformed {
    if (malformed) {
        *malformed = NO;
    }

    BOOL hasExclusiveAccess = NO;
    for (id segmentObject in segmentsObject) {
        if (![segmentObject isKindOfClass:[NSDictionary class]]) {
            if (malformed) {
                *malformed = YES;
            }
            return [self noLabelResult];
        }
        NSDictionary *segmentDictionary = (NSDictionary *)segmentObject;
        id categoryObject = segmentDictionary[@"category"];
        if (![categoryObject isKindOfClass:[NSString class]]) {
            if (malformed) {
                *malformed = YES;
            }
            return [self noLabelResult];
        }
        NSString *category = categoryObject;
        if ([category isEqualToString:NJSBThumbnailLabelSponsor]) {
            return [self resultWithCategory:NJSBThumbnailLabelSponsor text:@"推广"];
        }
        if ([category isEqualToString:NJSBThumbnailLabelExclusiveAccess]) {
            hasExclusiveAccess = YES;
        }
    }

    if (hasExclusiveAccess) {
        return [self resultWithCategory:NJSBThumbnailLabelExclusiveAccess text:@"独家"];
    }
    return [self noLabelResult];
}

- (void)resolveBVID:(NSString *)bvid result:(NJSponsorBlockThumbnailLabelResult *)result {
    dispatch_async(self.stateQueue, ^{
        if (![result.category isEqualToString:NJSBThumbnailLabelError]) {
            [self.cache setObject:result forKey:bvid];
        }
        [self finishBVID:bvid result:result];
    });
}

- (void)finishBVID:(NSString *)bvid result:(NJSponsorBlockThumbnailLabelResult *)result {
    NSMutableArray<NJSBThumbnailLabelCompletion> *completions = self.inflightCompletions[bvid];
    [self.inflightCompletions removeObjectForKey:bvid];

    for (NJSBThumbnailLabelCompletion completion in completions) {
        [self completeOnMainWithCompletion:completion result:result];
    }
}

- (void)completeOnMainWithCompletion:(NJSBThumbnailLabelCompletion)completion result:(NJSponsorBlockThumbnailLabelResult *)result {
    if (!completion) {
        return;
    }
    dispatch_async(dispatch_get_main_queue(), ^{
        if ([result.category isEqualToString:NJSBThumbnailLabelSponsor] ||
            [result.category isEqualToString:NJSBThumbnailLabelExclusiveAccess]) {
            completion(result.category, result.text);
            return;
        }
        completion(nil, nil);
    });
}

- (NJSponsorBlockThumbnailLabelResult *)resultWithCategory:(NSString *)category text:(NSString *)text {
    NJSponsorBlockThumbnailLabelResult *result = [[NJSponsorBlockThumbnailLabelResult alloc] init];
    result.category = category ?: NJSBThumbnailLabelNoLabel;
    result.text = text;
    return result;
}

- (NJSponsorBlockThumbnailLabelResult *)noLabelResult {
    return [self resultWithCategory:NJSBThumbnailLabelNoLabel text:nil];
}

- (NJSponsorBlockThumbnailLabelResult *)errorResult {
    return [self resultWithCategory:NJSBThumbnailLabelError text:nil];
}

@end
