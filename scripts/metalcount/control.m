// Ground truth: exactly 7 dispatches, 3 command buffers, 3 commits, 2 waits,
// 3 compute encoders, 1 blit encoder.
#import <Metal/Metal.h>
int main(void) { @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:
        @"kernel void nop(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = 1.0f; }"
        options:nil error:&e];
    if (!lib) { NSLog(@"lib: %@", e); return 1; }
    id<MTLComputePipelineState> p =
        [d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nop"] error:&e];
    id<MTLBuffer> b = [d newBufferWithLength:1024 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [d newCommandQueue];
    int dispatches[3] = {4, 2, 1};   // 7 total across 3 command buffers
    for (int c = 0; c < 3; c++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:b offset:0 atIndex:0];
        for (int i = 0; i < dispatches[c]; i++)
            [enc dispatchThreadgroups:MTLSizeMake(1,1,1)
                threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];
        if (c == 2) { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
                      [bl endEncoding]; }
        [cb commit];
        if (c != 1) [cb waitUntilCompleted];   // 2 waits
    }
    return 0;
} }
