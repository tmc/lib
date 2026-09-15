// metalcount: count Metal dispatches, command buffers, commits and waits in any
// process, without modifying it. Injected with DYLD_INSERT_LIBRARIES, it swizzles
// the concrete driver classes behind the MTL* protocols and writes a JSON tally at
// exit. Counts are deterministic integers, so unlike throughput they need no quiet
// host.
#import <Metal/Metal.h>
#import <objc/runtime.h>
#import <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static atomic_llong nDispatchThreadgroups, nDispatchThreads, nIndirect;
static atomic_llong nCommandBuffer, nCommit, nWait, nComputeEncoder, nBlitEncoder;

#define HOOK(counter, ret, name, args, call)                                   \
    static ret (*orig_##name) args;                                            \
    static ret hook_##name args {                                              \
        atomic_fetch_add(&counter, 1);                                         \
        return orig_##name call;                                               \
    }

HOOK(nDispatchThreadgroups, void, dtg,
     (id s, SEL c, MTLSize g, MTLSize t), (s, c, g, t))
HOOK(nDispatchThreads, void, dt,
     (id s, SEL c, MTLSize g, MTLSize t), (s, c, g, t))
HOOK(nIndirect, void, dtgi,
     (id s, SEL c, id<MTLBuffer> b, NSUInteger o, MTLSize t), (s, c, b, o, t))
static void report(void);
static void (*orig_commit)(id, SEL);
static void hook_commit(id self, SEL cmd) {
    atomic_fetch_add(&nCommit, 1); orig_commit(self, cmd); report();
}
static void (*orig_wait)(id, SEL);
static void hook_wait(id self, SEL cmd) {
    atomic_fetch_add(&nWait, 1); orig_wait(self, cmd); report();
}

static id (*orig_commandBuffer)(id, SEL);
static id hook_commandBuffer(id self, SEL cmd) {
    atomic_fetch_add(&nCommandBuffer, 1);
    return orig_commandBuffer(self, cmd);
}
static id (*orig_computeEncoder)(id, SEL);
static id hook_computeEncoder(id self, SEL cmd) {
    atomic_fetch_add(&nComputeEncoder, 1);
    return orig_computeEncoder(self, cmd);
}
static id (*orig_blitEncoder)(id, SEL);
static id hook_blitEncoder(id self, SEL cmd) {
    atomic_fetch_add(&nBlitEncoder, 1);
    return orig_blitEncoder(self, cmd);
}

// swizzle replaces cls's implementation of sel, saving the original. Returns 0
// when the selector is absent so a missing hook is visible rather than silent.
static int swizzle(Class cls, SEL sel, IMP repl, void *saved) {
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) return 0;
    *(IMP *)saved = method_getImplementation(m);
    method_setImplementation(m, repl);
    return 1;
}

static int installed[8];

static void report(void) {
    if (!getenv("METALCOUNT_OUT")) return;
    const char *path = getenv("METALCOUNT_OUT");
    FILE *f = path ? fopen(path, "w") : stderr;
    if (!f) f = stderr;
    fprintf(f,
        "{\n  \"dispatch_threadgroups\": %lld,\n  \"dispatch_threads\": %lld,\n"
        "  \"dispatch_indirect\": %lld,\n  \"dispatches_total\": %lld,\n"
        "  \"command_buffers\": %lld,\n  \"commits\": %lld,\n  \"waits\": %lld,\n"
        "  \"compute_encoders\": %lld,\n  \"blit_encoders\": %lld,\n"
        "  \"hooks_installed\": [%d,%d,%d,%d,%d,%d,%d,%d]\n}\n",
        (long long)nDispatchThreadgroups, (long long)nDispatchThreads,
        (long long)nIndirect,
        (long long)(nDispatchThreadgroups + nDispatchThreads + nIndirect),
        (long long)nCommandBuffer, (long long)nCommit, (long long)nWait,
        (long long)nComputeEncoder, (long long)nBlitEncoder,
        installed[0], installed[1], installed[2], installed[3],
        installed[4], installed[5], installed[6], installed[7]);
    if (f != stderr) fclose(f);
}

__attribute__((constructor)) static void setup(void) {
    @autoreleasepool {
        // The MTL* types are protocols; the classes to patch only exist once the
        // driver has produced real objects, so make one of each.
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "metalcount: no Metal device\n"); return; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        installed[0] = swizzle(object_getClass(enc),
            @selector(dispatchThreadgroups:threadsPerThreadgroup:),
            (IMP)hook_dtg, &orig_dtg);
        installed[1] = swizzle(object_getClass(enc),
            @selector(dispatchThreads:threadsPerThreadgroup:),
            (IMP)hook_dt, &orig_dt);
        installed[2] = swizzle(object_getClass(enc),
            @selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:),
            (IMP)hook_dtgi, &orig_dtgi);
        installed[3] = swizzle(object_getClass(q),
            @selector(commandBuffer), (IMP)hook_commandBuffer, &orig_commandBuffer);
        installed[4] = swizzle(object_getClass(cb),
            @selector(commit), (IMP)hook_commit, &orig_commit);
        installed[5] = swizzle(object_getClass(cb),
            @selector(waitUntilCompleted), (IMP)hook_wait, &orig_wait);
        installed[6] = swizzle(object_getClass(cb),
            @selector(computeCommandEncoder), (IMP)hook_computeEncoder, &orig_computeEncoder);
        installed[7] = swizzle(object_getClass(cb),
            @selector(blitCommandEncoder), (IMP)hook_blitEncoder, &orig_blitEncoder);

        [enc endEncoding];
        // The probe objects were created before the hooks existed; do not count them.
        atomic_store(&nCommandBuffer, 0);
        atomic_store(&nComputeEncoder, 0);
    }
    atexit(report);
}
