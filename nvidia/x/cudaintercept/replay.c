/* Off-hook finite replay owner. No function here is called by the hook. */
#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#include "replay.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <float.h>

#if (defined(CUDAINTERCEPT_PRIVATE_FIXTURE) || defined(CUDAINTERCEPT_CPU_DRIVER)) &&               \
    defined(CUDAINTERCEPT_REAL_HEADERS)
#error private CPU authority must never be used in a real-header/native build
#endif

#ifdef CUDAINTERCEPT_REAL_HEADERS
#include <cuda.h>
typedef CUresult result_t;
typedef CUcontext context_t;
typedef CUmodule module_t;
typedef CUfunction function_t;
typedef CUstream stream_t;
typedef CUdevice device_t;
typedef CUdevice_attribute attribute_t;
#define CAPABILITY_MAJOR CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define CAPABILITY_MINOR CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
_Static_assert(CAPABILITY_MAJOR == 75 && CAPABILITY_MINOR == 76,
               "unqualified host attribute constants must match CUDA headers");
typedef CUdeviceptr ptr_t;
typedef CUuuid replay_uuid_t;
typedef CUstreamCaptureStatus capture_t;
#else
/* Unqualified LP64 host declarations, tested against real headers separately.
 * They do not grant production admission. No vendor header is vendored. */
typedef int result_t, device_t, capture_t, attribute_t;
/* CUDA device attribute values, source-bound by the real-header assertions above. */
#define CAPABILITY_MAJOR 75
#define CAPABILITY_MINOR 76
typedef void *context_t, *module_t, *function_t, *stream_t;
typedef uint64_t ptr_t;
typedef struct {
    unsigned char bytes[16];
} replay_uuid_t;
#endif

_Static_assert(sizeof(void *) == 8 && sizeof(ptr_t) == 8 && sizeof(unsigned) == 4 &&
                   sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "owned replay requires LP64/FP32");

/* Private bounded SHA256; input snapshots never exceed IMAGE_LIMIT. */
struct hash {
    uint32_t h[8];
    uint64_t n;
    unsigned char b[64];
    size_t used;
};
static uint32_t rotr(uint32_t v, unsigned n) { return (v >> n) | (v << (32 - n)); }
static void hash_block(struct hash *s, const unsigned char *b) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++)
        w[i] = (uint32_t)b[4 * i] << 24 | (uint32_t)b[4 * i + 1] << 16 |
               (uint32_t)b[4 * i + 2] << 8 | b[4 * i + 3];
    for (unsigned i = 16; i < 64; i++) {
        uint32_t x = w[i - 15], y = w[i - 2];
        w[i] = w[i - 16] + (rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)) + w[i - 7] +
               (rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10));
    }
    uint32_t a = s->h[0], c = s->h[2], b0 = s->h[1], d = s->h[3], e = s->h[4], f = s->h[5],
             g = s->h[6], h = s->h[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t t =
            h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
        uint32_t u = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b0) ^ (a & c) ^ (b0 & c));
        h = g;
        g = f;
        f = e;
        e = d + t;
        d = c;
        c = b0;
        b0 = a;
        a = t + u;
    }
    s->h[0] += a;
    s->h[1] += b0;
    s->h[2] += c;
    s->h[3] += d;
    s->h[4] += e;
    s->h[5] += f;
    s->h[6] += g;
    s->h[7] += h;
}
static void hash_add(struct hash *s, const void *data, size_t n) {
    const unsigned char *p = data;
    s->n += n;
    while (n) {
        size_t k = 64 - s->used;
        if (k > n)
            k = n;
        memcpy(s->b + s->used, p, k);
        s->used += k;
        p += k;
        n -= k;
        if (s->used == 64) {
            hash_block(s, s->b);
            s->used = 0;
        }
    }
}
static struct hash hash_new(void) {
    return (struct hash){{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
                          0x1f83d9ab, 0x5be0cd19},
                         0,
                         {0},
                         0};
}
static void hash_finish(struct hash *s, char out[65]) {
    uint64_t bits = s->n * 8;
    unsigned char one = 0x80, zero = 0;
    hash_add(s, &one, 1);
    while (s->used != 56)
        hash_add(s, &zero, 1);
    unsigned char tail[8];
    for (unsigned i = 0; i < 8; i++)
        tail[7 - i] = (unsigned char)(bits >> (8 * i));
    hash_add(s, tail, 8);
    static const char hex[] = "0123456789abcdef";
    for (unsigned i = 0; i < 32; i++) {
        unsigned char v = (unsigned char)(s->h[i / 4] >> (24 - 8 * (i % 4)));
        out[2 * i] = hex[v >> 4];
        out[2 * i + 1] = hex[v & 15];
    }
    out[64] = 0;
}
static void digest(const void *p, size_t n, char out[65]) {
    struct hash s = hash_new();
    hash_add(&s, p, n);
    hash_finish(&s, out);
}

struct driver_api {
    result_t (*current)(context_t *);
    result_t (*device)(device_t *);
    result_t (*version)(int *);
    result_t (*attribute)(int *, attribute_t, device_t);
    result_t (*uuid)(replay_uuid_t *, device_t);
    result_t (*retain)(context_t *, device_t);
    result_t (*release)(device_t);
    result_t (*capture)(stream_t, capture_t *);
    result_t (*stream_context)(stream_t, context_t *);
    result_t (*load)(module_t *, const void *);
    result_t (*function)(function_t *, module_t, const char *);
    result_t (*unload)(module_t);
    result_t (*alloc)(ptr_t *, size_t);
    result_t (*free)(ptr_t);
    result_t (*write)(ptr_t, const void *, size_t);
    result_t (*read)(void *, ptr_t, size_t);
    result_t (*sync)(stream_t);
    result_t (*launch)(function_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                       unsigned, stream_t, void **, void **);
};
#ifdef CUDAINTERCEPT_REAL_HEADERS
#define API(field, symbol)                                                                         \
    _Static_assert(__builtin_types_compatible_p(__typeof__(((struct driver_api *)0)->field),       \
                                                __typeof__(&symbol)),                              \
                   #symbol)
API(current, cuCtxGetCurrent);
API(device, cuCtxGetDevice);
API(version, cuDriverGetVersion);
API(attribute, cuDeviceGetAttribute);
API(uuid, cuDeviceGetUuid);
API(retain, cuDevicePrimaryCtxRetain);
API(release, cuDevicePrimaryCtxRelease);
API(capture, cuStreamIsCapturing);
API(load, cuModuleLoadData);
API(function, cuModuleGetFunction);
API(stream_context, cuStreamGetCtx);
API(unload, cuModuleUnload);
API(alloc, cuMemAlloc);
API(free, cuMemFree);
API(write, cuMemcpyHtoD);
API(read, cuMemcpyDtoH);
API(sync, cuStreamSynchronize);
API(launch, cuLaunchKernel);
#endif
static int resolve_api(struct driver_api *a) {
#define GET(field, symbol)                                                                         \
    do {                                                                                           \
        void *p = dlsym(RTLD_NEXT, symbol);                                                        \
        _Static_assert(sizeof a->field == sizeof p, "CUDA function pointer representation");       \
        memcpy(&a->field, &p, sizeof p);                                                           \
        if (!a->field)                                                                             \
            return 0;                                                                              \
    } while (0)
    GET(current, "cuCtxGetCurrent");
    GET(device, "cuCtxGetDevice");
    GET(version, "cuDriverGetVersion");
    GET(attribute, "cuDeviceGetAttribute");
    GET(uuid, "cuDeviceGetUuid_v2");
    GET(retain, "cuDevicePrimaryCtxRetain");
    GET(release, "cuDevicePrimaryCtxRelease_v2");
    GET(capture, "cuStreamIsCapturing");
    GET(stream_context, "cuStreamGetCtx");
    GET(load, "cuModuleLoadData");
    GET(function, "cuModuleGetFunction");
    GET(unload, "cuModuleUnload");
    GET(alloc, "cuMemAlloc_v2");
    GET(free, "cuMemFree_v2");
    GET(write, "cuMemcpyHtoD_v2");
    GET(read, "cuMemcpyDtoH_v2");
    GET(sync, "cuStreamSynchronize");
    GET(launch, "cuLaunchKernel");
#undef GET
    return 1;
}

/* This row's domain is private CPU behavior ONLY. No runtime/env override.
 * Production count=0. A future source-reviewed admission must fill every field
 * from an authentic independently reviewed current packet, not historical PASS. */
struct admission {
    char baseline[65], candidate[65], qualifier[65], adapter[65], qualification[65];
    char caller[64];
    int driver, major, minor;
    unsigned char uuid[16];
    char driver_image[65];
};
#ifdef CUDAINTERCEPT_PRIVATE_FIXTURE
static const struct admission admission_table[] = {
    {"67b65d21f2ea02b1a4d6d4a3b5be9def524ce00644438a12e314802abe8e27d3",
     "77762339d3c566647c8fc1ee8cc43c4626b4e35013a19646b3e2ed23871dfcc1",
     "d89688e1c69ad5808828a1e53193c4f20287731670c93603d39b7c416ac7ad15",
     "9c249f861690fe913bcd3d19c031340276591a5787719185a81df051df1aabcb",
     "2997f3e7e8429c34a9ddaaaba001d0d6180f4bcfdd98dc59aac00238211d9aad",
     "owned-replay-host-fixture",
     13000,
     12,
     1,
     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
     "e0f7a4576c3851e93ef7d8c98bc149436788fb50893b288bbd72416f2d7f6659"}};
static const size_t admission_count = 1;
#else
static const struct admission admission_table[1] = {{.baseline = {0}}};
static const size_t admission_count = 0;
#endif
static const struct admission *admit(const struct cudaintercept_replay_request_v2 *r) {
    for (size_t i = 0; i < admission_count; i++) {
        const struct admission *a = &admission_table[i];
        if (!memcmp(a->baseline, r->operation.baseline, 65) &&
            !memcmp(a->candidate, r->operation.candidate, 65) &&
            !memcmp(a->qualifier, r->qualifier, 65) && !memcmp(a->adapter, r->adapter, 65) &&
            !memcmp(a->qualification, r->qualification, 65) &&
            !memcmp(a->caller, r->operation.caller, 64))
            return a;
    }
    return NULL;
}
struct cudaintercept_replay_owner_v2 {
    struct cudaintercept_owned_capsule_v2 capsule;
    struct driver_api api;
    module_t modules[2];
    function_t functions[2];
    ptr_t scratch[3];
    device_t device;
    pthread_t thread;
    unsigned char *images[2];
    int retained, published;
    void *driver_library;
    int driver_fd, descriptor_unknown, library_unknown;
    struct cudaintercept_replay_evidence_v2 evidence;
    atomic_int pending, used;
};

/* Observes and hashes the existing loader object's file, not an attestation
 * against a dishonest loader/filesystem. Native admission separately owns
 * that trusted installed cohort. No new library may be loaded here. */
static uint32_t driver_image(struct cudaintercept_replay_owner_v2 *o, const struct admission *a) {
    Dl_info info;
    memset(&info, 0, sizeof info);
    if (!dladdr((void *)o->api.current, &info) || !info.dli_fname)
        return CUDAINTERCEPT_DRIVER_ERROR;
    char path[PATH_MAX];
    if (!realpath(info.dli_fname, path))
        return CUDAINTERCEPT_DRIVER_ERROR;
#if !defined(CUDAINTERCEPT_PRIVATE_FIXTURE) && !defined(CUDAINTERCEPT_CPU_DRIVER)
#ifdef RTLD_NOLOAD
    o->driver_library = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!o->driver_library || dlsym(o->driver_library, "cuCtxGetCurrent") != (void *)o->api.current)
        return CUDAINTERCEPT_DRIVER_ERROR;
    /* Reject a partially interposed API set: every callable must resolve from
     * the same retained loader object, including the eventual tail launch. */
#define SAME(field, symbol)                                                                        \
    do {                                                                                           \
        void *observed = NULL;                                                                     \
        memcpy(&observed, &o->api.field, sizeof observed);                                         \
        if (dlsym(o->driver_library, symbol) != observed)                                          \
            return CUDAINTERCEPT_TOOLCHAIN;                                                        \
    } while (0)
    SAME(device, "cuCtxGetDevice");
    SAME(version, "cuDriverGetVersion");
    SAME(attribute, "cuDeviceGetAttribute");
    SAME(uuid, "cuDeviceGetUuid_v2");
    SAME(retain, "cuDevicePrimaryCtxRetain");
    SAME(release, "cuDevicePrimaryCtxRelease_v2");
    SAME(capture, "cuStreamIsCapturing");
    SAME(stream_context, "cuStreamGetCtx");
    SAME(load, "cuModuleLoadData");
    SAME(function, "cuModuleGetFunction");
    SAME(unload, "cuModuleUnload");
    SAME(alloc, "cuMemAlloc_v2");
    SAME(free, "cuMemFree_v2");
    SAME(write, "cuMemcpyHtoD_v2");
    SAME(read, "cuMemcpyDtoH_v2");
    SAME(sync, "cuStreamSynchronize");
    SAME(launch, "cuLaunchKernel");
#undef SAME
#else
    return CUDAINTERCEPT_DRIVER_ERROR;
#endif
#endif
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return CUDAINTERCEPT_DRIVER_ERROR;
    o->driver_fd = fd;
    struct stat before, after;
    uint32_t reason = CUDAINTERCEPT_DRIVER_ERROR;
    if (!fstat(fd, &before) && S_ISREG(before.st_mode) && before.st_size > 0 &&
        (uint64_t)before.st_size <= (1u << 30)) {
        struct hash s = hash_new();
        unsigned char bytes[8192];
        uint64_t n = 0;
        ssize_t k;
        while ((k = read(fd, bytes, sizeof bytes)) > 0) {
            n += (uint64_t)k;
            if (n > (1u << 30))
                break;
            hash_add(&s, bytes, (size_t)k);
        }
        if (k == 0 && n == (uint64_t)before.st_size && !fstat(fd, &after) &&
            before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
            before.st_size == after.st_size && before.st_mtime == after.st_mtime &&
            before.st_ctime == after.st_ctime) {
            hash_finish(&s, o->capsule.driver_image);
            reason = a && memcmp(o->capsule.driver_image, a->driver_image, 65)
                         ? CUDAINTERCEPT_TOOLCHAIN
                         : 0;
        }
    }
    if (close(fd)) {
        o->descriptor_unknown = 1;
        o->evidence.facts |= CUDAINTERCEPT_FACT_DESCRIPTOR_UNKNOWN;
        return CUDAINTERCEPT_DRIVER_ERROR;
    }
    o->driver_fd = -1;
    return reason;
}

static stream_t owner_stream(const struct cudaintercept_replay_owner_v2 *o) {
    return (stream_t)(uintptr_t)o->capsule.operation.stream;
}
static uint32_t current_guard(struct cudaintercept_replay_owner_v2 *o) {
    context_t context = NULL, stream_context = NULL;
    capture_t status = 0;
    if (o->capsule.process != (uint64_t)getpid())
        return CUDAINTERCEPT_GENERATION;
    if (!pthread_equal(o->thread, pthread_self()))
        return CUDAINTERCEPT_CONTEXT;
    if (o->api.current(&context))
        return CUDAINTERCEPT_DRIVER_ERROR;
    if ((uint64_t)(uintptr_t)context != o->capsule.operation.context)
        return CUDAINTERCEPT_CONTEXT;
    if (o->api.stream_context(owner_stream(o), &stream_context))
        return CUDAINTERCEPT_DRIVER_ERROR;
    if (stream_context != context)
        return CUDAINTERCEPT_STREAM;
    if (o->api.capture(owner_stream(o), &status))
        return CUDAINTERCEPT_DRIVER_ERROR;
    return status == 0 ? 0 : CUDAINTERCEPT_CAPTURE;
}
static void initial(unsigned char data[3][64]) {
    memset(data, 0xa5, 3 * 64);
    for (unsigned i = 0; i < 8; i++) {
        float x = (float)(i + 2), y = (float)(-8 - (int)i), w = (float)(5 + 2 * i);
        memcpy(data[0] + 16 + 4 * i, &x, 4);
        memcpy(data[1] + 16 + 4 * i, &y, 4);
        memcpy(data[2] + 16 + 4 * i, &w, 4);
    }
}
static uint32_t check_outputs(unsigned char got[3][64]) {
    unsigned char want[3][64];
    initial(want);
    for (unsigned i = 0; i < 8; i++) {
        float x = (float)(102 + i), y = (float)(7 + 3 * i);
        memcpy(want[0] + 16 + 4 * i, &x, 4);
        memcpy(want[1] + 16 + 4 * i, &y, 4);
    }
    /* Includes both writes, every stable-w byte and prefix/suffix guards.
     * Exact expected finite bits reject NaN/Inf/no-op/poison automatically. */
    for (unsigned j = 0; j < 3; j++)
        if (memcmp(got[j], want[j], 16) || memcmp(got[j] + 48, want[j] + 48, 16))
            return CUDAINTERCEPT_OUTPUT_MISMATCH;
    if (memcmp(got[0] + 16, want[0] + 16, 32))
        return CUDAINTERCEPT_OUTPUT_MISMATCH;
    if (memcmp(got[1] + 16, want[1] + 16, 32))
        return CUDAINTERCEPT_OUTPUT_MISMATCH;
    if (memcmp(got[2] + 16, want[2] + 16, 32))
        return CUDAINTERCEPT_OUTPUT_MISMATCH;
    return 0;
}
uint32_t cudaintercept_replay_finish_v2(struct cudaintercept_replay_owner_v2 *o) {
    if (cudaintercept_owned_in_hook_v2())
        return CUDAINTERCEPT_REENTRANT;
    if (!o)
        return CUDAINTERCEPT_OWNED_FRAME;
    if (!atomic_load(&o->pending))
        return 0;
    uint32_t r = current_guard(o);
    if (r)
        return r;
    if (o->api.sync(owner_stream(o)))
        return CUDAINTERCEPT_COMPLETION_UNKNOWN;
    atomic_store(&o->pending, 0);
    return 0;
}
static uint32_t canary(struct cudaintercept_replay_owner_v2 *o) {
    for (unsigned j = 0; j < 3; j++) {
        if (o->api.alloc(&o->scratch[j], 64) || !o->scratch[j])
            return CUDAINTERCEPT_DRIVER_ERROR;
    }
    /* Two separately reset baseline runs and one separately reset candidate. */
    for (unsigned arm = 0; arm < 3; arm++) {
        unsigned char data[3][64];
        initial(data);
        for (unsigned j = 0; j < 3; j++)
            if (o->api.write(o->scratch[j], data[j], 64))
                return CUDAINTERCEPT_DRIVER_ERROR;
        ptr_t x = o->scratch[0] + 16, y = o->scratch[1] + 16, w = o->scratch[2] + 16;
        unsigned n = 8;
        void *params[4] = {&x, &y, &w, &n};
        atomic_store(&o->pending, 1);
        result_t submitted = o->api.launch(o->functions[arm == 2], 1, 1, 1, 32, 1, 1, 0,
                                           owner_stream(o), params, NULL);
        uint32_t r = cudaintercept_replay_finish_v2(o);
        if (r)
            return r;
        if (submitted)
            return CUDAINTERCEPT_DRIVER_ERROR;
        for (unsigned j = 0; j < 3; j++)
            if (o->api.read(data[j], o->scratch[j], 64))
                return CUDAINTERCEPT_DRIVER_ERROR;
        memcpy(o->evidence.outputs[arm], data, sizeof data);
        o->evidence.completed_arms = arm + 1;
        r = check_outputs(data);
        if (r)
            return r;
    }
    return 0;
}
uint32_t cudaintercept_replay_cleanup_v2(struct cudaintercept_replay_owner_v2 **owner) {
    if (cudaintercept_owned_in_hook_v2())
        return CUDAINTERCEPT_REENTRANT;
    if (!owner || !*owner)
        return 0;
    struct cudaintercept_replay_owner_v2 *o = *owner;
    if (o->descriptor_unknown || o->library_unknown)
        return CUDAINTERCEPT_DRIVER_ERROR;
    if (o->published)
        return CUDAINTERCEPT_OWNER_LIFETIME;
    uint32_t r = 0;
    if (o->retained) {
        r = current_guard(o);
        if (r)
            return r;
        /* Even an earlier copy/load error may report an asynchronous fault.
         * Establish quiescence before freeing ANY adopted native resource. */
        atomic_store_explicit(&o->pending, 1, memory_order_release);
    }
    r = cudaintercept_replay_finish_v2(o);
    if (r)
        return r;
    for (unsigned j = 0; j < 3; j++)
        if (o->scratch[j]) {
            if (o->api.free(o->scratch[j]))
                return CUDAINTERCEPT_DRIVER_ERROR;
            o->scratch[j] = 0;
        }
    for (unsigned j = 0; j < 2; j++)
        if (o->modules[j]) {
            if (o->api.unload(o->modules[j]))
                return CUDAINTERCEPT_DRIVER_ERROR;
            o->modules[j] = NULL;
        }
    if (o->retained) {
        if (o->api.release(o->device))
            return CUDAINTERCEPT_DRIVER_ERROR;
        o->retained = 0;
    }
    if (o->driver_library) {
        if (dlclose(o->driver_library)) {
            o->library_unknown = 1;
            o->evidence.facts |= CUDAINTERCEPT_FACT_LIBRARY_UNKNOWN;
            return CUDAINTERCEPT_DRIVER_ERROR;
        }
        o->driver_library = NULL;
    }
    free(o->images[0]);
    free(o->images[1]);
    free(o);
    *owner = NULL;
    return 0;
}
static uint32_t prepare(const struct cudaintercept_replay_request_v2 *request,
                        struct cudaintercept_replay_owner_v2 **out, int publish) {
    if (cudaintercept_owned_in_hook_v2())
        return CUDAINTERCEPT_REENTRANT;
    if (!out)
        return CUDAINTERCEPT_METADATA_ABI;
    if (*out)
        return CUDAINTERCEPT_OWNER_LIFETIME; /* never overwrite adopted custody */
    if (!request || request->version != 2 || request->size != sizeof *request)
        return CUDAINTERCEPT_METADATA_ABI;
    struct cudaintercept_replay_request_v2 r = *request;
    if (r.operation.function || !r.baseline_image || !r.candidate_image || !r.baseline_bytes ||
        !r.candidate_bytes || r.baseline_bytes > CUDAINTERCEPT_IMAGE_LIMIT ||
        r.candidate_bytes > CUDAINTERCEPT_IMAGE_LIMIT)
        return CUDAINTERCEPT_IMAGE_INVALID;
    struct cudaintercept_replay_owner_v2 *o = calloc(1, sizeof *o);
    if (!o)
        return CUDAINTERCEPT_IMAGE_INVALID;
    *out = o;
    o->driver_fd = -1;
    o->thread = pthread_self();
    o->capsule.process = (uint64_t)getpid();
    /* Preserve all fixed v1 fields without hashing caller tail padding. */
    memcpy(&o->capsule.operation, &r.operation,
           offsetof(struct cudaintercept_replay_v1, flags) + sizeof r.operation.flags);
    o->evidence.version = 2;
    o->evidence.size = sizeof o->evidence;
    o->evidence.facts = CUDAINTERCEPT_FACT_PROCESS;
    o->evidence.process = o->capsule.process;
    atomic_init(&o->pending, 0);
    atomic_init(&o->used, 0);
    const void *input[2] = {r.baseline_image, r.candidate_image};
    uint64_t sizes[2] = {r.baseline_bytes, r.candidate_bytes};
    char hashes[2][65];
    for (unsigned j = 0; j < 2; j++) {
        o->images[j] = malloc((size_t)sizes[j]);
        if (!o->images[j])
            return CUDAINTERCEPT_IMAGE_INVALID;
        memcpy(o->images[j], input[j], (size_t)sizes[j]);
        digest(o->images[j], (size_t)sizes[j], hashes[j]);
    }
    memcpy(o->evidence.baseline, hashes[0], 65);
    memcpy(o->evidence.candidate, hashes[1], 65);
    o->evidence.facts |= CUDAINTERCEPT_FACT_IMAGES;
    if (memcmp(hashes[0], r.operation.baseline, 65) || memcmp(hashes[1], r.operation.candidate, 65))
        return CUDAINTERCEPT_IMAGE_INVALID;
    /* Complete independent matcher. Production has no row; in particular no
     * driver lookup/call can happen through production no-entry preparation. */
    const struct admission *a = publish ? admit(&r) : NULL;
    if (publish && !a)
        return CUDAINTERCEPT_ADAPTER_UNQUALIFIED;
    if (a) {
        memcpy(o->evidence.qualifier, a->qualifier, 65);
        memcpy(o->evidence.adapter, a->adapter, 65);
        memcpy(o->evidence.qualification, a->qualification, 65);
    }
#if !defined(CUDAINTERCEPT_REAL_HEADERS) && !defined(CUDAINTERCEPT_PRIVATE_FIXTURE) &&             \
    !defined(CUDAINTERCEPT_CPU_DRIVER)
    return CUDAINTERCEPT_ADAPTER_UNQUALIFIED;
#endif
    o->capsule.operation.function = 1; /* only for pre-load field validation */
    uint32_t reason = cudaintercept_owned_contract_v2(&o->capsule.operation);
    o->capsule.operation.function = 0;
    if (reason)
        return reason;
    if (!resolve_api(&o->api))
        return CUDAINTERCEPT_DRIVER_ERROR;
    context_t current = NULL, retained = NULL;
    replay_uuid_t uuid;
    int version = 0, major = 0, minor = 0;
    if (o->api.current(&current))
        return CUDAINTERCEPT_DRIVER_ERROR;
    o->evidence.context = (uint64_t)(uintptr_t)current;
    o->evidence.facts |= CUDAINTERCEPT_FACT_CONTEXT;
    if (!current || (uint64_t)(uintptr_t)current != r.operation.context)
        return CUDAINTERCEPT_CONTEXT;
    if (o->api.device(&o->device) || o->api.version(&version) ||
        o->api.attribute(&major, CAPABILITY_MAJOR, o->device) ||
        o->api.attribute(&minor, CAPABILITY_MINOR, o->device) || o->api.uuid(&uuid, o->device))
        return CUDAINTERCEPT_DRIVER_ERROR;
    if (o->device < 0 || version <= 0)
        return CUDAINTERCEPT_DRIVER_ERROR;
    o->evidence.driver = (uint32_t)version;
    o->evidence.major = (uint32_t)major;
    o->evidence.minor = (uint32_t)minor;
    memcpy(o->evidence.uuid, &uuid, 16);
    o->evidence.facts |= CUDAINTERCEPT_FACT_TARGET;
    if (major != 12 || minor != 1)
        return CUDAINTERCEPT_TARGET;
    if (a && (version != a->driver || major != a->major || minor != a->minor ||
              memcmp(&uuid, a->uuid, 16)))
        return CUDAINTERCEPT_TARGET;
    reason = driver_image(o, a);
    if (o->capsule.driver_image[0]) {
        memcpy(o->evidence.driver_image, o->capsule.driver_image, 65);
        o->evidence.facts |= CUDAINTERCEPT_FACT_DRIVER;
    }
    if (reason)
        return reason;
    result_t ret = o->api.retain(&retained, o->device);
    if (retained)
        o->retained = 1;
    if (ret || !retained)
        return CUDAINTERCEPT_DRIVER_ERROR;
    if (retained != current)
        return CUDAINTERCEPT_CONTEXT; /* primary-context owner only */
    reason = current_guard(o);
    if (reason)
        return reason;
    for (unsigned j = 0; j < 2; j++) {
        result_t loaded = o->api.load(&o->modules[j], o->images[j]);
        o->evidence.modules[j] = (uint64_t)(uintptr_t)o->modules[j];
        if (loaded || !o->modules[j])
            return CUDAINTERCEPT_DRIVER_ERROR;
        result_t resolved = o->api.function(&o->functions[j], o->modules[j], "replay");
        o->evidence.functions[j] = (uint64_t)(uintptr_t)o->functions[j];
        if (resolved || !o->functions[j])
            return CUDAINTERCEPT_DRIVER_ERROR;
    }
    if (o->functions[0] == o->functions[1])
        return CUDAINTERCEPT_IMAGE_INVALID;
    o->evidence.facts |= CUDAINTERCEPT_FACT_MODULES;
    reason = canary(o);
    if (reason)
        return reason;
    o->evidence.facts |= CUDAINTERCEPT_FACT_CANARY;
    digest(o->evidence.outputs, sizeof o->evidence.outputs, o->capsule.canary);
    if (publish && !cudaintercept_owned_prewarm_v2((uint64_t)(uintptr_t)o->api.launch))
        return CUDAINTERCEPT_DRIVER_ERROR;
    /* Completed scratch is discarded before publication. No temporary input
     * or borrowed application bytes supply the independent output oracle. */
    for (unsigned j = 0; j < 3; j++) {
        if (o->api.free(o->scratch[j]))
            return CUDAINTERCEPT_DRIVER_ERROR;
        o->scratch[j] = 0;
    }
    if (!publish)
        return 0; /* raw canary evidence NEVER creates eligibility */
    o->capsule.operation.function = (uint64_t)(uintptr_t)o->functions[0];
    o->capsule.candidate = (uint64_t)(uintptr_t)o->functions[1];
    for (unsigned j = 0; j < 2; j++)
        o->capsule.modules[j] = (uint64_t)(uintptr_t)o->modules[j];
    /* Capsule includes actual dynamic owner/handle identities as well as every
     * declaration and accepted qualifier/adapter/receipt identity. */
    struct {
        struct cudaintercept_owned_capsule_v2 capsule;
        char q[65], a[65], r[65];
        int driver, major, minor;
        unsigned char uuid[16];
    } identity;
    memset(&identity, 0, sizeof identity);
    identity.capsule = o->capsule;
    memcpy(identity.q, a->qualifier, 65);
    memcpy(identity.a, a->adapter, 65);
    memcpy(identity.r, a->qualification, 65);
    identity.driver = version;
    identity.major = major;
    identity.minor = minor;
    memcpy(identity.uuid, &uuid, 16);
    digest(&identity, sizeof identity, o->capsule.digest);
    reason = cudaintercept_owned_publish_v2(&o->capsule);
    if (reason)
        return reason;
    o->published = 1;
    o->evidence.facts |= CUDAINTERCEPT_FACT_PUBLISHED;
    memcpy(o->evidence.capsule, o->capsule.digest, 65);
    return 0;
}

uint32_t cudaintercept_replay_prepare_v2(const struct cudaintercept_replay_request_v2 *r,
                                         struct cudaintercept_replay_owner_v2 **out) {
    return prepare(r, out, 1);
}
uint32_t cudaintercept_replay_qualify_v2(const struct cudaintercept_replay_request_v2 *r,
                                         struct cudaintercept_replay_owner_v2 **out) {
    return prepare(r, out, 0);
}

uint32_t cudaintercept_replay_evidence_v2(const struct cudaintercept_replay_owner_v2 *o,
                                          struct cudaintercept_replay_evidence_v2 *out) {
    if (cudaintercept_owned_in_hook_v2())
        return CUDAINTERCEPT_REENTRANT;
    if (!o || !out)
        return CUDAINTERCEPT_METADATA_ABI;
    /* The off-hook owner contract excludes concurrent prepare/cleanup. */
    *out = o->evidence;
    return 0;
}

/* Exact exported boundary, also used by a host-only compiled fixture. */
/* Bind the legacy exported boundary explicitly, independent of any toolkit
 * macro mapping cuLaunchKernel to _ptsz. The real-header typedef gate remains. */
#ifdef __APPLE__
#define OWNED_BOUNDARY_SYMBOL "_cuLaunchKernel"
#else
#define OWNED_BOUNDARY_SYMBOL "cuLaunchKernel"
#endif
extern result_t owned_boundary(function_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, stream_t, void **,
                               void **) __asm__(OWNED_BOUNDARY_SYMBOL);
uint32_t cudaintercept_replay_launch_v2(struct cudaintercept_replay_owner_v2 *o,
                                        uint64_t generation, int *result) {
    if (!result)
        return CUDAINTERCEPT_OWNED_FRAME;
    *result = -1;
    if (cudaintercept_owned_in_hook_v2())
        return CUDAINTERCEPT_REENTRANT;
    if (!o || !o->published)
        return CUDAINTERCEPT_ADAPTER_UNQUALIFIED;
    uint32_t refusal = current_guard(o);
    if (refusal == CUDAINTERCEPT_DRIVER_ERROR)
        return refusal;
    if (atomic_load(&o->pending))
        return CUDAINTERCEPT_COMPLETION_UNKNOWN;
    if (generation != o->capsule.operation.caller_generation)
        refusal = CUDAINTERCEPT_GENERATION;
    if (atomic_exchange(&o->used, 1))
        refusal = CUDAINTERCEPT_OWNER_REUSED;
    if (!refusal) {
        if (o->api.sync(owner_stream(o))) {
            atomic_store(&o->pending, 1);
            return CUDAINTERCEPT_COMPLETION_UNKNOWN;
        }
        unsigned char initial_data[3][64], got[32];
        initial(initial_data);
        for (unsigned j = 0; j < 3; j++) {
            if (o->api.read(got, (ptr_t)o->capsule.operation.regions[j], 32))
                return CUDAINTERCEPT_DRIVER_ERROR;
            if (memcmp(got, initial_data[j] + 16, 32)) {
                refusal = CUDAINTERCEPT_POPULATION;
                break;
            }
        }
    }
    ptr_t x = (ptr_t)o->capsule.operation.regions[0], y = (ptr_t)o->capsule.operation.regions[1],
          w = (ptr_t)o->capsule.operation.regions[2];
    unsigned n = 8;
    void *params[4] = {&x, &y, &w, &n};
    struct cudaintercept_owned_frame_v2 frame = {.owner = &o->capsule,
                                                 .params = params,
                                                 .refusal = refusal,
                                                 .decision = CUDAINTERCEPT_OWNED_FRAME};
    if (!cudaintercept_owned_arm_v2(&frame))
        return CUDAINTERCEPT_REENTRANT;
    atomic_store(&o->pending, 1);
    *result = (int)owned_boundary((function_t)(uintptr_t)o->capsule.operation.function, 1, 1, 1, 32,
                                  1, 1, 0, owner_stream(o), params, NULL);
    cudaintercept_owned_clear_v2();
    return frame.decision;
}
