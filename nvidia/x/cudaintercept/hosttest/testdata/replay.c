/* Private CPU driver seam. Never a native qualification producer. */
#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#define dlsym fixture_dlsym
#define dladdr fixture_dladdr
#include "interpose-source.c"
#include "replay-source.c"
#include <stdio.h>

static const char *mode;
static const char *driver_file;
static struct cudaintercept_replay_owner_v2 *active_owner;
static atomic_uint launch_lookups;
static atomic_uint bad, driver_calls, base_calls, candidate_calls, loads, unloads, frees, retains,
    early_reads, allocations, major_queries, minor_queries;
static atomic_int fail_sync, fail_unload;
static int capture_status, driver_version = 13000, wrong_uuid, wrong_context, foreign_stream,
                           recursive;
static _Thread_local int in_real;
static _Thread_local struct {
    ptr_t x, y, w;
    unsigned n;
    int candidate, active;
} queued;
static _Thread_local function_t last_function;
static _Thread_local void **last_extra;
static float app[3][8];
static void check(int ok) {
    if (!ok)
        atomic_store(&bad, 1);
}
static result_t op(void) {
    atomic_fetch_add(&driver_calls, 1);
    check(!launching);
    return 0;
}
static result_t mock_current(context_t *v) {
    op();
    *v = (void *)(uintptr_t)(wrong_context ? 0x3001 : 0x3000);
    return 0;
}
static result_t mock_device(device_t *v) {
    op();
    *v = 7;
    return 0;
}
static result_t mock_version(int *v) {
    op();
    *v = driver_version;
    return 0;
}
static result_t mock_attribute(int *v, attribute_t attribute, device_t d) {
    op();
    check(d == 7);
    /* Literal CUDA values independently check the unqualified host constants. */
    check(attribute == 75 || attribute == 76);
    if (attribute == 75) {
        atomic_fetch_add(&major_queries, 1);
        *v = 12;
        return !strcmp(mode, "major_error") ? 2 : 0;
    }
    atomic_fetch_add(&minor_queries, 1);
    *v = 1;
    return !strcmp(mode, "minor_error") ? 2 : 0;
}
static result_t mock_uuid(replay_uuid_t *v, device_t d) {
    op();
    check(d == 7);
    for (unsigned i = 0; i < 16; i++)
        v->bytes[i] = (unsigned char)(i + 1 + wrong_uuid);
    return 0;
}
static result_t mock_retain(context_t *v, device_t d) {
    op();
    check(d == 7);
    *v = (void *)0x3000;
    atomic_fetch_add(&retains, 1);
    return 0;
}
static result_t mock_release(device_t d) {
    op();
    check(d == 7);
    atomic_fetch_sub(&retains, 1);
    return 0;
}
static result_t mock_capture(stream_t s, capture_t *v) {
    op();
    check(s == (void *)0x4000);
    *v = capture_status;
    return 0;
}
static result_t mock_stream_context(stream_t s, context_t *v) {
    op();
    check(s == (void *)0x4000);
    *v = (void *)(uintptr_t)(foreign_stream ? 0x3001 : 0x3000);
    return 0;
}
static result_t mock_load(module_t *v, const void *image) {
    op();
    int *m = malloc(sizeof *m);
    if (!m)
        return 2;
    *m = memcmp(image, "owned-test-baseline", 19) != 0;
    *v = m;
    unsigned n = atomic_fetch_add(&loads, 1);
    return !strcmp(mode, "partial_load") && n == 1 ? 2 : 0; /* nonnull on error */
}
static result_t mock_function(function_t *v, module_t m, const char *name) {
    op();
    check(!strcmp(name, "replay"));
    *v = (void *)(uintptr_t)(*(int *)m ? 0x2000 : 0x1000);
    return 0;
}
static result_t mock_unload(module_t m) {
    op();
    if (atomic_exchange(&fail_unload, 0))
        return 2;
    free(m);
    atomic_fetch_add(&unloads, 1);
    return 0;
}
static result_t mock_alloc(ptr_t *p, size_t n) {
    op();
    check(n == 64);
    atomic_fetch_add(&allocations, 1);
    *p = (ptr_t)(uintptr_t)malloc(n);
    return *p ? 0 : 2;
}
static result_t mock_free(ptr_t p) {
    op();
    free((void *)(uintptr_t)p);
    atomic_fetch_add(&frees, 1);
    return 0;
}
static result_t mock_write(ptr_t p, const void *v, size_t n) {
    op();
    check(!queued.active);
    memcpy((void *)(uintptr_t)p, v, n);
    return 0;
}
static result_t mock_read(void *v, ptr_t p, size_t n) {
    op();
    if (queued.active)
        atomic_fetch_add(&early_reads, 1);
    check(!queued.active);
    memcpy(v, (void *)(uintptr_t)p, n);
    return 0;
}
static void execute(void) {
    float *x = (float *)(uintptr_t)queued.x, *y = (float *)(uintptr_t)queued.y,
          *w = (float *)(uintptr_t)queued.w;
    if (queued.candidate && !strcmp(mode, "noop"))
        return;
    for (unsigned i = 0; i < queued.n; i++) {
        y[i] = x[i] + w[i];
        x[i] += 100;
        if (queued.candidate && (!strcmp(mode, "bad_y") || !strcmp(mode, "cleanup_unknown")))
            y[i] += 1;
        if (queued.candidate && !strcmp(mode, "bad_x"))
            x[i] -= 1;
        if (queued.candidate && !strcmp(mode, "bad_w"))
            w[i] += 1;
        if (queued.candidate && !strcmp(mode, "nan")) {
            uint32_t bits = 0x7fc00000;
            memcpy(&y[i], &bits, 4);
        }
    }
    if (queued.candidate && !strcmp(mode, "bad_guard"))
        ((unsigned char *)x)[-1] = 0;
}
static result_t mock_sync(stream_t s) {
    op();
    check(s == (void *)0x4000);
    if (atomic_load(&fail_sync))
        return 2;
    if (queued.active) {
        execute();
        queued.active = 0;
    }
    return 0;
}
static result_t mock_launch(function_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                            unsigned by, unsigned bz, unsigned shared, stream_t stream, void **p,
                            void **extra) {
    atomic_fetch_add(&driver_calls, 1);
    last_function = f;
    last_extra = extra;
    if (launching && !strcmp(mode, "owner_reentrant")) {
        unsigned before = atomic_load(&driver_calls);
        int result = 99;
        check(cudaintercept_replay_launch_v2(active_owner, 7, &result) == CUDAINTERCEPT_REENTRANT &&
              result == -1);
        check(atomic_load(&driver_calls) == before);
    }
    check(gx && gy && gz && bx && by && bz && shared == 0 && stream != NULL && p != NULL);
    if (f == (void *)0x2000)
        atomic_fetch_add(&candidate_calls, 1);
    else
        atomic_fetch_add(&base_calls, 1);
    /* CUDA copies parameter VALUES before asynchronous execution. */
    if (!extra && (f == (void *)0x1000 || f == (void *)0x2000 || f == (void *)0x5000)) {
        check(!queued.active);
        memcpy(&queued.x, p[0], sizeof(ptr_t));
        memcpy(&queued.y, p[1], sizeof(ptr_t));
        memcpy(&queued.w, p[2], sizeof(ptr_t));
        memcpy(&queued.n, p[3], sizeof(unsigned));
        check(queued.n <= 8);
        queued.candidate = f == (void *)0x2000;
        queued.active = 1;
    }
    if (recursive && launching && !in_real) {
        /* Nested call has its own arguments and no inherited owner token. */
        in_real = 1;
        void *dummy[4] = {0};
        check(cuLaunchKernel((void *)0x7777, 1, 1, 1, 32, 1, 1, 0, (void *)0x4000, dummy, NULL) ==
              77);
        in_real = 0;
    }
    return launching ? 77 : 0;
}
static result_t mock_other_launch(function_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                                  unsigned by, unsigned bz, unsigned shared, stream_t stream,
                                  void **p, void **extra) {
    return mock_launch(f, gx, gy, gz, bx, by, bz, shared, stream, p, extra);
}
void *fixture_dlsym(void *handle, const char *symbol) {
    check(handle == RTLD_NEXT);
    if (!strcmp(symbol, "cuLaunchKernel")) {
        unsigned n = atomic_fetch_add(&launch_lookups, 1);
        return n && !strcmp(mode, "tail_identity") ? (void *)mock_other_launch
                                                   : (void *)mock_launch;
    }
#define SYMBOL(name, fn)                                                                           \
    if (!strcmp(symbol, name))                                                                     \
    return (void *)fn
    SYMBOL("cuCtxGetCurrent", mock_current);
    SYMBOL("cuCtxGetDevice", mock_device);
    SYMBOL("cuDriverGetVersion", mock_version);
    SYMBOL("cuDeviceGetAttribute", mock_attribute);
    SYMBOL("cuDeviceGetUuid_v2", mock_uuid);
    SYMBOL("cuDevicePrimaryCtxRetain", mock_retain);
    SYMBOL("cuDevicePrimaryCtxRelease_v2", mock_release);
    SYMBOL("cuStreamIsCapturing", mock_capture);
    SYMBOL("cuStreamGetCtx", mock_stream_context);
    SYMBOL("cuModuleLoadData", mock_load);
    SYMBOL("cuModuleGetFunction", mock_function);
    SYMBOL("cuModuleUnload", mock_unload);
    SYMBOL("cuMemAlloc_v2", mock_alloc);
    SYMBOL("cuMemFree_v2", mock_free);
    SYMBOL("cuMemcpyHtoD_v2", mock_write);
    SYMBOL("cuMemcpyDtoH_v2", mock_read);
    SYMBOL("cuStreamSynchronize", mock_sync);
    SYMBOL("cuLaunchKernel", mock_launch);
#undef SYMBOL
    return NULL;
}
int fixture_dladdr(const void *symbol, Dl_info *info) {
    op();
    check(symbol == (void *)mock_current);
    memset(info, 0, sizeof *info);
    info->dli_fname = driver_file;
    return 1;
}
static void reset_app(void) {
    for (unsigned i = 0; i < 8; i++) {
        app[0][i] = (float)(i + 2);
        app[1][i] = (float)(-8 - (int)i);
        app[2][i] = (float)(5 + 2 * i);
    }
}
static struct cudaintercept_replay_request_v2 request(void) {
    struct cudaintercept_replay_request_v2 r = {.version = 2, .size = sizeof r};
    struct cudaintercept_replay_v1 *e = &r.operation;
    e->version = 1;
    e->size = sizeof *e;
    strcpy(e->operation, CUDAINTERCEPT_REPLAY_OPERATION);
    strcpy(e->caller, "owned-replay-host-fixture");
    e->caller_generation = 7;
    e->context_generation = 8;
    e->module_generation = 9;
    e->context = 0x3000;
    e->stream = 0x4000;
    for (unsigned j = 0; j < 3; j++) {
        e->regions[j] = (uint64_t)(uintptr_t)app[j];
        e->bytes[j] = 32;
        e->strides[j] = 4;
        e->parameter_bytes[j] = 8;
        e->parameter_kinds[j] = j == 2 ? 2 : 1;
        e->grid[j] = 1;
        e->block[j] = j == 0 ? 32 : 1;
    }
    e->parameter_bytes[3] = 4;
    e->parameter_kinds[3] = 3;
    e->n = 8;
    e->dtype = 1;
    e->layout = 1;
    e->alignment = 4;
    e->capture = 1;
    e->caller_owned = 1;
    strcpy(e->population, CUDAINTERCEPT_REPLAY_POPULATION);
    strcpy(e->target, "sm_121");
    strcpy(e->nvrtc, CUDAINTERCEPT_REPLAY_NVRTC);
    strcpy(e->builtins, CUDAINTERCEPT_REPLAY_BUILTINS);
    strcpy(e->flags, CUDAINTERCEPT_REPLAY_FLAGS);
    r.baseline_image = "owned-test-baseline";
    r.baseline_bytes = 19;
    r.candidate_image = "owned-test-candidate";
    r.candidate_bytes = 20;
    digest(r.baseline_image, 19, e->baseline);
    digest(r.candidate_image, 20, e->candidate);
    strcpy(r.qualifier, "d89688e1c69ad5808828a1e53193c4f20287731670c93603d39b7c416ac7ad15");
    strcpy(r.adapter, "9c249f861690fe913bcd3d19c031340276591a5787719185a81df051df1aabcb");
    strcpy(r.qualification, "2997f3e7e8429c34a9ddaaaba001d0d6180f4bcfdd98dc59aac00238211d9aad");
    return r;
}
static int original(void) {
    ptr_t x = (ptr_t)(uintptr_t)app[0], y = (ptr_t)(uintptr_t)app[1], w = (ptr_t)(uintptr_t)app[2];
    unsigned n = 8;
    void *p[4] = {&x, &y, &w, &n};
    /* Caller-owned original is independent of the qualifier's loaded modules. */
    return cuLaunchKernel((void *)0x5000, 1, 1, 1, 32, 1, 1, 0, (void *)0x4000, p, NULL);
}
static void assert_app(void) {
    for (unsigned i = 0; i < 8; i++)
        check(app[0][i] == (float)(102 + i) && app[1][i] == (float)(7 + 3 * i) &&
              app[2][i] == (float)(5 + 2 * i));
}
static void *cross(void *p) {
    int result = -1;
    check(cudaintercept_replay_launch_v2(p, 7, &result) == CUDAINTERCEPT_CONTEXT);
    check(result == 77);
    check(mock_sync((void *)0x4000) == 0);
    return NULL;
}
static void *initialize(void *unused) {
    (void)unused;
    struct cudaintercept_replay_request_v2 r = request();
    struct cudaintercept_replay_owner_v2 *o = NULL;
    uint32_t reason = cudaintercept_replay_prepare_v2(&r, &o);
    if (reason)
        check(reason == CUDAINTERCEPT_ALREADY_REGISTERED &&
              cudaintercept_replay_cleanup_v2(&o) == 0);
    else
        check(o != NULL);
    return NULL;
}
int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    mode = argv[1];
    driver_file = argv[2];
    reset_app();
    check(setenv("CUDAINTERCEPT_OBSERVE", "1", 1) == 0);
    /* Independent known SHA256 vectors cover padding/block-boundary code. */
    char h[65];
    digest("", 0, h);
    check(!strcmp(h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    digest("abc", 3, h);
    check(!strcmp(h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    digest("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 64, h);
    check(!strcmp(h, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"));
    cudaintercept_replay_enable_v1(1);
    cudaintercept_decisions_enable_v1(1);
    if (!strcmp(mode, "concurrent")) {
        pthread_t threads[2];
        for (unsigned j = 0; j < 2; j++)
            check(!pthread_create(&threads[j], NULL, initialize, NULL));
        for (unsigned j = 0; j < 2; j++)
            check(!pthread_join(threads[j], NULL));
        check(atomic_load(&loads) == 4 && atomic_load(&unloads) == 2 && atomic_load(&retains) == 1);
        goto done;
    }
    struct cudaintercept_replay_request_v2 r = request();
    char caller_baseline[] = "owned-test-baseline", caller_candidate[] = "owned-test-candidate";
    r.baseline_image = caller_baseline;
    r.candidate_image = caller_candidate;
    if (!strcmp(mode, "qualifier"))
        r.qualifier[0] ^= 1;
    if (!strcmp(mode, "adapter"))
        r.adapter[0] ^= 1;
    if (!strcmp(mode, "qualification"))
        r.qualification[0] ^= 1;
    if (!strcmp(mode, "image")) {
        caller_candidate[18] ^= 1;
    }
    if (!strcmp(mode, "shape"))
        r.operation.n = 7;
    if (!strcmp(mode, "graph"))
        r.operation.capture = 3;
    if (!strcmp(mode, "stream_special"))
        r.operation.stream = 2;
    if (!strcmp(mode, "driver"))
        driver_version++;
    if (!strcmp(mode, "uuid"))
        wrong_uuid = 1;
    if (!strcmp(mode, "sync_unknown"))
        atomic_store(&fail_sync, 1);
    struct cudaintercept_replay_owner_v2 *o = NULL;
    if (!strcmp(mode, "raw_only")) {
        check(admission_count == 0);
        check(cudaintercept_replay_qualify_v2(&r, &o) == 0 && o != NULL);
        struct cudaintercept_replay_evidence_v2 raw;
        check(cudaintercept_replay_evidence_v2(o, &raw) == 0 && raw.completed_arms == 3 &&
              (raw.facts & CUDAINTERCEPT_FACT_CANARY) &&
              !(raw.facts & CUDAINTERCEPT_FACT_PUBLISHED) && !raw.capsule[0]);
        int result = 99;
        unsigned before = atomic_load(&driver_calls);
        check(cudaintercept_replay_launch_v2(o, 7, &result) == CUDAINTERCEPT_ADAPTER_UNQUALIFIED &&
              result == -1);
        check(atomic_load(&driver_calls) == before);
        const uint32_t grid[3] = {1, 1, 1}, block[3] = {32, 1, 1};
        uint64_t generation = 0;
        check(cudaintercept_replay_decide_v1(0x1000, 0x4000, grid, block, 0, 1, 0, 0,
                                             &generation) == CUDAINTERCEPT_REGISTRY_MISS);
        check(cudaintercept_replay_cleanup_v2(&o) == 0 && o == NULL);
        unsigned candidates = atomic_load(&candidate_calls);
        check(original() == 77 && mock_sync((void *)0x4000) == 0);
        assert_app();
        check(atomic_load(&candidate_calls) == candidates);
        goto done;
    }
    uint32_t reason = cudaintercept_replay_prepare_v2(&r, &o);
    if (!strcmp(mode, "production") || !strcmp(mode, "qualifier") || !strcmp(mode, "adapter") ||
        !strcmp(mode, "qualification") || !strcmp(mode, "image")) {
        check(reason == (!strcmp(mode, "image") ? CUDAINTERCEPT_IMAGE_INVALID
                                                : CUDAINTERCEPT_ADAPTER_UNQUALIFIED));
        check(atomic_load(&driver_calls) == 0);
        check(cudaintercept_replay_cleanup_v2(&o) == 0 && o == NULL);
        goto done;
    }
    if (!strcmp(mode, "major_error") || !strcmp(mode, "minor_error")) {
        check(reason == CUDAINTERCEPT_DRIVER_ERROR && o != NULL);
        check(atomic_load(&major_queries) == 1);
        check(atomic_load(&minor_queries) == (unsigned)!strcmp(mode, "minor_error"));
        check(atomic_load(&loads) == 0 && atomic_load(&allocations) == 0 &&
              atomic_load(&base_calls) == 0 && atomic_load(&candidate_calls) == 0 &&
              atomic_load(&retains) == 0);
        struct cudaintercept_replay_evidence_v2 evidence;
        check(cudaintercept_replay_evidence_v2(o, &evidence) == 0 &&
              !(evidence.facts & (CUDAINTERCEPT_FACT_CANARY | CUDAINTERCEPT_FACT_PUBLISHED)) &&
              evidence.completed_arms == 0 && !evidence.capsule[0]);
        check(cudaintercept_replay_cleanup_v2(&o) == 0 && o == NULL);
        goto done;
    }
    if (!strcmp(mode, "shape") || !strcmp(mode, "graph") || !strcmp(mode, "stream_special") ||
        !strcmp(mode, "driver") || !strcmp(mode, "uuid")) {
        check(reason == (!strcmp(mode, "shape")            ? CUDAINTERCEPT_SHAPE
                         : !strcmp(mode, "graph")          ? CUDAINTERCEPT_GRAPH_EXECUTION
                         : !strcmp(mode, "stream_special") ? CUDAINTERCEPT_STREAM
                                                           : CUDAINTERCEPT_TARGET));
        check(atomic_load(&loads) == 0);
        check(cudaintercept_replay_cleanup_v2(&o) == 0);
        goto done;
    }
    if (!strcmp(mode, "partial_load") || !strcmp(mode, "sync_unknown") ||
        !strcmp(mode, "tail_identity")) {
        check(reason == (!strcmp(mode, "sync_unknown") ? CUDAINTERCEPT_COMPLETION_UNKNOWN
                                                       : CUDAINTERCEPT_DRIVER_ERROR));
        unsigned launches = atomic_load(&base_calls) + atomic_load(&candidate_calls);
        if (!strcmp(mode, "sync_unknown")) {
            check(cudaintercept_replay_cleanup_v2(&o) == CUDAINTERCEPT_COMPLETION_UNKNOWN &&
                  o != NULL);
            check(atomic_load(&unloads) == 0);
            atomic_store(&fail_sync, 0);
        }
        check(cudaintercept_replay_cleanup_v2(&o) == 0 && o == NULL);
        check(atomic_load(&unloads) == 2 && atomic_load(&retains) == 0);
        check(launches == atomic_load(&base_calls) + atomic_load(&candidate_calls));
        goto done;
    }
    if (!strcmp(mode, "bad_y") || !strcmp(mode, "bad_x") || !strcmp(mode, "bad_w") ||
        !strcmp(mode, "bad_guard") || !strcmp(mode, "nan") || !strcmp(mode, "noop") ||
        !strcmp(mode, "cleanup_unknown")) {
        check(reason == CUDAINTERCEPT_OUTPUT_MISMATCH && o != NULL);
        struct cudaintercept_replay_evidence_v2 failed;
        check(cudaintercept_replay_evidence_v2(o, &failed) == 0 && failed.completed_arms == 3 &&
              !(failed.facts & CUDAINTERCEPT_FACT_PUBLISHED) && !failed.capsule[0]);
        if (!strcmp(mode, "cleanup_unknown")) {
            atomic_store(&fail_sync, 1);
            check(cudaintercept_replay_cleanup_v2(&o) == CUDAINTERCEPT_COMPLETION_UNKNOWN &&
                  o != NULL);
            check(atomic_load(&frees) == 0 && atomic_load(&unloads) == 0);
            atomic_store(&fail_sync, 0);
        }
        if (!strcmp(mode, "bad_w")) {
            atomic_store(&fail_unload, 1);
            check(cudaintercept_replay_cleanup_v2(&o) == CUDAINTERCEPT_DRIVER_ERROR && o != NULL);
        }
        check(cudaintercept_replay_cleanup_v2(&o) == 0 && o == NULL);
        unsigned before = atomic_load(&candidate_calls);
        check(original() == 77);
        check(mock_sync((void *)0x4000) == 0);
        assert_app();
        check(atomic_load(&candidate_calls) == before);
        goto done;
    }
    check(reason == 0 && o != NULL);
    if (reason || !o)
        goto done;
    struct cudaintercept_replay_evidence_v2 evidence;
    check(cudaintercept_replay_evidence_v2(o, &evidence) == 0 && evidence.completed_arms == 3 &&
          evidence.facts == 255 && evidence.context == 0x3000 && evidence.functions[0] == 0x1000 &&
          evidence.functions[1] == 0x2000 && evidence.capsule[0]);
    unsigned char saved = evidence.outputs[0][0][16];
    evidence.outputs[0][0][16] ^= 0xff; /* detached evidence never mutates owner */
    check(cudaintercept_replay_evidence_v2(o, &evidence) == 0 &&
          evidence.outputs[0][0][16] == saved);
    struct cudaintercept_replay_owner_v2 *same_owner = o;
    unsigned calls_before_repeat = atomic_load(&driver_calls);
    check(cudaintercept_replay_prepare_v2(&r, &o) == CUDAINTERCEPT_OWNER_LIFETIME &&
          o == same_owner && atomic_load(&driver_calls) == calls_before_repeat);
    /* Caller storage changes cannot mutate image snapshots or the capsule. */
    memset(caller_baseline, 0, sizeof caller_baseline);
    memset(caller_candidate, 0, sizeof caller_candidate);
    check(!memcmp(o->images[0], "owned-test-baseline", 19) &&
          !memcmp(o->images[1], "owned-test-candidate", 20));
    memset(&r, 0, sizeof r);
    unsigned candidates = atomic_load(&candidate_calls), bases = atomic_load(&base_calls);
    active_owner = o;
    if (!strcmp(mode, "bounded")) {
        for (unsigned i = 0; i < 200; i++) {
            reset_app();
            int result = -1;
            check(cudaintercept_replay_launch_v2(o, 7, &result) ==
                      (i ? CUDAINTERCEPT_OWNER_REUSED : CUDAINTERCEPT_REPLACED) &&
                  result == 77);
            check(cudaintercept_replay_finish_v2(o) == 0);
            assert_app();
        }
        struct cudaintercept_owned_decision_v2 records[128];
        uint64_t dropped = 0;
        check(cudaintercept_owned_drain_v2(records, 128, &dropped) == 128 && dropped == 72);
        for (unsigned i = 0; i < 128; i++)
            check(records[i].seq == i && records[i].requested_function == 0x1000 &&
                  records[i].dispatched_function == (i ? 0x1000 : 0x2000) && records[i].capsule[0]);
        check(cudaintercept_owned_drain_v2(records, 1, &dropped) == 0 && dropped == 72);
        goto done;
    }
    if (!strncmp(mode, "frame_", 6)) {
        ptr_t x = (ptr_t)(uintptr_t)app[0], y = (ptr_t)(uintptr_t)app[1],
              w = (ptr_t)(uintptr_t)app[2];
        unsigned n = 8;
        void *params[4] = {&x, &y, &w, &n}, *other[4] = {&x, &y, &w, &n}, *extra[1] = {NULL};
        struct cudaintercept_owned_capsule_v2 cloned = o->capsule;
        struct cudaintercept_owned_frame_v2 f = {
            .owner = !strcmp(mode, "frame_owner") ? &cloned : &o->capsule, .params = params};
        check(cudaintercept_owned_arm_v2(&f));
        void **actual = !strcmp(mode, "frame_swap") ? other : params;
        void **actual_extra = !strcmp(mode, "frame_extra") ? extra : NULL;
        void *fn = !strcmp(mode, "frame_function") ? (void *)0x7777 : (void *)0x1000;
        unsigned gx = !strcmp(mode, "frame_geometry") ? 2 : 1;
        void *stream = !strcmp(mode, "frame_stream") ? (void *)0x4001 : (void *)0x4000;
        check(cuLaunchKernel(fn, gx, 1, 1, 32, 1, 1, 0, stream, actual, actual_extra) == 77);
        uint32_t expected = CUDAINTERCEPT_ARGUMENTS;
        if (!strcmp(mode, "frame_function"))
            expected = CUDAINTERCEPT_REGISTRY_MISS;
        if (!strcmp(mode, "frame_geometry"))
            expected = CUDAINTERCEPT_GEOMETRY;
        if (!strcmp(mode, "frame_stream"))
            expected = CUDAINTERCEPT_STREAM;
        if (!strcmp(mode, "frame_owner"))
            expected = CUDAINTERCEPT_ADAPTER_UNQUALIFIED;
        check(f.decision == expected && last_function == fn && last_extra == actual_extra);
        check(atomic_load(&candidate_calls) == candidates && atomic_load(&base_calls) == bases + 1);
        check(mock_sync((void *)0x4000) == 0);
        goto done;
    }
    if (!strcmp(mode, "cross_thread")) {
        pthread_t t;
        check(!pthread_create(&t, NULL, cross, o));
        check(!pthread_join(t, NULL));
        check(atomic_load(&candidate_calls) == candidates);
        check(cudaintercept_replay_finish_v2(o) == 0);
        goto done;
    }
    if (!strcmp(mode, "context"))
        wrong_context = 1;
    if (!strcmp(mode, "stream_context"))
        foreign_stream = 1;
    if (!strcmp(mode, "capture"))
        capture_status = 1;
    if (!strcmp(mode, "population"))
        app[0][0] += 1;
    if (!strcmp(mode, "default"))
        cudaintercept_replay_enable_v1(0);
    if (!strcmp(mode, "process"))
        cudaintercept_process_disable_v1(1);
    if (!strcmp(mode, "operation"))
        cudaintercept_replay_disable_v1(1);
    if (!strcmp(mode, "legacy_binding") || !strcmp(mode, "legacy_invalid")) {
        struct cudaintercept_replay_v1 declaration = o->capsule.operation;
        strcpy(declaration.baseline, CUDAINTERCEPT_REPLAY_BASELINE);
        strcpy(declaration.candidate, CUDAINTERCEPT_REPLAY_CANDIDATE);
        if (!strcmp(mode, "legacy_invalid"))
            declaration.version = 0;
        check(cudaintercept_replay_bind_v1(&declaration) ==
              (!strcmp(mode, "legacy_invalid") ? CUDAINTERCEPT_METADATA_ABI : 0));
    }
    recursive = !strcmp(mode, "recursive");
    int result = -1;
    uint32_t decision =
        cudaintercept_replay_launch_v2(o, !strcmp(mode, "generation") ? 6 : 7, &result);
    uint32_t expected = CUDAINTERCEPT_REPLACED;
    if (!strcmp(mode, "legacy_binding"))
        expected = CUDAINTERCEPT_ADAPTER_UNQUALIFIED;
    if (!strcmp(mode, "legacy_invalid"))
        expected = CUDAINTERCEPT_METADATA_ABI;
    if (!strcmp(mode, "context"))
        expected = CUDAINTERCEPT_CONTEXT;
    if (!strcmp(mode, "stream_context"))
        expected = CUDAINTERCEPT_STREAM;
    if (!strcmp(mode, "capture"))
        expected = CUDAINTERCEPT_CAPTURE;
    if (!strcmp(mode, "population"))
        expected = CUDAINTERCEPT_POPULATION;
    if (!strcmp(mode, "generation"))
        expected = CUDAINTERCEPT_GENERATION;
    if (!strcmp(mode, "default"))
        expected = CUDAINTERCEPT_DEFAULT_OFF;
    if (!strcmp(mode, "process"))
        expected = CUDAINTERCEPT_PROCESS_DISABLED;
    if (!strcmp(mode, "operation"))
        expected = CUDAINTERCEPT_OPERATION_DISABLED;
    check(decision == expected && result == 77);
    check(atomic_load(&candidate_calls) == candidates + (expected == CUDAINTERCEPT_REPLACED));
    check(atomic_load(&base_calls) ==
          bases + (expected != CUDAINTERCEPT_REPLACED) + (recursive ? 1 : 0));
    wrong_context = 0;
    capture_status = 0;
    foreign_stream = 0;
    check(cudaintercept_replay_finish_v2(o) == 0);
    if (expected == CUDAINTERCEPT_REPLACED)
        assert_app();
    struct cudaintercept_owned_decision_v2 record;
    uint64_t dropped = 99;
    check(cudaintercept_owned_drain_v2(&record, 1, &dropped) == 1 && dropped == 0);
    check(record.version == 2 && record.size == 128 && record.reason == expected &&
          record.result == 77);
    check(record.requested_function == 0x1000 &&
          record.dispatched_function == (expected == CUDAINTERCEPT_REPLACED ? 0x2000 : 0x1000) &&
          record.capsule[0]);
    struct cudaintercept_record observed[2];
    uint64_t observation_drops = 99;
    long observed_n = cudaintercept_drain(observed, 2, &observation_drops);
    check(observed_n == (recursive ? 2 : 1) && observation_drops == 0);
    if (observed_n > 0)
        check(observed[observed_n - 1].function == 0x1000 &&
              observed[observed_n - 1].result == 77 && observed[observed_n - 1].version == 1 &&
              (observed[observed_n - 1].flags & CUDAINTERCEPT_ARGUMENTS_UNAVAILABLE));
    check(cudaintercept_replay_cleanup_v2(&o) == CUDAINTERCEPT_OWNER_LIFETIME);
    if (!strcmp(mode, "single_use")) {
        reset_app();
        unsigned n = atomic_load(&candidate_calls);
        check(cudaintercept_replay_launch_v2(o, 7, &result) == CUDAINTERCEPT_OWNER_REUSED &&
              result == 77);
        check(cudaintercept_replay_finish_v2(o) == 0 && atomic_load(&candidate_calls) == n);
        assert_app();
    }
done:
    printf("mode=%s bad=%u baseline=%u candidate=%u driver=%u early_reads=%u\n", mode,
           atomic_load(&bad), atomic_load(&base_calls), atomic_load(&candidate_calls),
           atomic_load(&driver_calls), atomic_load(&early_reads));
    return atomic_load(&bad) ? 1 : 0;
}
