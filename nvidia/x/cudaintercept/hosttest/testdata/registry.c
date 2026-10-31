/* Host-only actual shim boundary fixture. No CUDA headers, tools or device. */
#define _GNU_SOURCE
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define dlsym fixture_dlsym
#include "interpose-source.c"

_Static_assert(CUDAINTERCEPT_DEFAULT_OFF == 1, "default_off enum value");
_Static_assert(CUDAINTERCEPT_ALREADY_REGISTERED == 32, "already_registered enum value");
_Static_assert(CUDAINTERCEPT_GRAPH_EXECUTION == 33, "graph_execution enum value");

static atomic_uint calls;
static atomic_uint winners;
static atomic_int bad;
static int recursive;
static _Thread_local int inside_real;
static void **const opaque_params = (void **)(uintptr_t)0x1234;
static void **const opaque_extra = (void **)(uintptr_t)0x5678;
static _Thread_local uint32_t launch_grid[3] = {1, 1, 1};
static _Thread_local uint32_t launch_block[3] = {32, 1, 1};
static _Thread_local void *launch_function = (void *)0x1000;
static _Thread_local void *launch_stream = (void *)0x2000;
static _Thread_local unsigned launch_shared;
static _Thread_local void **launch_params;
static _Thread_local void **launch_extra;
static _Thread_local int wanted_result = INT_MIN;

static void check(int ok) {
    if (!ok) atomic_store(&bad, 1);
}

static int launch(void) {
    return cuLaunchKernel(launch_function, launch_grid[0], launch_grid[1], launch_grid[2],
        launch_block[0], launch_block[1], launch_block[2], launch_shared,
        launch_stream, launch_params, launch_extra);
}

static int real_launch(void *f, unsigned x, unsigned y, unsigned z,
    unsigned bx, unsigned by, unsigned bz, unsigned sm, void *stream, void **p, void **e) {
    atomic_fetch_add(&calls, 1);
    check(f == launch_function && stream == launch_stream);
    check(x == launch_grid[0] && y == launch_grid[1] && z == launch_grid[2]);
    check(bx == launch_block[0] && by == launch_block[1] && bz == launch_block[2]);
    check(sm == launch_shared && p == launch_params && e == launch_extra);
    if (recursive && !inside_real) {
        inside_real = 1;
        check(launch() == wanted_result);
        inside_real = 0;
    }
    return wanted_result;
}

void *fixture_dlsym(void *handle, const char *symbol) {
    check(handle == RTLD_NEXT && strcmp(symbol, "cuLaunchKernel") == 0);
    return (void *)real_launch;
}

static struct cudaintercept_replay_v1 entry(void) {
    struct cudaintercept_replay_v1 e = {0};
    e.version = CUDAINTERCEPT_REPLAY_VERSION;
    e.size = sizeof e;
    strcpy(e.operation, CUDAINTERCEPT_REPLAY_OPERATION);
    strcpy(e.caller, "registry-host-fixture");
    e.caller_generation = 7;
    e.context_generation = 8;
    e.module_generation = 9;
    e.context = 0x3000;
    e.function = 0x1000;
    e.stream = 0x2000;
    for (int i = 0; i < 3; i++) {
        e.regions[i] = 0x4000 + 32 * i;
        e.bytes[i] = 32;
        e.strides[i] = 4;
        e.parameter_bytes[i] = 8;
        e.parameter_kinds[i] = i == 2 ? 2 : 1;
        e.grid[i] = 1;
        e.block[i] = i == 0 ? 32 : 1;
    }
    e.parameter_bytes[3] = 4;
    e.parameter_kinds[3] = 3;
    e.n = 8;
    e.dtype = 1;
    e.layout = 1;
    e.alignment = 4;
    e.capture = 1;
    e.caller_owned = 1;
    strcpy(e.population, CUDAINTERCEPT_REPLAY_POPULATION);
    strcpy(e.target, "sm_121");
    strcpy(e.baseline, CUDAINTERCEPT_REPLAY_BASELINE);
    strcpy(e.candidate, CUDAINTERCEPT_REPLAY_CANDIDATE);
    strcpy(e.nvrtc, CUDAINTERCEPT_REPLAY_NVRTC);
    strcpy(e.builtins, CUDAINTERCEPT_REPLAY_BUILTINS);
    strcpy(e.flags, CUDAINTERCEPT_REPLAY_FLAGS);
    return e;
}

static void expect(uint32_t reason, uint64_t generation, unsigned expected_calls) {
    struct cudaintercept_decision_v1 r[2];
    uint64_t dropped = UINT64_MAX;
    check(cudaintercept_decision_drain_v1(r, 2, &dropped) == 1);
    check(r[0].version == CUDAINTERCEPT_DECISION_VERSION && r[0].size == sizeof r[0]);
    check(r[0].reason == reason && r[0].caller_generation == generation);
    check(r[0].result == wanted_result);
    check(r[0].function == (uintptr_t)launch_function && r[0].stream == (uintptr_t)launch_stream);
    check(dropped == 0 && atomic_load(&calls) == expected_calls);
}

static void *initializer(void *unused) {
    (void)unused;
    struct cudaintercept_replay_v1 e = entry();
    uint32_t r = cudaintercept_replay_register_v1(&e);
    if (!r) atomic_fetch_add(&winners, 1);
    else check(r == CUDAINTERCEPT_ALREADY_REGISTERED);
    check(cudaintercept_replay_bind_v1(&e) == 0);
    launch_params = opaque_params;
    for (int i = 0; i < 8; i++) check(launch() == wanted_result);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const char *mode = argv[1];
    struct cudaintercept_replay_v1 e = entry(), call = e;
    launch_params = opaque_params;
    check(cudaintercept_decisions_enable_v1(1) == 0);
    if (!strcmp(mode, "owned_absent")) {
        check(cudaintercept_replay_register_v1(&e) == 0);
        check(cudaintercept_replay_enable_v1(1) == 0);
        struct cudaintercept_owned_capsule_v2 declared = {.operation=e};
        struct cudaintercept_owned_frame_v2 frame = {.owner=&declared, .params=opaque_params};
        check(cudaintercept_owned_arm_v2(&frame));
        check(launch() == wanted_result);
        struct cudaintercept_owned_decision_v2 record;
        uint64_t dropped = 99;
        check(cudaintercept_owned_drain_v2(&record, 1, &dropped) == 1);
        check(record.reason == CUDAINTERCEPT_ADAPTER_UNQUALIFIED);
        check(record.requested_function == record.dispatched_function && dropped == 0);
        check(atomic_load(&calls) == 1);
        printf("mode=%s bad=%d calls=%u\n", mode, atomic_load(&bad), atomic_load(&calls));
        return atomic_load(&bad) ? 1 : 0;
    }
    if (!strcmp(mode, "concurrent")) {
        check(cudaintercept_replay_enable_v1(1) == 0);
        pthread_t threads[16];
        for (int i = 0; i < 16; i++) check(pthread_create(&threads[i], NULL, initializer, NULL) == 0);
        for (int i = 0; i < 16; i++) check(pthread_join(threads[i], NULL) == 0);
        struct cudaintercept_decision_v1 r[CUDAINTERCEPT_DECISION_CAPACITY];
        uint64_t dropped;
        long n = cudaintercept_decision_drain_v1(r, CUDAINTERCEPT_DECISION_CAPACITY, &dropped);
        check(n == 128 && dropped == 0 && atomic_load(&calls) == 128 && atomic_load(&winners) == 1);
        unsigned unqualified = 0, no_binding = 0, unavailable = 0;
        for (long i = 0; i < n; i++) {
            check(r[i].seq == (uint64_t)i && r[i].result == INT_MIN);
            if (r[i].reason == CUDAINTERCEPT_ADAPTER_UNQUALIFIED) unqualified++;
            else if (r[i].reason == CUDAINTERCEPT_NO_BINDING) no_binding++;
            else if (r[i].reason == CUDAINTERCEPT_REGISTRY_UNAVAILABLE) unavailable++;
            else check(0);
        }
        check(unqualified <= 16 && no_binding <= 112 && unqualified + no_binding + unavailable == 128);
        check(cudaintercept_replay_bind_v1(&e) == 0);
        check(launch() == wanted_result);
        expect(CUDAINTERCEPT_ADAPTER_UNQUALIFIED, 7, 129);
    } else if (!strcmp(mode, "bounded")) {
        for (unsigned i = 0; i < CUDAINTERCEPT_DECISION_CAPACITY + 3; i++) check(launch() == wanted_result);
        struct cudaintercept_decision_v1 r[CUDAINTERCEPT_DECISION_CAPACITY];
        uint64_t dropped;
        long n = cudaintercept_decision_drain_v1(r, CUDAINTERCEPT_DECISION_CAPACITY, &dropped);
        check(n == CUDAINTERCEPT_DECISION_CAPACITY && dropped == 3);
        for (long i = 0; i < n; i++) check(r[i].seq == (uint64_t)i && r[i].reason == CUDAINTERCEPT_DEFAULT_OFF);
        check(launch() == wanted_result);
        check(cudaintercept_decision_drain_v1(r, 1, &dropped) == 1 && r[0].seq == CUDAINTERCEPT_DECISION_CAPACITY && dropped == 3);
    } else if (!strcmp(mode, "decision_off")) {
        uint64_t dropped = 99;
        check(cudaintercept_decisions_enable_v1(0) == 0);
        check(launch() == wanted_result);
        check(cudaintercept_decision_drain_v1(NULL, 0, &dropped) == -1 && dropped == 0);
        check(cudaintercept_decisions_enable_v1(1) == 0);
        check(cudaintercept_decision_drain_v1(NULL, 0, &dropped) == 0 && dropped == 0);
    } else {
        uint32_t reason = CUDAINTERCEPT_ADAPTER_UNQUALIFIED;
        if (strcmp(mode, "miss")) check(cudaintercept_replay_register_v1(&e) == 0);
        check(cudaintercept_replay_enable_v1(1) == 0);
        if (!strcmp(mode, "default")) { cudaintercept_replay_enable_v1(0); reason = CUDAINTERCEPT_DEFAULT_OFF; }
        else if (!strcmp(mode, "process")) { cudaintercept_process_disable_v1(1); reason = CUDAINTERCEPT_PROCESS_DISABLED; }
        else if (!strcmp(mode, "operation_disabled")) { cudaintercept_replay_disable_v1(1); reason = CUDAINTERCEPT_OPERATION_DISABLED; }
        else if (!strcmp(mode, "miss")) reason = CUDAINTERCEPT_REGISTRY_MISS;
        else if (!strcmp(mode, "operation")) { call.operation[0] = '?'; reason = CUDAINTERCEPT_OPERATION; }
        else if (!strcmp(mode, "caller")) { call.caller[0] = '?'; reason = CUDAINTERCEPT_CALLER; }
        else if (!strcmp(mode, "generation")) { call.caller_generation++; reason = CUDAINTERCEPT_GENERATION; }
        else if (!strcmp(mode, "context_generation")) { call.context_generation++; reason = CUDAINTERCEPT_GENERATION; }
        else if (!strcmp(mode, "module_generation")) { call.module_generation++; reason = CUDAINTERCEPT_GENERATION; }
        else if (!strcmp(mode, "unknown_generation")) { call.caller_generation = 0; reason = CUDAINTERCEPT_GENERATION; }
        else if (!strcmp(mode, "context")) { call.context++; reason = CUDAINTERCEPT_CONTEXT; }
        else if (!strcmp(mode, "unknown_context")) { call.context = 0; reason = CUDAINTERCEPT_CONTEXT; }
        else if (!strcmp(mode, "function")) { launch_function = (void *)0x9000; reason = CUDAINTERCEPT_REGISTRY_MISS; }
        else if (!strcmp(mode, "declared_function")) { call.function++; reason = CUDAINTERCEPT_REGISTRY_MISS; }
        else if (!strcmp(mode, "stream")) { launch_stream = (void *)0x9000; reason = CUDAINTERCEPT_STREAM; }
        else if (!strcmp(mode, "declared_stream")) { call.stream++; reason = CUDAINTERCEPT_STREAM; }
        else if (!strcmp(mode, "abi")) { call.parameter_bytes[0] = 4; reason = CUDAINTERCEPT_ARGUMENT_ABI; }
        else if (!strcmp(mode, "pointer_intent")) { call.parameter_kinds[0] = 2; reason = CUDAINTERCEPT_ARGUMENT_ABI; }
        else if (!strcmp(mode, "shape")) { call.n = 9; reason = CUDAINTERCEPT_SHAPE; }
        else if (!strcmp(mode, "extent")) { call.bytes[0] = 31; reason = CUDAINTERCEPT_SHAPE; }
        else if (!strcmp(mode, "layout")) { call.layout = 2; reason = CUDAINTERCEPT_LAYOUT; }
        else if (!strcmp(mode, "stride")) { call.strides[1] = 8; reason = CUDAINTERCEPT_LAYOUT; }
        else if (!strcmp(mode, "dtype")) { call.dtype = 2; reason = CUDAINTERCEPT_DTYPE; }
        else if (!strcmp(mode, "alignment")) { call.alignment = 8; reason = CUDAINTERCEPT_ALIGNMENT; }
        else if (!strcmp(mode, "unaligned")) { call.regions[1]++; reason = CUDAINTERCEPT_ALIGNMENT; }
        else if (!strcmp(mode, "regions")) { call.regions[0] = 0x8000; reason = CUDAINTERCEPT_REGIONS; }
        else if (!strcmp(mode, "overlap")) { call.regions[0] = call.regions[1]; reason = CUDAINTERCEPT_REGIONS; }
        else if (!strcmp(mode, "overflow")) { call.regions[0] = UINT64_MAX - 3; reason = CUDAINTERCEPT_REGIONS; }
        else if (!strcmp(mode, "workspace")) { call.workspace_bytes = 4; reason = CUDAINTERCEPT_WORKSPACE; }
        else if (!strcmp(mode, "target")) { call.target[0] = '?'; reason = CUDAINTERCEPT_TARGET; }
        else if (!strcmp(mode, "nvrtc")) { call.nvrtc[0] = '?'; reason = CUDAINTERCEPT_TOOLCHAIN; }
        else if (!strcmp(mode, "builtins")) { call.builtins[0] = '?'; reason = CUDAINTERCEPT_TOOLCHAIN; }
        else if (!strcmp(mode, "flags")) { call.flags[0] = '?'; reason = CUDAINTERCEPT_TOOLCHAIN; }
        else if (!strcmp(mode, "baseline")) { call.baseline[0] = '?'; reason = CUDAINTERCEPT_BASELINE_ARTIFACT; }
        else if (!strcmp(mode, "candidate")) { call.candidate[0] = '?'; reason = CUDAINTERCEPT_CANDIDATE_ARTIFACT; }
        else if (!strcmp(mode, "capture_unknown")) { call.capture = 0; reason = CUDAINTERCEPT_CAPTURE_UNKNOWN; }
        else if (!strcmp(mode, "capture")) { call.capture = 2; reason = CUDAINTERCEPT_CAPTURE; }
        else if (!strcmp(mode, "graph_execution")) {
            struct cudaintercept_replay_v1 wrong = e;
            wrong.capture = 3;
            check(cudaintercept_replay_register_v1(&wrong) == CUDAINTERCEPT_GRAPH_EXECUTION);
            call.capture = 3;
            reason = CUDAINTERCEPT_GRAPH_EXECUTION;
        }
        else if (!strcmp(mode, "unbound_function")) {
            launch_function = (void *)0x9000;
            reason = CUDAINTERCEPT_REGISTRY_MISS;
        }
        else if (!strcmp(mode, "ownership")) { call.caller_owned = 0; reason = CUDAINTERCEPT_OWNERSHIP_UNKNOWN; }
        else if (!strcmp(mode, "population")) { call.population[0] = '?'; reason = CUDAINTERCEPT_POPULATION; }
        else if (!strcmp(mode, "grid")) { launch_grid[0] = 2; reason = CUDAINTERCEPT_GEOMETRY; }
        else if (!strcmp(mode, "block")) { launch_block[1] = 2; reason = CUDAINTERCEPT_GEOMETRY; }
        else if (!strcmp(mode, "shared")) { launch_shared = 4; reason = CUDAINTERCEPT_GEOMETRY; }
        else if (!strcmp(mode, "declared_grid")) { call.grid[0] = 2; reason = CUDAINTERCEPT_GEOMETRY; }
        else if (!strcmp(mode, "params")) { launch_params = NULL; reason = CUDAINTERCEPT_ARGUMENTS; }
        else if (!strcmp(mode, "extra")) { launch_extra = opaque_extra; reason = CUDAINTERCEPT_ARGUMENTS; }
        else if (!strcmp(mode, "bad_registration")) {
            struct cudaintercept_replay_v1 wrong = e;
            wrong.candidate[0] = '?';
            check(cudaintercept_replay_register_v1(&wrong) == CUDAINTERCEPT_CANDIDATE_ARTIFACT);
        } else if (!strcmp(mode, "result_max")) wanted_result = INT_MAX;
        else if (!strcmp(mode, "result_zero")) wanted_result = 0;
        else if (!strcmp(mode, "reentrant")) recursive = 1;
        else if (strcmp(mode, "full") && strcmp(mode, "no_binding") && strcmp(mode, "binding_abi") &&
                 strcmp(mode, "copy") && strcmp(mode, "consumed") && strcmp(mode, "switches")) return 3;
        if (!strcmp(mode, "binding_abi")) {
            call.version++;
            check(cudaintercept_replay_bind_v1(&call) == CUDAINTERCEPT_METADATA_ABI);
            reason = CUDAINTERCEPT_METADATA_ABI;
        } else if (!strcmp(mode, "no_binding")) {
            reason = CUDAINTERCEPT_NO_BINDING;
        } else if (strcmp(mode, "unbound_function")) check(cudaintercept_replay_bind_v1(&call) == 0);
        uint64_t generation = (!strcmp(mode, "no_binding") || !strcmp(mode, "unbound_function") || !strcmp(mode, "binding_abi")) ? 0 : call.caller_generation;
        if (!strcmp(mode, "copy")) memset(&call, 0, sizeof call);
        if (!strcmp(mode, "switches")) {
            check(cudaintercept_replay_enable_v1(2) == -1);
            check(cudaintercept_process_disable_v1(2) == -1);
            check(cudaintercept_replay_disable_v1(2) == -1);
            check(cudaintercept_decisions_enable_v1(2) == -1);
        }
        check(launch() == wanted_result);
        if (recursive) {
            struct cudaintercept_decision_v1 r[2];
            check(cudaintercept_decision_drain_v1(r, 2, NULL) == 2);
            check(r[0].reason == CUDAINTERCEPT_REENTRANT && r[1].reason == CUDAINTERCEPT_ADAPTER_UNQUALIFIED);
            check(r[0].caller_generation == 0 && r[1].caller_generation == 7 && atomic_load(&calls) == 2);
        } else expect(reason, generation, 1);
        if (!strcmp(mode, "consumed") || !strcmp(mode, "default") || !strcmp(mode, "process") || !strcmp(mode, "operation_disabled")) {
            cudaintercept_replay_enable_v1(1);
            cudaintercept_process_disable_v1(0);
            cudaintercept_replay_disable_v1(0);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_NO_BINDING, 0, 2);
            check(cudaintercept_replay_bind_v1(&e) == 0);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_ADAPTER_UNQUALIFIED, 7, 3);
        } else if (!strcmp(mode, "unbound_function")) {
            launch_function = (void *)0x1000;
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_NO_BINDING, 0, 2);
            launch_function = (void *)0x9000;
            call = e;
            call.version++;
            check(cudaintercept_replay_bind_v1(&call) == CUDAINTERCEPT_METADATA_ABI);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_METADATA_ABI, 0, 3);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_REGISTRY_MISS, 0, 4);
            call = e;
            call.operation[0] = '?';
            check(cudaintercept_replay_bind_v1(&call) == 0);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_OPERATION, 7, 5);
        } else if (!strcmp(mode, "graph_execution")) {
            call = e;
            call.capture = 2;
            check(cudaintercept_replay_bind_v1(&call) == 0);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_CAPTURE, 7, 2);
            call = e;
            call.capture = 0;
            check(cudaintercept_replay_bind_v1(&call) == 0);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_CAPTURE_UNKNOWN, 7, 3);
            call = e;
            call.capture = 4;
            check(cudaintercept_replay_bind_v1(&call) == 0);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_CAPTURE, 7, 4);
            check(cudaintercept_replay_bind_v1(&e) == 0);
            check(launch() == wanted_result);
            expect(CUDAINTERCEPT_ADAPTER_UNQUALIFIED, 7, 5);
        }
    }
    printf("mode=%s bad=%d calls=%u\n", mode, atomic_load(&bad), atomic_load(&calls));
    return atomic_load(&bad) ? 1 : 0;
}
