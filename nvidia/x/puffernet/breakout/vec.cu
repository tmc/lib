// vec.cu compiles PufferLib's breakout.cu, unchanged, with NVRTC and adds
// the kernels that stand in for its host code. With BREAKOUT_OPT defined it
// compiles breakout_opt.cu instead.
//
// NVRTC rejects host functions, so it compiles with -default-device, which
// makes unannotated functions device functions, and __host__ is defined away
// for the one function breakout.cu marks as host only. The shim headers turn
// the calls the host code makes into no-ops, except for dict_get and exit,
// which breakout_config needs.
#undef __host__
#define __host__
#ifdef BREAKOUT_OPT
#include "breakout_opt.cu"
#else
#include "breakout.cu"
#endif

// breakout_config fills cfg from the n items of kwargs, the [env] section of
// config/breakout.ini, as puf_vec_create does on the host. It sets exit_code
// if it rejects them.
extern "C" __global__ void breakout_config(DictItem* kwargs, int n, GpuBreakoutConfig* cfg) {
    Dict d = {kwargs, n};
    GpuBreakoutConfig c = {};
    gpu_breakout_host_fill_config(&c, &d);
    *cfg = c;
}

// breakout_init does what puf_vec_create does to each env on the host.
// breakout.cu keeps all of an env's state in its Env and ignores state.
extern "C" __global__ void breakout_init(Env* envs, uint4* state, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    Env e = {};
    e.num_agents = 1;
#ifdef BREAKOUT_OPT
    for (int k = 0; k < BREAKOUT_STATE_GROUPS; k++) {
        state[k * n + i] = make_uint4(0, 0, 0, 0);
    }
    state[BREAKOUT_BOUNCE * n + i].w = (unsigned int)(i + 1); // rng
    state[BREAKOUT_ROUND * n + i].w = 0xffffu; // num_balls = -1
#else
    e.rng = (unsigned int)(i + 1);
    e.num_balls = -1;
#endif
    envs[i] = e;
}

// breakout_log adds the log of every Env to sum and clears it.
extern "C" __global__ void breakout_log(Env* envs, int n, Log* sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    Log* l = &envs[i].log;
    if (l->n == 0.0f) {
        return;
    }
    atomicAdd(&sum->perf, l->perf);
    atomicAdd(&sum->score, l->score);
    atomicAdd(&sum->episode_return, l->episode_return);
    atomicAdd(&sum->episode_length, l->episode_length);
    atomicAdd(&sum->n, l->n);
    *l = Log{};
}
