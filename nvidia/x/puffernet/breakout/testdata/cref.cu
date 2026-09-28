// cref.cu runs PufferLib's breakout.cu through its own host API, compiled by
// nvcc as build.sh compiles it, to give the reference the Go package is
// tested and benchmarked against. Build it against a PufferLib 5.0 checkout
// whose raylib is built for this machine:
//
//	R=$PUFFERLIB/raylib-5.5_linux_amd64
//	nvcc -O2 -arch=native -std=c++17 -I$PUFFERLIB -I$PUFFERLIB/src -I$R/include \
//		-Xcompiler=-DPLATFORM_DESKTOP --diag-suppress=2361,111,128 \
//		-o cref testdata/cref.cu $R/lib/libraylib.a -lGL -lm -lpthread
//
// With -DOPT it runs breakout_opt.cu instead.
//
// Usage:
//
//	cref trajectory N STEPS   print a hash of the outputs of reset and each step
//	cref bench N STEPS        print the time per step
//
// Actions come from the same generator as the Go tests.
#ifdef OPT
#include "../breakout_opt.cu"
#else
#include "ocean/breakout/breakout.cu"
#endif

#include <chrono>

static unsigned int seed;

static unsigned int next(void) {
    seed = seed * 1664525u + 1013904223u;
    return seed >> 16;
}

static void actions_for(float* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)(next() % 3);
    }
}

typedef struct Buffers {
    obs_t* obs;
    float* actions;
    float* rewards;
    float* terminals;
} Buffers;

static Env* create(int n, Buffers* b) {
    Dict kw = {0};
    dict_set(&kw, "frameskip", 3);
    dict_set(&kw, "width", 576);
    dict_set(&kw, "height", 330);
    dict_set(&kw, "paddle_width", 62);
    dict_set(&kw, "paddle_height", 8);
    dict_set(&kw, "ball_width", 32);
    dict_set(&kw, "ball_height", 32);
    dict_set(&kw, "brick_width", 32);
    dict_set(&kw, "brick_height", 12);
    dict_set(&kw, "brick_rows", 6);
    dict_set(&kw, "brick_cols", 18);
    dict_set(&kw, "initial_ball_speed", 256);
    dict_set(&kw, "max_ball_speed", 448);
    dict_set(&kw, "paddle_speed", 620);
    dict_set(&kw, "continuous", 0);
    cudaMalloc((void**)&b->obs, (size_t)n * OBS_SIZE * sizeof(obs_t));
    cudaMalloc((void**)&b->actions, (size_t)n * sizeof(float));
    cudaMalloc((void**)&b->rewards, (size_t)n * sizeof(float));
    cudaMalloc((void**)&b->terminals, (size_t)n * sizeof(float));
    return puf_vec_create(n, &kw, b->obs, b->actions, b->rewards, b->terminals);
}

static unsigned long long fnv(unsigned long long h, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void print_hash(int n, Buffers* b, void* obs, float* rewards, float* terminals) {
    cudaMemcpy(obs, b->obs, (size_t)n * OBS_SIZE * sizeof(obs_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(rewards, b->rewards, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(terminals, b->terminals, n * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned long long h = 14695981039346656037ull;
    h = fnv(h, obs, (size_t)n * OBS_SIZE * sizeof(obs_t));
    h = fnv(h, rewards, n * sizeof(float));
    h = fnv(h, terminals, n * sizeof(float));
    printf("%016llx\n", h);
}

static void trajectory(int n, int steps) {
    Buffers b;
    Env* env = create(n, &b);
    void* obs = malloc((size_t)n * OBS_SIZE * sizeof(obs_t));
    float* rewards = (float*)malloc(n * sizeof(float));
    float* terminals = (float*)malloc(n * sizeof(float));
    float* actions = (float*)malloc(n * sizeof(float));
    seed = 1;
    puf_reset(env);
    print_hash(n, &b, obs, rewards, terminals);
    for (int t = 0; t < steps; t++) {
        actions_for(actions, n);
        cudaMemcpy(b.actions, actions, n * sizeof(float), cudaMemcpyHostToDevice);
        puf_step(env);
        print_hash(n, &b, obs, rewards, terminals);
    }
}

static void bench(int n, int steps) {
    Buffers b;
    Env* env = create(n, &b);
    float* actions = (float*)malloc(n * sizeof(float));
    seed = 1;
    actions_for(actions, n);
    cudaMemcpy(b.actions, actions, n * sizeof(float), cudaMemcpyHostToDevice);
    puf_reset(env);
    for (int t = 0; t < 100; t++) {
        puf_step(env);
    }
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < steps; t++) {
        puf_step(env);
    }
    cudaDeviceSynchronize();
    double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    printf("n=%d steps=%d ns/step=%.0f env-steps/s=%.4g\n", n, steps, ns / steps, (double)n * steps / ns * 1e9);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: cref trajectory|bench N STEPS\n");
        return 2;
    }
    int n = atoi(argv[2]);
    int steps = atoi(argv[3]);
    if (strcmp(argv[1], "trajectory") == 0) {
        trajectory(n, steps);
    } else if (strcmp(argv[1], "bench") == 0) {
        bench(n, steps);
    } else {
        fprintf(stderr, "usage: cref trajectory|bench N STEPS\n");
        return 2;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cref: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
