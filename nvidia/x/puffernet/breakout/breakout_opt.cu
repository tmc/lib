// CUDA Breakout, optimized. A drop-in replacement for PufferLib's
// ocean/breakout/breakout.cu with the same host API, physics and outputs:
// every step writes the same observations, rewards and terminals, bit for
// bit. What differs is how the kernels do it.
//
// breakout.cu gives each env 8 threads. Lane 0 runs the physics in place on
// the Env in global memory while the other 7 wait, then all 8 write the
// observations. A warp holds 4 envs, most of its threads idle, and every
// state access is a dependent global load or store. Here:
//
//   - One thread runs one env, so a warp steps 32 envs.
//   - The physics state lives apart from Env in four arrays of 16-byte
//     groups, 64 bytes per env. A thread loads its groups into registers
//     with one 16-byte load each and stores them back once, so a warp's
//     access to a group is four whole 128-byte lines. Groups are ordered by
//     how often they change, and a group that did not change is not stored.
//     The step kernel keeps all of it in registers, with no local memory
//     but the frame sinf and cosf use for huge arguments.
//   - Env keeps only what the trainer reads (the log, num_agents) and the
//     fields it names in code shared with CPU envs. The log is written only
//     when an episode ends.
//   - Each thread stages its scalar observations and brick mask in shared
//     memory, and the warp writes the 32 envs' observations, which are
//     contiguous, with consecutive 4-byte stores.
//
// It writes the observations breakout.cu writes and no others. breakout.cu
// never writes observations 8 and 9 (num_balls and paddle_width): its
// writer gives the 10 scalars to lanes 0 to 9 of an env, which has only 8.
// Nor does it write the slots past the last brick.
#ifndef PUFFER_BREAKOUT_GPU_CU
#define PUFFER_BREAKOUT_GPU_CU

// Device Env* batch backend (see pufferenv.h PUF_BACKEND).
#define PUF_BACKEND PUF_GPU

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Fixed env obs dtype (not tied to train precision build flags).
typedef __nv_bfloat16 obs_t;
#include "pufferenv.h"

#define ACT_SIZES {3}
#define NUM_ATNS 1
#define OBS_SIZE 118
#define HALF_PADDLE_WIDTH 31
#define Y_OFFSET 50
#define TICK_RATE (1.0f / 60.0f)
#define LEFT 1
#define RIGHT 2
#define BRICK_INDEX_NO_COLLISION -4
#define BRICK_INDEX_SIDEWALL_COLLISION -3
#define BRICK_INDEX_BACKWALL_COLLISION -2
#define BRICK_INDEX_PADDLE_COLLISION -1

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

#define BREAKOUT_GPU_MAX_BRICKS 128
#define BREAKOUT_GPU_BRICK_WORDS ((BREAKOUT_GPU_MAX_BRICKS + 31) / 32)
#define BREAKOUT_GPU_PI 3.14159265358979323846f
#define BREAKOUT_BLOCK 256
static int breakout_grid(int n) {
    return (n + BREAKOUT_BLOCK - 1) / BREAKOUT_BLOCK;
}

// Shared config (constant mem).
typedef struct GpuBreakoutConfig {
    int width;
    int height;
    int brick_width;
    int brick_height;
    int brick_rows;
    int brick_cols;
    int num_bricks;
    int ball_width;
    int ball_height;
    float paddle_height;
    float paddle_speed;
    float initial_paddle_width;
    float initial_ball_speed;
    float max_ball_speed;
    int max_score;
    int half_max_score;
    int frameskip;
    int continuous;
} GpuBreakoutConfig;

// Per-env trainer-facing record. Log first for the trainer's log reduce.
// The physics state is in BreakoutState.
typedef struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
} Env;

// The physics state of n envs is BREAKOUT_STATE_GROUPS arrays of n uint4:
// group g of env i is state[g * n + i].
enum {
    BREAKOUT_MOTION, // paddle_x, ball_x, ball_y, tick: change every step
    BREAKOUT_BOUNCE, // ball_vx, ball_vy, hits, rng: change on a bounce or launch
    BREAKOUT_ROUND,  // paddle_width, ball_speed, score, num_balls | balls_fired<<16
    BREAKOUT_BRICKS, // brick_mask
    BREAKOUT_STATE_GROUPS,
};
static_assert(BREAKOUT_GPU_MAX_BRICKS <= 128, "brick mask is one uint4");

// The physics state of one env, in registers. Field names match
// breakout.cu's Env so the physics below reads as it does there.
// paddle_y is not stored: after the first reset it is always the same.
typedef struct BreakoutState {
    float paddle_x;
    float paddle_y;
    float ball_x;
    float ball_y;
    float ball_vx;
    float ball_vy;
    float paddle_width;
    float ball_speed;
    int score;
    int balls_fired;
    int hits;
    int num_balls;
    int tick;
    unsigned int rng;
    // Bit set = destroyed; clear = alive. Bricks 0-63 are in bricks_lo,
    // 64-127 in bricks_hi. Two named words rather than an array: nvcc turns
    // any lookup into an array, even an unrolled select, back into an
    // indexed load, which puts the whole state in local memory.
    unsigned long long bricks_lo;
    unsigned long long bricks_hi;
} BreakoutState;

typedef struct GpuBreakoutCollisionInfo {
    float t;
    float overlap;
    float x;
    float y;
    float vx;
    float vy;
    int brick_index;
} GpuBreakoutCollisionInfo;

__constant__ GpuBreakoutConfig d_bcfg;

// ---- brick bitset helpers ----
__device__ static inline int gpu_brick_alive(const BreakoutState* env, int i) {
    unsigned long long w = i < 64 ? env->bricks_lo : env->bricks_hi;
    return ((w >> (i & 63)) & 1ull) == 0ull;
}

__device__ static inline void gpu_brick_destroy(BreakoutState* env, int i) {
    unsigned long long bit = 1ull << (i & 63);
    if (i < 64) {
        env->bricks_lo |= bit;
    } else {
        env->bricks_hi |= bit;
    }
}

__device__ static inline void gpu_brick_clear_all(BreakoutState* env) {
    env->bricks_lo = 0ull;
    env->bricks_hi = 0ull;
}

static int gpu_breakout_compute_half_max_score(int brick_rows, int brick_cols) {
    int half = 0;
    for (int row = 0; row < brick_rows; row++) {
        for (int col = 0; col < brick_cols; col++) {
            int idx = row * brick_cols + col;
            half += 7 - 3 * (idx / brick_cols / 2);
        }
    }
    return half;
}

// Cheap xorshift32 — only used for a coin flip on ball launch.
__device__ static inline unsigned int gpu_breakout_xorshift(unsigned int* seed) {
    unsigned int x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *seed = x ? x : 0xA341316Cu; // never stick at 0
    return *seed;
}

__device__ static inline void gpu_breakout_add_log(const BreakoutState* env, Log* log) {
    log->episode_length += env->tick;
    log->episode_return += env->score;
    log->score += env->score;
    log->perf += env->score / (float)d_bcfg.max_score;
    log->n += 1.0f;
}

__device__ static inline bool gpu_breakout_calc_vline_collision(float xw, float yw, float hw,
        float x, float y, float vx, float vy, float h, GpuBreakoutCollisionInfo* col) {
    float t_new = (xw - x) / vx;
    float topmost = fminf(yw + hw, y + h + vy * t_new);
    float botmost = fmaxf(yw, y + vy * t_new);
    float overlap_new = topmost - botmost;

    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f &&
            (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = xw;
        col->y = y + vy * t_new;
        col->vx = -vx;
        col->vy = vy;
        return true;
    }
    return false;
}

__device__ static inline bool gpu_breakout_calc_hline_collision(float xw, float yw, float ww,
        float x, float y, float vx, float vy, float w, GpuBreakoutCollisionInfo* col) {
    float t_new = (yw - y) / vy;
    float rightmost = fminf(xw + ww, x + w + vx * t_new);
    float leftmost = fmaxf(xw, x + vx * t_new);
    float overlap_new = rightmost - leftmost;

    if (overlap_new > 0.0f && t_new > 0.0f && t_new <= 1.0f &&
            (t_new < col->t || (t_new == col->t && overlap_new > col->overlap))) {
        col->t = t_new;
        col->overlap = overlap_new;
        col->x = x + vx * t_new;
        col->y = yw;
        col->vx = vx;
        col->vy = -vy;
        return true;
    }
    return false;
}

__device__ static inline void gpu_breakout_calc_brick_collision(
        float brick_x, float brick_y, float bw, float bh,
        float ball_x, float ball_y, float ball_vx, float ball_vy,
        float ball_w, float ball_h, int idx,
        GpuBreakoutCollisionInfo* collision_info) {
    bool collision = false;

    if (ball_vx > 0.0f) {
        if (gpu_breakout_calc_vline_collision(brick_x, brick_y, bh,
                ball_x + ball_w, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision = true;
            collision_info->x -= ball_w;
        }
    } else if (ball_vx < 0.0f) {
        if (gpu_breakout_calc_vline_collision(brick_x + bw, brick_y, bh,
                ball_x, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision = true;
        }
    }

    if (ball_vy > 0.0f) {
        if (gpu_breakout_calc_hline_collision(brick_x, brick_y, bw,
                ball_x, ball_y + ball_h, ball_vx, ball_vy, ball_w, collision_info)) {
            collision = true;
            collision_info->y -= ball_h;
        }
    } else if (ball_vy < 0.0f) {
        if (gpu_breakout_calc_hline_collision(brick_x, brick_y + bh, bw,
                ball_x, ball_y, ball_vx, ball_vy, ball_w, collision_info)) {
            collision = true;
        }
    }
    if (collision) {
        collision_info->brick_index = idx;
    }
}

__device__ static inline void gpu_breakout_calc_all_brick_collisions(BreakoutState* env,
        GpuBreakoutCollisionInfo* collision_info) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float ball_x = env->ball_x;
    float ball_vx = env->ball_vx;
    float ball_y = env->ball_y;
    float ball_vy = env->ball_vy;
    float ball_w = (float)c->ball_width;
    float ball_h = (float)c->ball_height;
    float ball_x_dst = ball_x + ball_vx;
    float ball_y_dst = ball_y + ball_vy;
    int bw = c->brick_width;
    int bh = c->brick_height;
    int rows = c->brick_rows;
    int cols = c->brick_cols;

    // Early-out: ball fully below the brick field and moving down can't hit bricks.
    float brick_bottom = (float)(Y_OFFSET + rows * bh);
    float ball_top = fminf(ball_y, ball_y_dst);
    if (ball_top >= brick_bottom) {
        return;
    }

    int row_from = (int)((fminf(ball_y, ball_y_dst) - Y_OFFSET) / (float)bh);
    if (row_from < 0) row_from = 0;
    if (row_from >= rows) return;

    int column_from = (int)(fminf(ball_x, ball_x_dst) / (float)bw);
    if (column_from < 0) column_from = 0;

    float ball_x_end = ball_x + ball_w;
    float ball_x_dst_end = ball_x_dst + ball_w;
    int column_to = (int)(fmaxf(ball_x_end, ball_x_dst_end) / (float)bw);
    if (column_to >= cols) column_to = cols - 1;

    float ball_y_end = ball_y + ball_h;
    float ball_y_dst_end = ball_y_dst + ball_h;
    int row_to = (int)((fmaxf(ball_y_end, ball_y_dst_end) - Y_OFFSET) / (float)bh);
    if (row_to >= rows) row_to = rows - 1;

    float bwf = (float)bw;
    float bhf = (float)bh;

    for (int row = row_from; row <= row_to; row++) {
        float brick_y = (float)(row * bh + Y_OFFSET);
        int row_base = row * cols;
        for (int column = column_from; column <= column_to; column++) {
            int brick_index = row_base + column;
            if (gpu_brick_alive(env, brick_index)) {
                float brick_x = (float)(column * bw);
                gpu_breakout_calc_brick_collision(
                    brick_x, brick_y, bwf, bhf,
                    ball_x, ball_y, ball_vx, ball_vy, ball_w, ball_h,
                    brick_index, collision_info);
            }
        }
    }
}

__device__ static inline bool gpu_breakout_calc_paddle_ball_collisions(BreakoutState* env,
        GpuBreakoutCollisionInfo* collision_info) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float base_angle = BREAKOUT_GPU_PI / 4.0f;
    float ball_h = (float)c->ball_height;
    float ball_w = (float)c->ball_width;

    if (env->ball_y + ball_h + env->ball_vy < env->paddle_y) {
        return false;
    }

    if (!gpu_breakout_calc_hline_collision(env->paddle_x, env->paddle_y, env->paddle_width,
            env->ball_x, env->ball_y + ball_h, env->ball_vx, env->ball_vy,
            ball_w, collision_info) || collision_info->t > 1.0f) {
        return false;
    }

    collision_info->y -= ball_h;
    collision_info->brick_index = BRICK_INDEX_PADDLE_COLLISION;

    float relative_intersection =
        ((env->ball_x + ball_w * 0.5f) - env->paddle_x) / env->paddle_width;
    float angle = -base_angle + relative_intersection * 2.0f * base_angle;
    float speed_tick = env->ball_speed * TICK_RATE;
    env->ball_vx = sinf(angle) * speed_tick;
    env->ball_vy = -cosf(angle) * speed_tick;
    env->hits += 1;
    if (env->hits % 4 == 0 && env->ball_speed < c->max_ball_speed) {
        env->ball_speed += 64;
    }
    if (env->score == c->half_max_score) {
        gpu_brick_clear_all(env);
    }
    return true;
}

__device__ static inline void gpu_breakout_calc_all_wall_collisions(BreakoutState* env,
        GpuBreakoutCollisionInfo* collision_info) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float ball_x = env->ball_x;
    float ball_y = env->ball_y;
    float ball_vx = env->ball_vx;
    float ball_vy = env->ball_vy;
    float ball_w = (float)c->ball_width;
    float ball_h = (float)c->ball_height;
    float width = (float)c->width;
    float height = (float)c->height;

    if (ball_vx < 0.0f) {
        if (gpu_breakout_calc_vline_collision(0.0f, 0.0f, height,
                ball_x, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    } else if (ball_vx > 0.0f) {
        if (gpu_breakout_calc_vline_collision(width, 0.0f, height,
                ball_x + ball_w, ball_y, ball_vx, ball_vy, ball_h, collision_info)) {
            collision_info->x -= ball_w;
            collision_info->brick_index = BRICK_INDEX_SIDEWALL_COLLISION;
        }
    }
    if (ball_vy < 0.0f) {
        if (gpu_breakout_calc_hline_collision(0.0f, 0.0f, width,
                ball_x, ball_y, ball_vx, ball_vy, ball_w, collision_info)) {
            collision_info->brick_index = BRICK_INDEX_BACKWALL_COLLISION;
        }
    }
}

__device__ static inline void gpu_breakout_check_wall_bounds(BreakoutState* env) {
    float offset = d_bcfg.max_ball_speed * 1.1f * TICK_RATE;
    float width = (float)d_bcfg.width;
    if (env->ball_x < 0.0f) {
        env->ball_x += offset;
    }
    if (env->ball_x > width) {
        env->ball_x -= offset;
    }
    if (env->ball_y < 0.0f) {
        env->ball_y += offset;
    }
}

__device__ static inline void gpu_breakout_destroy_brick(BreakoutState* env, int brick_idx,
        float* reward) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float gained_points = 7 - 3 * ((brick_idx / c->brick_cols) / 2);

    env->score += (int)gained_points;
    gpu_brick_destroy(env, brick_idx);
    *reward += gained_points;

    if (brick_idx / c->brick_cols < 3) {
        env->ball_speed = c->max_ball_speed;
    }
}

__device__ static inline bool gpu_breakout_handle_collisions(BreakoutState* env, float* reward) {
    GpuBreakoutCollisionInfo collision_info = {
        .t = 2.0f,
        .overlap = -1.0f,
        .x = 0.0f,
        .y = 0.0f,
        .vx = 0.0f,
        .vy = 0.0f,
        .brick_index = BRICK_INDEX_NO_COLLISION,
    };

    gpu_breakout_check_wall_bounds(env);

    gpu_breakout_calc_all_brick_collisions(env, &collision_info);
    gpu_breakout_calc_all_wall_collisions(env, &collision_info);
    gpu_breakout_calc_paddle_ball_collisions(env, &collision_info);
    if (collision_info.brick_index != BRICK_INDEX_PADDLE_COLLISION && collision_info.t <= 1.0f) {
        env->ball_x = collision_info.x;
        env->ball_y = collision_info.y;
        env->ball_vx = collision_info.vx;
        env->ball_vy = collision_info.vy;
        if (collision_info.brick_index >= 0) {
            gpu_breakout_destroy_brick(env, collision_info.brick_index, reward);
        }
        if (collision_info.brick_index == BRICK_INDEX_BACKWALL_COLLISION) {
            env->paddle_width = HALF_PADDLE_WIDTH;
        }
    }
    return collision_info.brick_index != BRICK_INDEX_NO_COLLISION;
}

__device__ static inline void gpu_breakout_reset_round(BreakoutState* env) {
    const GpuBreakoutConfig* c = &d_bcfg;
    env->balls_fired = 0;
    env->hits = 0;
    env->ball_speed = c->initial_ball_speed;
    env->paddle_width = c->initial_paddle_width;

    env->paddle_x = c->width / 2.0f - env->paddle_width / 2.0f;
    env->paddle_y = c->height - c->paddle_height - 10;

    env->ball_x = env->paddle_x + (env->paddle_width / 2.0f - c->ball_width / 2.0f);
    env->ball_y = c->height / 2.0f - 30;

    env->ball_vx = 0.0f;
    env->ball_vy = 0.0f;
}

__device__ static inline void gpu_breakout_reset_state(BreakoutState* env) {
    env->score = 0;
    env->num_balls = 5;
    gpu_brick_clear_all(env);
    gpu_breakout_reset_round(env);
    env->tick = 0;
}

__device__ static inline void gpu_breakout_step_frame(BreakoutState* env, Log* log,
        float action, float* reward, float* terminal) {
    const GpuBreakoutConfig* c = &d_bcfg;
    float act = 0.0f;
    if (env->balls_fired == 0) {
        env->balls_fired = 1;
        float direction = BREAKOUT_GPU_PI / 3.25f;
        float speed_tick = env->ball_speed * TICK_RATE;

        env->ball_vy = cosf(direction) * speed_tick;
        env->ball_vx = sinf(direction) * speed_tick;
        if ((gpu_breakout_xorshift(&env->rng) & 1u) == 0u) {
            env->ball_vx = -env->ball_vx;
        }
    } else if (action == LEFT) {
        act = -1.0f;
    } else if (action == RIGHT) {
        act = 1.0f;
    }
    if (c->continuous) {
        act = action;
    }
    env->paddle_x = fminf((float)c->width - env->paddle_width,
        fmaxf(0.0f, env->paddle_x + act * c->paddle_speed * TICK_RATE));

    if (!gpu_breakout_handle_collisions(env, reward)) {
        env->ball_x += env->ball_vx;
        env->ball_y += env->ball_vy;
    }

    if (env->ball_y >= env->paddle_y + c->paddle_height) {
        env->num_balls -= 1;
        gpu_breakout_reset_round(env);
    }
    if (env->num_balls < 0 || env->score == c->max_score) {
        *terminal = 1.0f;
        gpu_breakout_add_log(env, log);
        gpu_breakout_reset_state(env);
    }
}

// ---- state load/store ----

__device__ static inline void gpu_breakout_unpack(BreakoutState* s, const uint4* g) {
    const GpuBreakoutConfig* c = &d_bcfg;
    s->paddle_x = __uint_as_float(g[BREAKOUT_MOTION].x);
    s->ball_x = __uint_as_float(g[BREAKOUT_MOTION].y);
    s->ball_y = __uint_as_float(g[BREAKOUT_MOTION].z);
    s->tick = (int)g[BREAKOUT_MOTION].w;
    s->ball_vx = __uint_as_float(g[BREAKOUT_BOUNCE].x);
    s->ball_vy = __uint_as_float(g[BREAKOUT_BOUNCE].y);
    s->hits = (int)g[BREAKOUT_BOUNCE].z;
    s->rng = g[BREAKOUT_BOUNCE].w;
    s->paddle_width = __uint_as_float(g[BREAKOUT_ROUND].x);
    s->ball_speed = __uint_as_float(g[BREAKOUT_ROUND].y);
    s->score = (int)g[BREAKOUT_ROUND].z;
    s->num_balls = (short)(g[BREAKOUT_ROUND].w & 0xffffu);
    s->balls_fired = (int)(g[BREAKOUT_ROUND].w >> 16);
    s->bricks_lo = (unsigned long long)g[BREAKOUT_BRICKS].y << 32 | g[BREAKOUT_BRICKS].x;
    s->bricks_hi = (unsigned long long)g[BREAKOUT_BRICKS].w << 32 | g[BREAKOUT_BRICKS].z;
    s->paddle_y = c->height - c->paddle_height - 10;
}

__device__ static inline void gpu_breakout_pack(uint4* g, const BreakoutState* s) {
    g[BREAKOUT_MOTION] = make_uint4(__float_as_uint(s->paddle_x), __float_as_uint(s->ball_x),
        __float_as_uint(s->ball_y), (unsigned int)s->tick);
    g[BREAKOUT_BOUNCE] = make_uint4(__float_as_uint(s->ball_vx), __float_as_uint(s->ball_vy),
        (unsigned int)s->hits, s->rng);
    g[BREAKOUT_ROUND] = make_uint4(__float_as_uint(s->paddle_width), __float_as_uint(s->ball_speed),
        (unsigned int)s->score, ((unsigned int)s->num_balls & 0xffffu) | (unsigned int)s->balls_fired << 16);
    g[BREAKOUT_BRICKS] = make_uint4((unsigned int)s->bricks_lo, (unsigned int)(s->bricks_lo >> 32),
        (unsigned int)s->bricks_hi, (unsigned int)(s->bricks_hi >> 32));
}

__device__ static inline void gpu_breakout_load(uint4* g, const uint4* __restrict__ state, int i, int n) {
    #pragma unroll
    for (int k = 0; k < BREAKOUT_STATE_GROUPS; k++) {
        g[k] = state[k * n + i];
    }
}

// gpu_breakout_store stores the groups of s that changed. It rereads the
// old groups, which are still in L1, rather than keep them in registers
// through the step.
__device__ static inline void gpu_breakout_store(uint4* __restrict__ state,
        const BreakoutState* s, int i, int n) {
    uint4 g[BREAKOUT_STATE_GROUPS];
    gpu_breakout_pack(g, s);
    #pragma unroll
    for (int k = 0; k < BREAKOUT_STATE_GROUPS; k++) {
        uint4 old = state[k * n + i];
        if (g[k].x != old.x || g[k].y != old.y || g[k].z != old.z || g[k].w != old.w) {
            state[k * n + i] = g[k];
        }
    }
}

// ---- observations ----

#define BREAKOUT_OBS_WORDS (OBS_SIZE / 2)
#define BREAKOUT_SCALAR_WORDS 4 // observations 0 to 7; 8 and 9 are not written
#define BREAKOUT_BRICK_OBS_WORD 5
#define BREAKOUT_STAGE_WORDS (BREAKOUT_SCALAR_WORDS + BREAKOUT_GPU_BRICK_WORDS)
#define BREAKOUT_STAGE_STRIDE (BREAKOUT_STAGE_WORDS | 1) // odd: no bank conflicts

__device__ static inline unsigned int gpu_breakout_obs_pair(float lo, float hi) {
    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(lo)) |
        (unsigned int)__bfloat16_as_ushort(__float2bfloat16(hi)) << 16;
}

// gpu_breakout_stage writes env's scalar observations, in bfloat16 pairs,
// and its brick mask to stage.
__device__ static inline void gpu_breakout_stage(const BreakoutState* env, unsigned int* stage) {
    const GpuBreakoutConfig* c = &d_bcfg;
    stage[0] = gpu_breakout_obs_pair(env->paddle_x / (float)c->width, env->paddle_y / (float)c->height);
    stage[1] = gpu_breakout_obs_pair(env->ball_x / (float)c->width, env->ball_y / (float)c->height);
    stage[2] = gpu_breakout_obs_pair(env->ball_vx / 512.0f, env->ball_vy / 512.0f);
    stage[3] = gpu_breakout_obs_pair(env->balls_fired / 5.0f, env->score / 864.0f);
    stage[BREAKOUT_SCALAR_WORDS + 0] = (unsigned int)env->bricks_lo;
    stage[BREAKOUT_SCALAR_WORDS + 1] = (unsigned int)(env->bricks_lo >> 32);
    stage[BREAKOUT_SCALAR_WORDS + 2] = (unsigned int)env->bricks_hi;
    stage[BREAKOUT_SCALAR_WORDS + 3] = (unsigned int)(env->bricks_hi >> 32);
}

// gpu_breakout_write_obs writes the observations of the warp's envs, staged
// by gpu_breakout_stage in stage[lane], to obs. The envs' observations are
// contiguous, so lane l writes words l, l+32, l+64 and so on of them.
__device__ static inline void gpu_breakout_write_obs(
        const unsigned int (*stage)[BREAKOUT_STAGE_STRIDE], obs_t* obs, int first, int n) {
    int lane = threadIdx.x & 31;
    int count = min(32, n - first);
    if (count <= 0) {
        return;
    }
    unsigned int* out = (unsigned int*)obs + (long)first * BREAKOUT_OBS_WORDS;
    int total = count * BREAKOUT_OBS_WORDS;
    int num_bricks = d_bcfg.num_bricks;
    for (int w = lane; w < total; w += 32) {
        int e = w / BREAKOUT_OBS_WORDS;
        int j = w - e * BREAKOUT_OBS_WORDS;
        if (j < BREAKOUT_SCALAR_WORDS) {
            out[w] = stage[e][j];
            continue;
        }
        // Bricks b and b+1: 1.0 (0x3f80) if destroyed, else 0.
        int b = 2 * (j - BREAKOUT_BRICK_OBS_WORD);
        if (b < 0 || b >= num_bricks) {
            continue;
        }
        unsigned int m = stage[e][BREAKOUT_SCALAR_WORDS + (b >> 5)] >> (b & 31);
        unsigned int lo = (m & 1u) ? 0x3f80u : 0u;
        if (b + 1 < num_bricks) {
            out[w] = lo | ((m & 2u) ? 0x3f800000u : 0u);
        } else {
            *(unsigned short*)&out[w] = (unsigned short)lo;
        }
    }
}

// ---- kernels ----

__global__ void __launch_bounds__(BREAKOUT_BLOCK) gpu_breakout_reset_kernel(
        Env* __restrict__ envs, uint4* __restrict__ state, obs_t* __restrict__ observations,
        float* __restrict__ rewards, float* __restrict__ terminals, int num_envs) {
    // No early return: the whole warp writes the observations.
    __shared__ unsigned int stage[BREAKOUT_BLOCK][BREAKOUT_STAGE_STRIDE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_envs) {
        BreakoutState s;
        {
            uint4 g[BREAKOUT_STATE_GROUPS];
            gpu_breakout_load(g, state, i, num_envs);
            gpu_breakout_unpack(&s, g);
        }
        gpu_breakout_reset_state(&s);
        rewards[i] = 0.0f;
        terminals[i] = 0.0f;
        gpu_breakout_store(state, &s, i, num_envs);
        gpu_breakout_stage(&s, stage[threadIdx.x]);
    }
    __syncwarp();
    gpu_breakout_write_obs(&stage[threadIdx.x & ~31], observations, i - (threadIdx.x & 31), num_envs);
}

__global__ void __launch_bounds__(BREAKOUT_BLOCK) gpu_breakout_step_kernel(
        Env* __restrict__ envs, uint4* __restrict__ state,
        const float* __restrict__ actions, obs_t* __restrict__ observations,
        float* __restrict__ rewards, float* __restrict__ terminals, int num_envs) {
    // No early return: the whole warp writes the observations.
    __shared__ unsigned int stage[BREAKOUT_BLOCK][BREAKOUT_STAGE_STRIDE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_envs) {
        BreakoutState s;
        {
            uint4 g[BREAKOUT_STATE_GROUPS];
            gpu_breakout_load(g, state, i, num_envs);
            gpu_breakout_unpack(&s, g);
        }
        float reward = 0.0f;
        float terminal = 0.0f;
        float action = actions[(long)i * NUM_ATNS];
        int frameskip = d_bcfg.frameskip;
        for (int f = 0; f < frameskip; f++) {
            s.tick += 1;
            gpu_breakout_step_frame(&s, &envs[i].log, action, &reward, &terminal);
        }
        rewards[i] = reward;
        terminals[i] = terminal;
        gpu_breakout_store(state, &s, i, num_envs);
        gpu_breakout_stage(&s, stage[threadIdx.x]);
    }
    __syncwarp();
    gpu_breakout_write_obs(&stage[threadIdx.x & ~31], observations, i - (threadIdx.x & 31), num_envs);
}


static void gpu_breakout_host_fill_config(GpuBreakoutConfig* cfg, Dict* kwargs) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->frameskip = dict_get(kwargs, "frameskip");
    cfg->width = dict_get(kwargs, "width");
    cfg->height = dict_get(kwargs, "height");
    cfg->initial_paddle_width = dict_get(kwargs, "paddle_width");
    cfg->paddle_height = dict_get(kwargs, "paddle_height");
    cfg->ball_width = dict_get(kwargs, "ball_width");
    cfg->ball_height = dict_get(kwargs, "ball_height");
    cfg->brick_width = dict_get(kwargs, "brick_width");
    cfg->brick_height = dict_get(kwargs, "brick_height");
    cfg->brick_rows = dict_get(kwargs, "brick_rows");
    cfg->brick_cols = dict_get(kwargs, "brick_cols");
    cfg->initial_ball_speed = dict_get(kwargs, "initial_ball_speed");
    cfg->max_ball_speed = dict_get(kwargs, "max_ball_speed");
    cfg->paddle_speed = dict_get(kwargs, "paddle_speed");
    cfg->continuous = dict_get(kwargs, "continuous");
    cfg->num_bricks = cfg->brick_rows * cfg->brick_cols;
    if (cfg->num_bricks <= 0 || cfg->num_bricks > BREAKOUT_GPU_MAX_BRICKS ||
            cfg->num_bricks > OBS_SIZE - 10) {
        fprintf(stderr, "Breakout GPU env supports 1..%d bricks and OBS_SIZE-10 slots; got %d\n",
            BREAKOUT_GPU_MAX_BRICKS, cfg->num_bricks);
        exit(1);
    }
    cfg->half_max_score = gpu_breakout_compute_half_max_score(cfg->brick_rows, cfg->brick_cols);
    cfg->max_score = 2 * cfg->half_max_score;
}

// Bound EnvBuf + device Env array. Same puf_reset/step/close(Env*) sig as CPU:
// Env* is the device base; step/reset always run the full batch (env handles the loop).
typedef struct BreakoutClient {
    int width;
    int height;
    Texture2D ball;
} BreakoutClient;

static struct {
    Env* envs;
    uint4* state;
    int n;
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    cudaStream_t stream;
    GpuBreakoutConfig cfg;
    BreakoutClient* client;
} g_gpu;

static Color BREAKOUT_BRICK_COLORS[6] = {
    RED, ORANGE, YELLOW, GREEN, SKYBLUE, BLUE,
};

// Trainer create: returns device Env base (also stored in g_gpu).
Env* puf_vec_create(int n, Dict* env_kwargs,
        obs_t* observations, float* actions, float* rewards, float* terminals) {
    gpu_breakout_host_fill_config(&g_gpu.cfg, env_kwargs);
    cudaMemcpyToSymbol(d_bcfg, &g_gpu.cfg, sizeof(GpuBreakoutConfig));

    Env* host_envs = (Env*)calloc(n, sizeof(Env));
    uint4* host_state = (uint4*)calloc((size_t)n * BREAKOUT_STATE_GROUPS, sizeof(uint4));
    for (int i = 0; i < n; i++) {
        host_envs[i].num_agents = 1;
        host_state[BREAKOUT_BOUNCE * n + i].w = (unsigned int)(i + 1); // rng
        host_state[BREAKOUT_ROUND * n + i].w = 0xffffu; // num_balls = -1
    }
    Env* envs = NULL;
    uint4* state = NULL;
    cudaMalloc((void**)&envs, n * sizeof(Env));
    cudaMalloc((void**)&state, (size_t)n * BREAKOUT_STATE_GROUPS * sizeof(uint4));
    cudaMemcpy(envs, host_envs, n * sizeof(Env), cudaMemcpyHostToDevice);
    cudaMemcpy(state, host_state, (size_t)n * BREAKOUT_STATE_GROUPS * sizeof(uint4),
        cudaMemcpyHostToDevice);
    free(host_envs);
    free(host_state);

    g_gpu.envs = envs;
    g_gpu.state = state;
    g_gpu.n = n;
    g_gpu.observations = observations;
    g_gpu.actions = actions;
    g_gpu.rewards = rewards;
    g_gpu.terminals = terminals;
    g_gpu.stream = 0;
    g_gpu.client = NULL;
    return envs;
}

void puf_bind_stream(cudaStream_t stream) {
    g_gpu.stream = stream;
}

// Same signatures as CPU. GPU path uses batch base; create fills state.
void puf_init(Env* env, Dict*) {
    env->num_agents = 1;
}

void puf_reset(Env* env) {
    gpu_breakout_reset_kernel<<<breakout_grid(g_gpu.n), BREAKOUT_BLOCK>>>(
        g_gpu.envs, g_gpu.state, g_gpu.observations, g_gpu.rewards, g_gpu.terminals, g_gpu.n);
}

void puf_step(Env* env) {
    gpu_breakout_step_kernel<<<breakout_grid(g_gpu.n), BREAKOUT_BLOCK, 0, g_gpu.stream>>>(
        g_gpu.envs, g_gpu.state, g_gpu.actions, g_gpu.observations, g_gpu.rewards,
        g_gpu.terminals, g_gpu.n);
}

void puf_close(Env* env) {
    if (g_gpu.client) {
        UnloadTexture(g_gpu.client->ball);
        CloseWindow();
        free(g_gpu.client);
        g_gpu.client = NULL;
    }
    cudaFree(g_gpu.envs);
    cudaFree(g_gpu.state);
    g_gpu.envs = NULL;
    g_gpu.state = NULL;
}

// D2H env 0 and draw (same layout as breakout.h). env is device batch base.
void puf_render(Env* env) {
    if (!g_gpu.envs || g_gpu.n < 1) {
        return;
    }
    if (g_gpu.stream) {
        cudaStreamSynchronize(g_gpu.stream);
    }
    uint4 g[BREAKOUT_STATE_GROUPS];
    for (int k = 0; k < BREAKOUT_STATE_GROUPS; k++) {
        cudaMemcpy(&g[k], g_gpu.state + (size_t)k * g_gpu.n, sizeof(uint4), cudaMemcpyDeviceToHost);
    }
    GpuBreakoutConfig* c = &g_gpu.cfg;
    float paddle_x, ball_x, ball_y, ball_vx, paddle_width;
    memcpy(&paddle_x, &g[BREAKOUT_MOTION].x, sizeof(float));
    memcpy(&ball_x, &g[BREAKOUT_MOTION].y, sizeof(float));
    memcpy(&ball_y, &g[BREAKOUT_MOTION].z, sizeof(float));
    memcpy(&ball_vx, &g[BREAKOUT_BOUNCE].x, sizeof(float));
    memcpy(&paddle_width, &g[BREAKOUT_ROUND].x, sizeof(float));
    float paddle_y = c->height - c->paddle_height - 10;
    int score = (int)g[BREAKOUT_ROUND].z;
    int num_balls = (short)(g[BREAKOUT_ROUND].w & 0xffffu);
    const unsigned int brick_mask[4] = {
        g[BREAKOUT_BRICKS].x, g[BREAKOUT_BRICKS].y, g[BREAKOUT_BRICKS].z, g[BREAKOUT_BRICKS].w,
    };

    if (!g_gpu.client) {
        BreakoutClient* client = (BreakoutClient*)calloc(1, sizeof(BreakoutClient));
        client->width = c->width;
        client->height = c->height;
        InitWindow(c->width, c->height, "PufferLib Breakout (GPU)");
        SetTargetFPS(60 / (c->frameskip > 0 ? c->frameskip : 1));
        client->ball = LoadTexture("resources/shared/puffers_128.png");
        g_gpu.client = client;
    }
    BreakoutClient* client = g_gpu.client;

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(
        (int)paddle_x, (int)paddle_y,
        (int)paddle_width, (int)c->paddle_height,
        (Color){0, 255, 255, 255});

    DrawTexturePro(
        client->ball,
        (Rectangle){
            (ball_vx > 0) ? 0.0f : 128.0f,
            0, 128, 128,
        },
        (Rectangle){
            ball_x,
            ball_y,
            (float)c->ball_width,
            (float)c->ball_height,
        },
        (Vector2){0, 0},
        0,
        WHITE);

    for (int row = 0; row < c->brick_rows; row++) {
        for (int col = 0; col < c->brick_cols; col++) {
            int brick_idx = row * c->brick_cols + col;
            if ((brick_mask[brick_idx >> 5] & (1u << (brick_idx & 31))) != 0u) {
                continue;
            }
            int x = col * c->brick_width;
            int y = row * c->brick_height + Y_OFFSET;
            Color brick_color = BREAKOUT_BRICK_COLORS[row % 6];
            DrawRectangle(x, y, c->brick_width, c->brick_height, brick_color);
        }
    }

    DrawText(TextFormat("Score: %i", score), 10, 10, 20, WHITE);
    DrawText(TextFormat("Balls: %i", num_balls), client->width - 80, 10, 20, WHITE);
    EndDrawing();
    puf_web_vsync();
}

#endif
