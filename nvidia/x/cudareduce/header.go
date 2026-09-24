package cudareduce

// Threads is the block size every stage of a reduction launches with, and
// CUDAREDUCE_THREADS in [Header].
const Threads = 256

// Header is the NVRTC-includable CUDA header cudareduce.cuh. Pass it to
// nvrtcx.WithHeader("cudareduce.cuh", cudareduce.Header) and include it to
// fuse a column reduction into a kernel of your own: write a stage 1 kernel
// around cudareduce_column_partials or cudareduce_segment_partials with a
// functor that loads and combines your inputs, launch it on the grid [Split]
// describes, and finish with [Reducer.Finish].
//
// Every sum is accumulated in double in a fixed order, and no two threads
// touch the same memory, so results are deterministic and the header needs
// no synchronization.
const Header = `#pragma once
#ifndef CUDAREDUCE_CUH
#define CUDAREDUCE_CUH

#define CUDAREDUCE_THREADS 256

// Stage 1 of a column reduction over rows [0, rows) and columns [0, n).
//
// Launch it with CUDAREDUCE_THREADS threads per block on the grid
// (ceil(n / CUDAREDUCE_THREADS), splits). Block (x, s) covers the columns
// x*CUDAREDUCE_THREADS + threadIdx.x and the rows
// [s*rows_per_split, min(rows, (s+1)*rows_per_split)). For each row in
// order, f(row, col, acc) adds that row's contribution to each of the K
// sums in acc[0..K). The thread then stores acc[k] at
// partials[(k*splits + s)*n + col].
template <int K, typename F>
__device__ __forceinline__ void cudareduce_column_partials(
    F f, long long rows, long long n, long long rows_per_split, double* partials)
{
    long long col = (long long)blockIdx.x * CUDAREDUCE_THREADS + threadIdx.x;
    if (col >= n) return;
    long long split = blockIdx.y, splits = gridDim.y;
    long long r0 = split * rows_per_split;
    long long r1 = r0 + rows_per_split;
    if (r1 > rows) r1 = rows;
    double acc[K];
    #pragma unroll
    for (int k = 0; k < K; k++) acc[k] = 0;
    for (long long r = r0; r < r1; r++) f(r, col, acc);
    #pragma unroll
    for (int k = 0; k < K; k++) partials[(k * splits + split) * n + col] = acc[k];
}

// Stage 1 of a segmented column reduction: row r adds f(r, col) to the sum
// of group seg[r], and rows whose group is outside [0, groups) are skipped.
// The launch is as for cudareduce_column_partials with K = groups, plus
// groups*CUDAREDUCE_THREADS doubles of dynamic shared memory. Each thread
// uses only its own slots of it.
template <typename F>
__device__ __forceinline__ void cudareduce_segment_partials(
    F f, const int* seg, int groups, long long rows, long long n,
    long long rows_per_split, double* partials)
{
    extern __shared__ double cudareduce_smem[];
    long long col = (long long)blockIdx.x * CUDAREDUCE_THREADS + threadIdx.x;
    if (col >= n) return;
    double* acc = cudareduce_smem + threadIdx.x;
    for (int g = 0; g < groups; g++) acc[g * CUDAREDUCE_THREADS] = 0;
    long long split = blockIdx.y, splits = gridDim.y;
    long long r0 = split * rows_per_split;
    long long r1 = r0 + rows_per_split;
    if (r1 > rows) r1 = rows;
    for (long long r = r0; r < r1; r++) {
        int g = seg[r];
        if (g >= 0 && g < groups) acc[g * CUDAREDUCE_THREADS] += f(r, col);
    }
    for (int g = 0; g < groups; g++)
        partials[((long long)g * splits + split) * n + col] = acc[g * CUDAREDUCE_THREADS];
}

// Stage 2: out[k*n + col] = sum over s in order of
// partials[(k*splits + s)*n + col], plus the old out[k*n + col] if
// accumulate is nonzero, rounded once to float. Launch on the grid
// (ceil(n / CUDAREDUCE_THREADS), K) with CUDAREDUCE_THREADS threads.
__device__ __forceinline__ void cudareduce_finish(
    const double* partials, long long splits, long long n, float* out, int accumulate)
{
    long long col = (long long)blockIdx.x * CUDAREDUCE_THREADS + threadIdx.x;
    if (col >= n) return;
    long long k = blockIdx.y;
    const double* p = partials + k * splits * n + col;
    double sum = 0;
    for (long long s = 0; s < splits; s++) sum += p[s * n];
    float* o = out + k * n + col;
    if (accumulate) sum += (double)*o;
    *o = (float)sum;
}

#endif // CUDAREDUCE_CUH
`

// kernelSource instantiates the header for the reductions [Reducer] launches.
const kernelSource = `#include "cudareduce.cuh"

extern "C" __global__ void cudareduce_colsum(
    const float* __restrict__ x, long long rows, long long n,
    long long rows_per_split, double* __restrict__ partials)
{
    cudareduce_column_partials<1>(
        [=](long long r, long long c, double* acc) { acc[0] += (double)x[r * n + c]; },
        rows, n, rows_per_split, partials);
}

extern "C" __global__ void cudareduce_segment_colsum(
    const float* __restrict__ x, const int* __restrict__ seg, int groups,
    long long rows, long long n, long long rows_per_split, double* __restrict__ partials)
{
    cudareduce_segment_partials(
        [=](long long r, long long c) { return (double)x[r * n + c]; },
        seg, groups, rows, n, rows_per_split, partials);
}

// dw = sum_rows(dy * xhat) in partials k=0 and db = sum_rows(dy) in k=1.
// xhat = (x - mean[r]) * rstd[r], or x itself when mean is null.
extern "C" __global__ void cudareduce_layernorm_partials(
    const float* __restrict__ dy, const float* __restrict__ x,
    const float* __restrict__ mean, const float* __restrict__ rstd,
    long long rows, long long n, long long rows_per_split, double* __restrict__ partials)
{
    cudareduce_column_partials<2>(
        [=](long long r, long long c, double* acc) {
            double g = dy[r * n + c];
            double xh = x[r * n + c];
            if (mean) xh = (xh - (double)mean[r]) * (double)rstd[r];
            acc[0] += g * xh;
            acc[1] += g;
        },
        rows, n, rows_per_split, partials);
}

extern "C" __global__ void cudareduce_finish_kernel(
    const double* __restrict__ partials, long long splits, long long n,
    float* out, int accumulate)
{
    cudareduce_finish(partials, splits, n, out, accumulate);
}
`
