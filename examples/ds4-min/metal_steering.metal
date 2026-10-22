// Matches antirez/ds4 11e8cad metal/dsv4_misc.metal:392-437.
#include <metal_stdlib>
using namespace metal;
struct SteeringArgs { uint width, rows, layer, threads; float scale; };
kernel void ds4_directional_project(constant SteeringArgs &args [[buffer(0)]],
        device float *x [[buffer(1)]], device const float *directions [[buffer(2)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;
    device float *xr = x + (ulong)row * args.width;
    device const float *dir = directions + (ulong)args.layer * args.width;
    float sum = 0;
    for (uint i = tid; i < args.width; i += args.threads) sum += xr[i] * dir[i];
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = args.threads >> 1; step; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float coefficient = args.scale * scratch[0];
    for (uint i = tid; i < args.width; i += args.threads) xr[i] -= coefficient * dir[i];
}
