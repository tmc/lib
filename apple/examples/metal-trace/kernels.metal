#include <metal_stdlib>
using namespace metal;

// vector_add performs basic elementwise addition.
kernel void vector_add(device const float *a [[buffer(0)]],
                       device const float *b [[buffer(1)]],
                       device float *c [[buffer(2)]],
                       constant uint &n [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

// compute performs a configurable chained fused multiply-add (FMA) operation.
kernel void compute(device float *c [[buffer(0)]],
                    constant uint &n [[buffer(1)]],
                    constant uint &rounds [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
    if (id < n) {
        float v = c[id];
        for (uint r = 0; r < rounds; r++) {
            v = fma(v, 1.000001f, 0.0001f);
        }
        c[id] = v;
    }
}
