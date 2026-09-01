package metalsample

// SampleArgmaxCPU returns index of maximum logit value (greedy sampling).
func SampleArgmaxCPU(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	bestIdx := 0
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	return int32(bestIdx)
}

// ArgmaxSamplingMSL is the embedded MSL kernel for GPU-side argmax token
// extraction. Dispatch it as a single threadgroup of up to 1024 threads;
// a one-thread dispatch is also correct, just serial. Ties resolve to the
// lowest index, matching [SampleArgmaxCPU].
const ArgmaxSamplingMSL = `
#include <metal_stdlib>
using namespace metal;

kernel void ArgmaxSampling(
    device const float* logits [[buffer(0)]],
    device int* token_out      [[buffer(1)]],
    constant int& vocab_size   [[buffer(2)]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    threadgroup float s_val[1024];
    threadgroup int s_idx[1024];

    int best_idx = -1;
    float best_val = -INFINITY;
    for (int i = int(tid); i < vocab_size; i += int(tg_size)) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Halving reduction that also covers threadgroup sizes that are not
    // powers of two: n is the count still holding a partial result.
    for (uint n = tg_size; n > 1; ) {
        uint step = (n + 1) / 2;
        if (tid + step < n) {
            float other = s_val[tid + step];
            int oidx = s_idx[tid + step];
            if (other > s_val[tid] || (other == s_val[tid] && oidx < s_idx[tid])) {
                s_val[tid] = other;
                s_idx[tid] = oidx;
            }
        }
        n = step;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        token_out[0] = s_idx[0] < 0 ? 0 : s_idx[0];
    }
}
`
