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

// ArgmaxSamplingMSL is the embedded MSL kernel for GPU-side argmax token extraction.
const ArgmaxSamplingMSL = `
#include <metal_stdlib>
using namespace metal;

kernel void ArgmaxSampling(
    device const float* logits [[buffer(0)]],
    device int* token_out      [[buffer(1)]],
    constant int& vocab_size   [[buffer(2)]],
    uint tid                   [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    token_out[0] = max_idx;
}
`
