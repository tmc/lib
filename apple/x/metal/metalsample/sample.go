package metalsample

import (
	_ "embed"
)

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
//
//go:embed sample.metal
var ArgmaxSamplingMSL string
