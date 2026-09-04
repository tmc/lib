package metalsample

import (
	_ "embed"
	"math"
)

// SampleArgmaxCPU returns the index of the maximum finite or infinite logit.
// It ignores NaNs and returns zero when logits is empty or has no value greater
// than negative infinity. Ties favor the lowest index. These rules match
// [ArgmaxSamplingMSL].
func SampleArgmaxCPU(logits []float32) int32 {
	bestIdx := -1
	bestVal := float32(math.Inf(-1))
	for i := 0; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	if bestIdx < 0 {
		return 0
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
