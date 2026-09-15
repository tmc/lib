//go:build darwin

package metalprof

import (
	apple "github.com/tmc/apple/metal"
)

// GPUTiming reports the GPU execution window of a completed command buffer.
//
// The timestamps come from Metal's own GPUStartTime and GPUEndTime, so unlike
// a capture they cost nothing and do not perturb what they measure. Call it
// after the command buffer has completed; before that both timestamps are zero
// and the result is not [CommandBufferTiming.Valid].
//
// Subtracting this from wall-clock time around a submission attributes the
// remainder to host work: allocation, encoding, submission, and copy-back.
func GPUTiming(cb apple.MTLCommandBuffer) CommandBufferTiming {
	if cb.GetID() == 0 {
		return CommandBufferTiming{}
	}
	return CommandBufferTiming{
		Start: seconds(float64(cb.GPUStartTime())),
		End:   seconds(float64(cb.GPUEndTime())),
	}
}
