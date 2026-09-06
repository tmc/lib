//go:build !darwin

package main

import (
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// UploadRetainedModelToGPU uploads a model that was loaded in its source
// dtypes.
//
// The device presently reads float32 weights, so the widening happens here,
// once, at the upload boundary, and the result goes through the same
// [UploadModelToGPU] path an F32 checkpoint takes. That keeps the F32 device
// behavior identical while moving the conversion out of the loader: when
// device kernels can read half storage, this function is where they get
// selected, and nothing upstream of it has to change.
func UploadRetainedModelToGPU(dev *cuda.Device, tracker *cudamem.Tracker, m *RetainedModel, cfg *ModelConfig) (*GPUModelWeights, error) {
	return UploadModelToGPU(dev, tracker, m.Float32(), cfg)
}
