// Package cudaoptim implements optimizer steps for training with
// hand-written CUDA kernels: AdamW with decoupled weight decay, global
// gradient-norm clipping, and non-finite gradient detection.
//
// [AdamW] updates a fixed list of float32 tensors, grouped by learning-rate
// scale and weight decay, with a constant number of kernel launches per step.
// The learning rate, step count and clip scale live in device memory
// ([State]), so a step needs no host synchronization and can be captured in a
// CUDA graph. The gradient norm is a two-stage float64 reduction with a fixed
// order, so steps are bitwise reproducible.
//
// Usage:
//
//	opt, err := cudaoptim.NewAdamW(dev, []cudaoptim.Group{{
//		Tensors:     tensors,
//		LRScale:     1,
//		WeightDecay: 0.01,
//	}}, cudaoptim.Config{LR: 1e-4, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, MaxNorm: 1})
//	...
//	err = opt.Step(stream) // after the backward pass has written the gradients
//
// The float64 host reference the tests compare against is
// [github.com/tmc/lib/nvidia/x/cudaref.AdamW].
package cudaoptim
