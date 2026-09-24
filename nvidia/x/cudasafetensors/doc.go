// Package cudasafetensors copies tensors from safetensors files into CUDA
// device memory.
//
// Package [github.com/tmc/lib/nvidia/x/safetensors] reads and validates the
// files; this package only moves tensor bytes to the device. [LoadSlice]
// checks that the Go element type matches the tensor's dtype and copies the
// bytes unchanged, so an F16 or BF16 tensor loads as uint16.
//
// Usage:
//
//	f, err := safetensors.Open("model.safetensors")
//	defer f.Close()
//	w, err := cudasafetensors.LoadSlice[uint16](f, "lm_head.weight", dev)
//	defer w.Free()
package cudasafetensors
