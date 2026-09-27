// Package safetensors reads and writes the Hugging Face safetensors format.
//
// A file is an 8-byte little-endian header length, a JSON header, and the raw
// tensor bytes. The header maps each tensor name to its dtype, shape and byte
// range, plus an optional "__metadata__" map of strings. [Write] and
// [WriteFile] produce files that safetensors.torch.load_file accepts, and
// [Open] and [NewFile] read them back without converting or copying tensor
// bytes, so a frozen F16 weight can be copied from one file to another
// byte for byte. [OpenSharded] opens a checkpoint split across several files
// by a model.safetensors.index.json weight map.
//
// The package does not touch the GPU. [EncodeFloat32] and the Float16 and
// BFloat16 conversions turn float32 host data into F16 or BF16 bytes with
// round-to-nearest-even.
//
// Usage:
//
//	t := safetensors.Tensor{Name: "w", Dtype: safetensors.F32, Shape: []int{2}, Data: data}
//	err := safetensors.WriteFile("model.safetensors", []safetensors.Tensor{t}, nil)
//
//	f, err := safetensors.Open("model.safetensors")
//	defer f.Close()
//	w, err := f.Tensor("w")
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package safetensors
