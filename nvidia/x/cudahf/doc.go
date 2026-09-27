// Package cudahf finds model files in the local Hugging Face hub cache.
//
// It reads the cache that huggingface_hub writes and never touches the
// network. The snapshot directory it resolves is what
// [github.com/tmc/lib/nvidia/x/safetensors.OpenSharded] opens, and the
// result of that is what [github.com/tmc/lib/nvidia/x/cudasafetensors.LoadSlice]
// copies to the device.
//
// Usage:
//
//	mf, err := cudahf.Resolve("meta-llama/Llama-3.2-1B-Instruct")
//	ckpt, err := safetensors.OpenSharded(mf.SnapshotDir)
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudahf
