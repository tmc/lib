// Package cudasafetensors reads Hugging Face .safetensors files and copies
// tensors into CUDA device memory.
//
// Parsers validate tensor dtype, shape, byte length, non-overlapping complete
// data coverage, and sharded index membership before exposing tensor bytes.
package cudasafetensors
