package metalattn

import (
	_ "embed"
	"math"
)

// Config defines the dimensional configuration for Multi-Head or Grouped Query Attention.
type Config struct {
	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	Scale      float32
}

// DefaultConfig creates an attention configuration with standard 1/sqrt(headDim) softmax scale.
func DefaultConfig(numQHeads, numKVHeads, headDim int) Config {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	return Config{
		NumQHeads:  numQHeads,
		NumKVHeads: numKVHeads,
		HeadDim:    headDim,
		Scale:      scale,
	}
}

// Dispatch invariants for CausalFlashAttentionDecode.
//
// The kernel reduces across exactly DecodeThreadgroupSize/32 SIMD groups, so it
// must be dispatched with DecodeThreadgroupSize threads per threadgroup and one
// threadgroup per query head. Sequence length is unbounded: keys and values are
// consumed in tiles of DecodeTileSize with an online-softmax rescale between
// tiles. Head dimension is bounded by DecodeMaxHeadDim, which is the size of the
// threadgroup output accumulator.
const (
	DecodeThreadgroupSize = 128
	DecodeTileSize        = 256
	DecodeMaxHeadDim      = 256
)

// CausalFlashAttentionMSL is the embedded Metal Shading Language compute shader for causal attention decode.
//
//go:embed attention.metal
var CausalFlashAttentionMSL string
