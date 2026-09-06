package metalattn

import (
	_ "embed"
	"errors"
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
	DecodeThreadgroupSize         = 128
	DecodeTileSize                = 256
	DecodeMaxHeadDim              = 256
	PersistentGQAMaxRatio         = 8
	PersistentGQAMaxSequence      = 32768
	PersistentGQAThreadgroupBytes = PersistentGQAMaxRatio*(DecodeTileSize+DecodeMaxHeadDim)*4 + (4+2*PersistentGQAMaxRatio)*4
)

// ValidatePersistentGQA rejects shapes outside the correctness prototype's
// fixed threadgroup storage. The prototype is not a production route.
func ValidatePersistentGQA(cfg Config, sequence int) error {
	if cfg.NumQHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 || sequence <= 0 {
		return errors.New("persistent GQA: dimensions must be positive")
	}
	if cfg.NumQHeads%cfg.NumKVHeads != 0 || cfg.NumQHeads/cfg.NumKVHeads > PersistentGQAMaxRatio {
		return errors.New("persistent GQA: unsupported head ratio")
	}
	if cfg.HeadDim > DecodeMaxHeadDim {
		return errors.New("persistent GQA: unsupported head dimension")
	}
	if sequence > PersistentGQAMaxSequence {
		return errors.New("persistent GQA: unsupported sequence length")
	}
	return nil
}

// CausalFlashAttentionMSL is the embedded Metal Shading Language compute shader for causal attention decode.
//
//go:embed attention.metal
var CausalFlashAttentionMSL string

// PersistentGQAMSL is an experimental correctness-only kernel. It is not used
// by production. One threadgroup owns a KV head and computes all query heads
// in its group while loading each key and value element once.
//
//go:embed persistent_gqa.metal
var PersistentGQAMSL string
