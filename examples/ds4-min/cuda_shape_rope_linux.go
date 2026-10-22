//go:build linux

package main

import (
	_ "embed"
	"errors"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

//go:embed cuda_shape_rope.cu
var cudaShapeRopeSource string

// encodeShapeRoPE rotates actual normalized State Q/K in place. It must run
// exactly once after QKV and before KV quantization/cache commit. The caller
// owns the entry's module and must admit the model's default RoPE metadata.
func encodeShapeRoPE(state *cudaShapeForward, entry driver.Function, layer int, position uint32) error {
	if state == nil || state.closed || entry == 0 {
		return errors.New("cuda shape RoPE: closed state or missing entry")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	if state.plan.context == 0 || uint64(position) >= state.plan.context {
		return errors.New("cuda shape RoPE: position outside State context")
	}
	c := state.plan.config
	p, err := defaultShapeRope(c, layer)
	if err != nil {
		return err
	}
	q, kv := state.scratch["q"], state.scratch["kv"]
	if q.Len() != c.Heads*c.HeadDim || kv.Len() != c.HeadDim {
		return errors.New("cuda shape RoPE: invalid Q/K extent")
	}
	if err := checkMXFP4DeviceRanges(mxfp4DeviceRange{uint64(q.Ptr()), uint64(q.Len()) * 4}, mxfp4DeviceRange{uint64(kv.Ptr()), uint64(kv.Len()) * 4}); err != nil {
		return err
	}
	return state.r.withContext(func() error {
		for _, v := range []struct {
			x     cudamem.Slice[float32]
			heads int
		}{{q, c.Heads}, {kv, 1}} {
			pairs := v.heads * c.RopeDim / 2
			if err := cudalaunch.Kernel(entry).Grid(uint32((pairs+255)/256), 1, 1).Block(256, 1, 1).On(state.r.stream()).Launch(v.x, uint32(1), uint32(v.heads), uint32(c.HeadDim), uint32(c.RopeDim), position, uint32(1), p.originalContext, int32(0), p.base, p.scale, p.extension, p.attentionFactor, p.betaFast, p.betaSlow); err != nil {
				return err
			}
		}
		return nil
	})
}

// encodeShapeAttentionInverseRoPE mirrors ds4.c:24629 at the pinned revision:
// the unfused mixed-attention result is inverse-rotated before output A. It
// borrows the same module and layer parameters as the forward Q/K rotation.
func encodeShapeAttentionInverseRoPE(state *cudaShapeForward, entry driver.Function, layer int, position uint32) error {
	if state == nil || state.closed || entry == 0 {
		return errors.New("cuda shape inverse RoPE: closed state or missing entry")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	if state.plan.context == 0 || uint64(position) >= state.plan.context {
		return errors.New("cuda shape inverse RoPE: position outside State context")
	}
	c := state.plan.config
	p, err := defaultShapeRope(c, layer)
	if err != nil {
		return err
	}
	heads := state.scratch["heads"]
	if heads.Len() != c.Heads*c.HeadDim {
		return errors.New("cuda shape inverse RoPE: invalid head extent")
	}
	if err := checkMXFP4DeviceRanges(mxfp4DeviceRange{uint64(heads.Ptr()), uint64(heads.Len()) * 4}); err != nil {
		return err
	}
	pairs := c.Heads * c.RopeDim / 2
	return state.r.withContext(func() error {
		return cudalaunch.Kernel(entry).Grid(uint32((pairs+255)/256), 1, 1).Block(256, 1, 1).On(state.r.stream()).Launch(heads, uint32(1), uint32(c.Heads), uint32(c.HeadDim), uint32(c.RopeDim), position, uint32(1), p.originalContext, int32(1), p.base, p.scale, p.extension, p.attentionFactor, p.betaFast, p.betaSlow)
	})
}
