//go:build linux

package main

import (
	_ "embed"
	"errors"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

//go:embed cuda_shape_attention.cu
var cudaShapeAttentionSource string

//go:embed cuda_shape_attention_external.cu
var cudaShapeAttentionExternalSource string

// The request describes committed cache rows, not inferred compressor progress.
// The cache writers supply these counts after enqueuing their writes.
type shapeAttentionRequest struct {
	layer                                       int
	position, rawRows, rawStart, compressedRows uint32
	useMask                                     bool
}

func shapeCacheLayerBuffer(cache *cudaShapeCache, layer int, name string) (cudamem.Slice[float32], error) {
	if cache == nil || cache.closed || len(cache.buffers) != len(cache.plan.buffers) {
		return cudamem.Slice[float32]{}, errors.New("cuda shape attention: closed or incomplete cache")
	}
	for i, p := range cache.plan.buffers {
		if p.layer == layer && p.name == name {
			v := cache.buffers[i]
			if v.Len() != p.elements || v.Ptr() == 0 {
				return v, errors.New("cuda shape attention: invalid cache extent")
			}
			return v, nil
		}
	}
	return cudamem.Slice[float32]{}, errors.New("cuda shape attention: missing cache buffer")
}

// encodeShapeAttention consumes actual State Q, persistent cache rows and
// borrowed sinks/mask into State heads. The compiled entry and its module are
// borrowed; this bounded mixed-decode route is not full forward admission.
func encodeShapeAttention(state *cudaShapeForward, fn driver.Function, req shapeAttentionRequest, sinks, mask cudamem.Slice[float32]) error {
	return encodeShapeAttentionRoute(state, fn, req, sinks, mask, cudamem.Slice[float32]{}, false)
}

func encodeShapeAttentionExternal(state *cudaShapeForward, fn driver.Function, req shapeAttentionRequest, sinks, mask, scores cudamem.Slice[float32]) error {
	return encodeShapeAttentionRoute(state, fn, req, sinks, mask, scores, true)
}

func encodeShapeAttentionRoute(state *cudaShapeForward, fn driver.Function, req shapeAttentionRequest, sinks, mask, scores cudamem.Slice[float32], external bool) error {
	if state == nil || state.closed || fn == 0 || state.cache == nil || state.cache.r != state.r {
		return errors.New("cuda shape attention: closed or mismatched state")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	c := state.plan.config
	if err := c.Validate(); err != nil {
		return err
	}
	if state.cache.plan.layers != c.Layers {
		return errors.New("cuda shape attention: cache family mismatch")
	}
	layer, err := c.Layer(req.layer)
	if err != nil {
		return err
	}
	if state.plan.context == 0 || uint64(req.position) >= state.plan.context || req.position == math.MaxUint32 || sinks.Len() != c.Heads {
		return errors.New("cuda shape attention: invalid position or sinks")
	}
	raw, err := shapeCacheLayerBuffer(state.cache, req.layer, "raw")
	if err != nil {
		return err
	}
	if raw.Len()%c.HeadDim != 0 {
		return errors.New("cuda shape attention: truncated raw cache row")
	}
	cap := uint32(raw.Len() / c.HeadDim)
	if cap == 0 || req.rawRows > cap || req.rawStart >= cap || req.rawRows > req.position+1 || req.rawRows > 256 {
		return errors.New("cuda shape attention: invalid raw cache bounds")
	}
	// The zero-prefix route consumes the current post-FP8 float KV operand,
	// not the half-stored ring. Keeping the two forms separate is required
	// even when the selected row and token position are the same.
	if req.position == 0 && req.rawRows == 1 && req.compressedRows == 0 {
		raw = state.scratch["kv"]
		if raw.Len() != c.HeadDim || raw.Ptr() == 0 {
			return errors.New("cuda shape attention: invalid current KV operand")
		}
		cap, req.rawStart = 1, 0
	}
	comp := raw
	if layer.CompressRatio != 0 {
		comp, err = shapeCacheLayerBuffer(state.cache, req.layer, "attention-cache")
		if err != nil {
			return err
		}
		if uint64(req.compressedRows)*uint64(c.HeadDim) > uint64(comp.Len()) || req.compressedRows > (req.position+1)/uint32(layer.CompressRatio) {
			return errors.New("cuda shape attention: invalid compressed cache bounds")
		}
	} else if req.compressedRows != 0 || req.useMask {
		return errors.New("cuda shape attention: compressed rows on raw layer")
	}
	if !external && uint64(req.rawRows)+uint64(req.compressedRows) > 8192 {
		return errors.New("cuda shape attention: mixed kernel score capacity exceeded")
	}
	var scoreStride uint32
	if external {
		var count uint64
		scoreStride, count, err = shapeAttentionScoreLayout(c, state.plan.context)
		if err != nil {
			return err
		}
		if uint64(scores.Len()) != count || scores.Ptr() == 0 || uint64(req.rawRows)+uint64(req.compressedRows) > uint64(scoreStride) {
			return errors.New("cuda shape attention: invalid context score extent")
		}
	}
	q, heads := state.scratch["q"], state.scratch["heads"]
	if q.Len() != c.Heads*c.HeadDim || heads.Len() != q.Len() || req.useMask && mask.Len() != int(req.compressedRows) {
		return errors.New("cuda shape attention: invalid query, output or mask extent")
	}
	ranges := []mxfp4DeviceRange{}
	for _, v := range []cudamem.Slice[float32]{q, heads, sinks, raw} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len()) * 4})
	}
	if layer.CompressRatio != 0 {
		ranges = append(ranges, mxfp4DeviceRange{uint64(comp.Ptr()), uint64(comp.Len()) * 4})
	}
	if external {
		ranges = append(ranges, mxfp4DeviceRange{uint64(scores.Ptr()), uint64(scores.Len()) * 4})
	}
	ranges, err = appendShapeAttentionMaskRange(ranges, req.useMask, req.compressedRows, mxfp4DeviceRange{uint64(mask.Ptr()), uint64(mask.Len()) * 4})
	if err != nil {
		return err
	}
	if err := checkMXFP4DeviceRanges(ranges...); err != nil {
		return err
	}
	useMask := uint32(0)
	var maskPtr driver.DevicePtr
	if req.useMask {
		useMask = 1
		maskPtr = mask.Ptr()
	}
	return state.r.withContext(func() error {
		if external {
			return cudalaunch.Kernel(fn).Grid(1, uint32(c.Heads), 1).Block(512, 1, 1).On(state.r.stream()).Launch(heads, sinks, q, raw, comp, maskPtr, useMask, uint32(1), req.position, req.rawRows, cap, req.rawStart, req.compressedRows, uint32(c.SlidingWindow), uint32(layer.CompressRatio), uint32(c.Heads), uint32(c.HeadDim), scores, scoreStride, uint32(0))
		}
		return cudalaunch.Kernel(fn).Grid(1, uint32(c.Heads), 1).Block(512, 1, 1).On(state.r.stream()).Launch(heads, sinks, q, raw, comp, maskPtr, useMask, uint32(1), req.position, req.rawRows, cap, req.rawStart, req.compressedRows, uint32(c.SlidingWindow), uint32(layer.CompressRatio), uint32(c.Heads), uint32(c.HeadDim), uint32(0))
	})
}
