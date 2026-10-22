//go:build linux

package main

import (
	_ "embed"
	"errors"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

//go:embed cuda_shape_raw_commit.cu
var cudaShapeRawCommitSource string

// shapeRawProgress tracks enqueued writes on the State stream, not observed
// device completion. Any asynchronous failure invalidates the owning request.
type shapeRawProgress struct {
	next  uint32
	rows  uint32
	start uint32
}

// commitShapeRaw mutates the nonrotary KV coordinates through the pinned FP8
// round trip, then half-stores all coordinates to the ring. The normalized and
// post-RoPE KV vector remains float for direct-current-operand routes.
func commitShapeRaw(state *cudaShapeForward, entry driver.Function, layer int, position uint32, progress *shapeRawProgress) error {
	if state == nil || state.closed || entry == 0 || progress == nil {
		return errors.New("cuda shape raw commit: invalid owner or entry")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	c := state.plan.config
	if err := c.Validate(); err != nil {
		return err
	}
	if _, err := c.Layer(layer); err != nil {
		return err
	}
	if state.cache == nil || state.cache.r != state.r || state.cache.plan.layers != c.Layers {
		return errors.New("cuda shape raw commit: mismatched cache owner")
	}
	if state.plan.context == 0 || uint64(position) >= state.plan.context || position != progress.next {
		return errors.New("cuda shape raw commit: invalid position or write order")
	}
	raw, err := shapeCacheLayerBuffer(state.cache, layer, "raw")
	if err != nil {
		return err
	}
	kv := state.scratch["kv"]
	if kv.Len() != c.HeadDim || raw.Len()%c.HeadDim != 0 {
		return errors.New("cuda shape raw commit: invalid KV extent")
	}
	capacity := uint32(raw.Len() / c.HeadDim)
	if capacity == 0 || capacity > uint32(c.SlidingWindow) {
		return errors.New("cuda shape raw commit: invalid raw capacity")
	}
	if err := checkMXFP4DeviceRanges(mxfp4DeviceRange{uint64(kv.Ptr()), uint64(kv.Len()) * 4}, mxfp4DeviceRange{uint64(raw.Ptr()), uint64(raw.Len()) * 4}); err != nil {
		return err
	}
	if err := state.r.withContext(func() error {
		return cudalaunch.Kernel(entry).Grid(1, 1, 1).Block(64, 1, 1).On(state.r.stream()).Launch(kv, raw, capacity, position, uint32(c.HeadDim), uint32(c.RopeDim))
	}); err != nil {
		return err
	}
	progress.next = position + 1
	progress.rows = min(progress.next, capacity)
	progress.start = (progress.next - progress.rows) % capacity
	return nil
}
