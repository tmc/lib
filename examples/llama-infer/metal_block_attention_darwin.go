//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/examples/internal/metalprefix"
)

const maxMetalPrefixBlocks = 8

// RunBlockTableAttention computes one decode attention row without copying its
// immutable prefix blocks into the writable suffix. q and suffixKV are already
// RoPE-transformed; the lease remains pinned through command completion.
func RunBlockTableAttention(ctx context.Context, eng *ModelEngine, cfg Config, lease *metalprefix.Lease, layer int, q, suffixK, suffixV []float32) ([]float32, error) {
	qd := cfg.NumHeads * cfg.HeadDim
	kd := cfg.NumKVHeads * cfg.HeadDim
	if ctx == nil || eng == nil || eng.dev == nil || lease == nil || layer < 0 || layer >= cfg.NumLayers {
		return nil, errors.New("block attention: invalid engine, lease, or layer")
	}
	if len(q) != qd || len(suffixK) == 0 || len(suffixK) != len(suffixV) || len(suffixK)%kd != 0 ||
		cfg.HeadDim <= 0 || cfg.HeadDim > metalattn.DecodeMaxHeadDim || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, errors.New("block attention: invalid tensor shape")
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("block attention: %w", err)
	}
	pin, err := lease.PinLayer(layer)
	if err != nil {
		return nil, fmt.Errorf("block attention: prefix views: %w", err)
	}
	defer pin.Release()
	views := pin.Views()
	if len(views) == 0 || len(views) > maxMetalPrefixBlocks {
		return nil, errors.New("block attention: unsupported prefix block count")
	}
	wantStart := 0
	for _, view := range views {
		if view.Start != wantStart || view.End <= view.Start || view.End > lease.Tokens() {
			return nil, errors.New("block attention: non-contiguous block table")
		}
		wantStart = view.End
	}
	if wantStart != lease.Tokens() {
		return nil, errors.New("block attention: incomplete block table")
	}

	qbuf, err := mempool.AllocFromSlice(eng.dev, q)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate q: %w", err)
	}
	defer qbuf.Free()
	kbuf, err := mempool.AllocFromSlice(eng.dev, suffixK)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate suffix K: %w", err)
	}
	defer kbuf.Free()
	vbuf, err := mempool.AllocFromSlice(eng.dev, suffixV)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate suffix V: %w", err)
	}
	defer vbuf.Free()
	out, err := mempool.Alloc[float32](eng.dev, qd)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate output: %w", err)
	}
	defer out.Free()
	desc := make([]uint32, 4*len(views))
	for i, view := range views {
		desc[4*i], desc[4*i+1], desc[4*i+2] = uint32(view.Start), uint32(view.End), uint32(i)
	}
	dbuf, err := mempool.AllocFromSlice(eng.dev, desc)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate descriptors: %w", err)
	}
	defer dbuf.Free()
	params := []uint32{uint32(lease.Tokens()), uint32(len(suffixK) / kd), uint32(len(views)), uint32(cfg.NumHeads), uint32(cfg.NumKVHeads), uint32(cfg.HeadDim), math.Float32bits(float32(1 / math.Sqrt(float64(cfg.HeadDim))))}
	pbuf, err := mempool.AllocFromSlice(eng.dev, params)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate parameters: %w", err)
	}
	defer pbuf.Free()

	cb := eng.dev.Queue().CommandBuffer()
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("block attention: %w", err)
	}
	ce := cb.ComputeCommandEncoder()
	ce.SetComputePipelineState(eng.pso.BlockAttn)
	for i, s := range []mempool.Slice[float32]{qbuf, kbuf, vbuf, out} {
		ce.SetBufferWithOffsetAtIndex(s.Buffer(), s.Offset(), uint(i))
	}
	ce.SetBufferWithOffsetAtIndex(dbuf.Buffer(), dbuf.Offset(), 4)
	ce.SetBufferWithOffsetAtIndex(pbuf.Buffer(), pbuf.Offset(), 5)
	for i := 0; i < maxMetalPrefixBlocks; i++ {
		view := views[min(i, len(views)-1)]
		ce.SetBufferWithOffsetAtIndex(view.K, view.KOffset, uint(6+i))
		ce.SetBufferWithOffsetAtIndex(view.V, view.VOffset, uint(14+i))
	}
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(cfg.NumHeads), metal.Threads(metalattn.DecodeThreadgroupSize))
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	return append([]float32(nil), out.HostSlice()...), nil
}
