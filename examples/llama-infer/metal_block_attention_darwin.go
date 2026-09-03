//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"math"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/examples/internal/metalprefix"
)

type metalBlockDescriptor struct {
	Start, End uint32
	K, V       uint64
}

type metalBlockRow struct {
	PrefixTokens, SuffixTokens uint32
	SuffixOffset, BlockOffset  uint32
	BlockCount                 uint32
}

// RunBlockTableAttentionBatch computes B decode attention rows in one shared
// dispatch. It reads immutable prefix blocks in place and holds independent
// layer pins until the command buffer has completed.
func RunBlockTableAttentionBatch(ctx context.Context, eng *ModelEngine, cfg Config, leases []*metalprefix.Lease, layer int, qRows, suffixKRows, suffixVRows [][]float32) ([][]float32, error) {
	b := len(leases)
	qd := cfg.NumHeads * cfg.HeadDim
	kd := cfg.NumKVHeads * cfg.HeadDim
	if ctx == nil || eng == nil || eng.dev == nil || b == 0 || b > 4 || len(qRows) != b || len(suffixKRows) != b || len(suffixVRows) != b || layer < 0 || layer >= cfg.NumLayers {
		return nil, errors.New("block attention: invalid engine, leases, or layer")
	}
	if cfg.HeadDim <= 0 || cfg.HeadDim > metalattn.DecodeMaxHeadDim || cfg.NumKVHeads <= 0 || cfg.NumHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 {
		return nil, errors.New("block attention: unsupported configuration")
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("block attention: %w", err)
	}

	pins := make([]*metalprefix.ViewLease, 0, b)
	defer func() {
		for i := len(pins) - 1; i >= 0; i-- {
			pins[i].Release()
		}
	}()
	var desc []metalBlockDescriptor
	rows := make([]metalBlockRow, b)
	q := make([]float32, 0, b*qd)
	var suffixK, suffixV []float32
	for i, lease := range leases {
		if lease == nil || len(qRows[i]) != qd || len(suffixKRows[i]) == 0 || len(suffixKRows[i]) != len(suffixVRows[i]) || len(suffixKRows[i])%kd != 0 {
			return nil, fmt.Errorf("block attention: invalid row %d", i)
		}
		pin, err := lease.PinLayer(layer)
		if err != nil {
			return nil, fmt.Errorf("block attention: row %d prefix views: %w", i, err)
		}
		pins = append(pins, pin)
		views := pin.Views()
		suffixTokens := len(suffixKRows[i]) / kd
		if len(views) == 0 || len(views) > cfg.MaxSeqLen || lease.Tokens()+suffixTokens > cfg.MaxSeqLen {
			return nil, fmt.Errorf("block attention: row %d unsupported block table", i)
		}
		wantStart := 0
		rows[i] = metalBlockRow{PrefixTokens: uint32(lease.Tokens()), SuffixTokens: uint32(suffixTokens), SuffixOffset: uint32(len(suffixK) / kd), BlockOffset: uint32(len(desc)), BlockCount: uint32(len(views))}
		for _, view := range views {
			if view.Start != wantStart || view.End <= view.Start || view.End > lease.Tokens() {
				return nil, fmt.Errorf("block attention: row %d non-contiguous block table", i)
			}
			desc = append(desc, metalBlockDescriptor{Start: uint32(view.Start), End: uint32(view.End), K: uint64(view.K.GpuAddress()) + uint64(view.KOffset), V: uint64(view.V.GpuAddress()) + uint64(view.VOffset)})
			wantStart = view.End
		}
		if wantStart != lease.Tokens() {
			return nil, fmt.Errorf("block attention: row %d incomplete block table", i)
		}
		q = append(q, qRows[i]...)
		suffixK = append(suffixK, suffixKRows[i]...)
		suffixV = append(suffixV, suffixVRows[i]...)
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
	out, err := mempool.Alloc[float32](eng.dev, b*qd)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate output: %w", err)
	}
	defer out.Free()
	dbuf, err := mempool.AllocFromSlice(eng.dev, desc)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate descriptors: %w", err)
	}
	defer dbuf.Free()
	rbuf, err := mempool.AllocFromSlice(eng.dev, rows)
	if err != nil {
		return nil, fmt.Errorf("block attention: allocate rows: %w", err)
	}
	defer rbuf.Free()

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("block attention: %w", err)
	}
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	ce.SetComputePipelineState(eng.pso.BlockAttn)
	for i, s := range []mempool.Slice[float32]{qbuf, kbuf, vbuf, out} {
		ce.SetBufferWithOffsetAtIndex(s.Buffer(), s.Offset(), uint(i))
	}
	ce.SetBufferWithOffsetAtIndex(dbuf.Buffer(), dbuf.Offset(), 4)
	ce.SetBufferWithOffsetAtIndex(rbuf.Buffer(), rbuf.Offset(), 5)
	metal.SetScalar(ce, uint32(cfg.NumHeads), 6)
	metal.SetScalar(ce, uint32(cfg.NumKVHeads), 7)
	metal.SetScalar(ce, uint32(cfg.HeadDim), 8)
	metal.SetScalar(ce, float32(1/math.Sqrt(float64(cfg.HeadDim))), 9)
	for _, pin := range pins {
		for _, view := range pin.Views() {
			ce.UseResourceUsage(view.K, apple.MTLResourceUsageRead)
			ce.UseResourceUsage(view.V, apple.MTLResourceUsageRead)
		}
	}
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.NumHeads), metal.Threads(metalattn.DecodeThreadgroupSize))
	ce.EndEncoding()
	cb.Commit()
	if err := metal.WaitCommandBuffer(ctx, cb); err != nil {
		return nil, fmt.Errorf("block attention: command buffer: %w", err)
	}
	flat := out.HostSlice()
	result := make([][]float32, b)
	for i := range result {
		result[i] = append([]float32(nil), flat[i*qd:(i+1)*qd]...)
	}
	return result, nil
}

// RunBlockTableAttention computes one row using the batched physical path.
func RunBlockTableAttention(ctx context.Context, eng *ModelEngine, cfg Config, lease *metalprefix.Lease, layer int, q, suffixK, suffixV []float32) ([]float32, error) {
	rows, err := RunBlockTableAttentionBatch(ctx, eng, cfg, []*metalprefix.Lease{lease}, layer, [][]float32{q}, [][]float32{suffixK}, [][]float32{suffixV})
	if err != nil {
		return nil, err
	}
	return rows[0], nil
}
