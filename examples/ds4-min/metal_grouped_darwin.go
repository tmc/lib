//go:build darwin && arm64

package main

import (
	"context"
	"errors"
	"fmt"
	"math"

	apple "github.com/tmc/apple/metal"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// metalGroupedExecution owns scratch until every submitted buffer is terminal.
// Free has no error result; command completion and retained ownership do.
type metalGroupedExecution struct {
	batches     []*metalBatch
	scratch     []metalmem.Slice[float32]
	released    bool
	afterEncode func(int) error         // after actual encoding; production is nil
	wait        func(*metalBatch) error // completion fault seam; production is nil
	terminal    func(*metalBatch) bool  // synthetic status seam; production is nil
}

func (x *metalGroupedExecution) isTerminal(batch *metalBatch) bool {
	if x.terminal != nil {
		return x.terminal(batch)
	}
	return metalGroupedTerminal(batch)
}

func metalGroupedTerminal(batch *metalBatch) bool {
	status := batch.buffer.Status()
	return status == apple.MTLCommandBufferStatusCompleted || status == apple.MTLCommandBufferStatusError
}

func (x *metalGroupedExecution) checkpoint(phase int) error {
	if x.afterEncode != nil {
		return x.afterEncode(phase)
	}
	return nil
}

func (x *metalGroupedExecution) close() error {
	if x.released {
		return nil
	}
	var err error
	live := false
	for _, batch := range x.batches {
		if !batch.done {
			batch.endEncoding()
			batch.ops.hc.complete(batch)
			continue
		}
		if x.wait != nil {
			err = errors.Join(err, x.wait(batch))
		} else if !x.isTerminal(batch) {
			err = errors.Join(err, libmetal.WaitCommandBuffer(context.Background(), batch.buffer))
		}
		if !x.isTerminal(batch) {
			live = true
			err = errors.Join(err, fmt.Errorf("metal grouped cleanup: nonterminal command buffer"))
		} else {
			batch.ops.hc.complete(batch)
		}
	}
	if live {
		return err // preserve every scratch owner and the runtime arena
	}
	for _, buffer := range x.scratch {
		buffer.Free()
	}
	x.released = true
	return err
}

func (x *metalGroupedExecution) finish(r *metalRuntime, err error) error {
	err = errors.Join(err, x.close())
	if !x.released {
		r.groupedPending = x
		r.failed = errors.Join(err, dsv4.ErrDesynchronized)
	}
	return err
}

// admitMetalGroup bounds the device-only forward schedule. The large-context
// indexer fallback reads intermediate buffers on the host and is not admitted.
// This route groups command submission, not the per-layer FFN matrix shapes.
func admitMetalGroup(r *metalRuntime, tokens []int, position int) error {
	if r != nil {
		if err := admitGroupedPower(r.power); err != nil {
			return err
		}
	}
	if r == nil || r.streaming != nil || r.failed != nil || r.speculativeEpoch == ^uint64(0) || r.ctx > ds4RawCap || r.ctx <= 0 ||
		position != r.pos || position <= 0 || len(tokens) == 0 || len(tokens) > 8 ||
		len(tokens) > r.ctx-position || r.collectors != nil || r.steering != nil {
		return fmt.Errorf("metal grouped forward: unsupported state or extent")
	}
	for _, token := range tokens {
		if token < 0 || token >= Flash.Vocab {
			return fmt.Errorf("metal grouped forward: invalid token %d", token)
		}
	}
	for i := 2; i < len(r.layers); i += 2 {
		if r.state.layers[i].indexer.n >= 512 {
			return fmt.Errorf("metal grouped forward: host indexer route at layer %d", i)
		}
	}
	return nil
}

// EncodeEmbedding writes all four residual streams without a host readback.
func (b *metalBatch) EncodeEmbedding(t boundMetalTensor, token int, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if t.typeID != 1 || len(t.dimensions) != 2 || t.dimensions[0] != ds4Width ||
		token < 0 || uint64(token) >= t.dimensions[1] || out.Len() != ds4HCWidth {
		return fmt.Errorf("metal grouped embedding: invalid shape or token")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.GetRowF16)
	for stream := range 4 {
		row, err := out.Slice(stream*ds4Width, (stream+1)*ds4Width)
		if err != nil {
			return err
		}
		enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(row.Buffer(), row.Offset(), 1)
		libmetal.SetScalar(enc, uint32(token), 2)
		libmetal.SetScalar(enc, uint32(ds4Width), 3)
		enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(ds4Width), libmetal.Threads(256))
	}
	b.encoded++
	return nil
}

func (b *metalBatch) encodeCopy(in, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if in.Len() == 0 || in.Len() != out.Len() {
		return fmt.Errorf("metal grouped copy: invalid shape")
	}
	b.endEncoding()
	enc := b.buffer.BlitCommandEncoder()
	enc.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(in.Buffer(), in.Offset(), out.Buffer(), out.Offset(), uint(in.Len()*4))
	enc.EndEncoding()
	b.encoded++
	return nil
}

// evalGrouped computes every teacher-forced row. All layer work is encoded in
// one ordered command buffer, with residual rows copied before scratch reuse.
// The two output matmuls operate on all rows. Only the output-HC sigmoid stays
// on the host, retaining the scalar path's float64 exponential and stream order.
// The caller owns a frontier snapshot and must restore it on any error.
func (r *metalRuntime) evalGrouped(tokens []int, position int) (rows [][]float32, resultErr error) {
	return r.evalGroupedWithHooks(tokens, position, nil)
}

func (r *metalRuntime) evalGroupedWithHooks(tokens []int, position int, hooks *metalGroupedExecution) (rows [][]float32, resultErr error) {
	if err := admitMetalGroup(r, tokens, position); err != nil {
		return nil, err
	}
	if r.dev == nil || r.ops == nil || r.arena == nil {
		return nil, fmt.Errorf("metal grouped forward: missing device resources")
	}
	x := hooks
	if x == nil {
		x = new(metalGroupedExecution)
	}
	if x.released || len(x.batches) != 0 || len(x.scratch) != 0 {
		return nil, fmt.Errorf("metal grouped forward: execution owner already used")
	}
	defer func() {
		resultErr = x.finish(r, resultErr)
	}()
	count := len(tokens)
	hc, err := metalmem.Alloc[float32](r.dev, count*ds4HCWidth)
	if err != nil {
		return nil, err
	}
	x.scratch = append(x.scratch, hc)
	flat, err := metalmem.Alloc[float32](r.dev, count*ds4HCWidth)
	if err != nil {
		return nil, err
	}
	x.scratch = append(x.scratch, flat)
	pre, err := metalmem.Alloc[float32](r.dev, count*4)
	if err != nil {
		return nil, err
	}
	x.scratch = append(x.scratch, pre)
	weighted, err := metalmem.Alloc[float32](r.dev, count*ds4Width)
	if err != nil {
		return nil, err
	}
	x.scratch = append(x.scratch, weighted)
	output, err := metalmem.Alloc[float32](r.dev, count*Flash.Vocab)
	if err != nil {
		return nil, err
	}
	x.scratch = append(x.scratch, output)
	// Compile lazy readout pipelines before any state-changing dispatch.
	if _, err := r.ops.ensureMatmulF16(); err != nil {
		return nil, err
	}
	if _, err := r.ops.ensureMatmulQ80(); err != nil {
		return nil, err
	}
	batch, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	x.batches = append(x.batches, batch)
	for row, token := range tokens {
		if r.dsparkFeatures != nil {
			if err := r.dsparkFeatures.beginGrouped(position+row, token, batch); err != nil {
				return nil, err
			}
		}
		if err := batch.EncodeEmbedding(r.output.embedding, token, r.state.hc[0]); err != nil {
			return nil, err
		}
		for layer := range r.layers {
			next, err := r.evalLayer(batch, layer, int32(token), position+row)
			if err != nil {
				return nil, fmt.Errorf("metal grouped forward row %d layer %d: %w", row, layer, err)
			}
			if next != batch {
				return nil, fmt.Errorf("metal grouped forward: unexpected host boundary")
			}
		}
		if r.dsparkFeatures != nil {
			if err := r.dsparkFeatures.finishGroupedRow(); err != nil {
				return nil, err
			}
		}
		dst, _ := hc.Slice(row*ds4HCWidth, (row+1)*ds4HCWidth)
		if err := batch.encodeCopy(r.state.hc[0], dst); err != nil {
			return nil, err
		}
	}
	if err := batch.EncodeRMS(hc, r.output.norm, flat, count, false); err != nil {
		return nil, err
	}
	if err := batch.EncodeMatmulF16(r.output.hcFn, flat, pre, count); err != nil {
		return nil, err
	}
	if err := x.checkpoint(0); err != nil {
		return nil, err
	}
	if err := batch.Wait(context.Background()); err != nil {
		return nil, err
	}
	if err := collapseMetalOutputRows(hc.HostSlice(), pre.HostSlice(), boundF32(r.output.hcBase), boundF32(r.output.hcScale), weighted.HostSlice(), count); err != nil {
		return nil, err
	}
	head, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	x.batches = append(x.batches, head)
	if err := head.EncodeRMS(weighted, r.output.norm, weighted, count, true); err != nil {
		return nil, err
	}
	if err := head.EncodeMatmulQ80(r.output.weight, weighted, output, count); err != nil {
		return nil, err
	}
	if err := x.checkpoint(1); err != nil {
		return nil, err
	}
	if err := head.Wait(context.Background()); err != nil {
		return nil, err
	}
	rows = make([][]float32, count)
	for row := range count {
		values := output.HostSlice()[row*Flash.Vocab : (row+1)*Flash.Vocab]
		if err := checkFiniteLogits("metal grouped row", values); err != nil {
			return nil, fmt.Errorf("row %d: %w", row, err)
		}
		rows[row] = append([]float32(nil), values...)
	}
	if r.dsparkFeatures != nil {
		if err := r.dsparkFeatures.sealGrouped(count); err != nil {
			return nil, err
		}
	}
	copy(r.state.logits.HostSlice(), rows[count-1])
	r.pos += count
	r.speculativeEpoch++
	r.lastBatch = metalBatchStats{Encoded: batch.encoded + head.encoded, Commits: 2, Waits: 2}
	return rows, nil
}

func collapseMetalOutputRows(hc, pre, base, scale, out []float32, count int) error {
	if count <= 0 || len(hc) != count*ds4HCWidth || len(pre) != count*4 || len(base) != 4 || len(scale) != 1 || len(out) != count*ds4Width {
		return fmt.Errorf("metal grouped output: invalid shape")
	}
	clear(out)
	for row := range count {
		for stream := range 4 {
			weight := float32(1/(1+math.Exp(-float64(pre[row*4+stream]*scale[0]+base[stream]))) + 1e-6)
			for i, value := range hc[row*ds4HCWidth+stream*ds4Width : row*ds4HCWidth+(stream+1)*ds4Width] {
				out[row*ds4Width+i] += value * weight
			}
		}
	}
	return nil
}
