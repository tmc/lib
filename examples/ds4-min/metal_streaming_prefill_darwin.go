//go:build darwin && arm64

package main

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

type metalStreamingFrontier struct {
	hc, norm, gate, up, weights []float32
	ids                         []int32
}

func (r *metalRuntime) streamingPrefillEligible(tokens int) bool {
	if r.streaming == nil || r.collectors != nil {
		return false
	}
	for _, l := range r.layers {
		if !dsv4expertcache.MetalPrefillEnabled(uint32(tokens), uint32(l.expertGate.dimensions[2]), 6, l.expertGate.typeID, l.expertDown.typeID, uint32(r.streaming.capacity), false) {
			return false
		}
	}
	return true
}

// evalStreamingPrefill is layer-major: finish every row's attention/router,
// reserve the entire selected set once, then consume every row before releasing
// the cache lease. Expert kernels retain the scalar arithmetic order; this is
// selected-address batching, not a claim of fused batched-kernel performance.
func (r *metalRuntime) evalStreamingPrefill(tokens []int, position int) (err error) {
	if err := r.validateStreamingBatch(tokens, position); err != nil {
		return err
	}
	if position != r.pos || !r.streamingPrefillEligible(len(tokens)) {
		return errors.New("Metal streaming prefill not admitted")
	}
	x := new(metalGroupedExecution)
	r.streamingExecution = x
	defer func() {
		r.streamingFront, r.streamingHold = false, false
		r.streamingExecution = nil
		err = r.finishStreamingExecution(x, err)
	}()
	hc := make([][]float32, len(tokens))
	for row, token := range tokens {
		batch, e := newMetalBatch(r.ops)
		if e != nil {
			return e
		}
		x.batches = append(x.batches, batch)
		if e = batch.EncodeEmbedding(r.output.embedding, token, r.state.hc[0]); e != nil {
			return e
		}
		if e = batch.Commit(); e != nil {
			return e
		}
		if e = batch.Wait(context.Background()); e != nil {
			return e
		}
		hc[row] = append([]float32(nil), r.state.hc[0].HostSlice()...)
	}
	var frontier []metalStreamingFrontier
	var allIDs []int32
	var layerStarted time.Time
	err = runMetalStreamingRows(len(r.layers), len(tokens), func(layer, row int) error {
		if row == 0 {
			layerStarted = r.power.start()
			frontier = make([]metalStreamingFrontier, len(tokens))
			allIDs = make([]int32, 0, len(tokens)*6)
			r.streamingFront = true
		}
		token := tokens[row]
		copy(r.state.hc[0].HostSlice(), hc[row])
		batch, e := newMetalBatch(r.ops)
		if e != nil {
			return e
		}
		x.batches = append(x.batches, batch)
		if _, e = r.evalLayer(batch, layer, int32(token), position+row); e != nil {
			return fmt.Errorf("streaming prefill layer %d row %d: %w", layer, row, e)
		}
		frontier[row] = metalStreamingFrontier{
			hc:      append([]float32(nil), r.state.hc[1].HostSlice()...),
			norm:    append([]float32(nil), r.state.ffnNorm.HostSlice()...),
			gate:    append([]float32(nil), r.state.sharedGate.HostSlice()...),
			up:      append([]float32(nil), r.state.sharedUp.HostSlice()...),
			weights: append([]float32(nil), r.state.expertW.HostSlice()...),
			ids:     append([]int32(nil), r.state.expertIDs.HostSlice()...),
		}
		allIDs = append(allIDs, frontier[row].ids...)
		return nil
	}, func(layer int) error {
		r.streamingFront = false
		l := &r.layers[layer]
		if e := r.streaming.prepare(layer, [3]boundMetalTensor{l.expertGate, l.expertUp, l.expertDown}, allIDs); e != nil {
			return e
		}
		r.streamingHold = true
		return nil
	}, func(layer, row int) error {
		f := frontier[row]
		copy(r.state.hc[1].HostSlice(), f.hc)
		copy(r.state.ffnNorm.HostSlice(), f.norm)
		copy(r.state.sharedGate.HostSlice(), f.gate)
		copy(r.state.sharedUp.HostSlice(), f.up)
		copy(r.state.expertW.HostSlice(), f.weights)
		copy(r.state.expertIDs.HostSlice(), f.ids)
		batch, e := newMetalBatch(r.ops)
		if e != nil {
			return e
		}
		x.batches = append(x.batches, batch)
		if _, e = r.evalLayerFFN(batch, layer, r.state.hc[1], true); e != nil {
			return e
		}
		copy(hc[row], r.state.hc[0].HostSlice())
		return nil
	}, func(layer int) error {
		r.streamingHold = false
		if err := r.streaming.finish(); err != nil {
			return err
		}
		// Each row's FFN was committed and waited above; no queued layer work
		// remains at this layer-major completion boundary.
		return r.power.finishLayer(layer, layerStarted, func() error { return nil })
	})
	if err != nil {
		return err
	}
	copy(r.state.hc[0].HostSlice(), hc[len(hc)-1])
	out, err := r.outputLogits(hc[len(hc)-1])
	if err != nil {
		return err
	}
	copy(r.state.logits.HostSlice(), out)
	r.pos += len(tokens)
	return nil
}

func (r *metalRuntime) validateStreamingBatch(tokens []int, position int) error {
	if r.failed != nil {
		return fmt.Errorf("Metal runtime: reset required: %w", r.failed)
	}
	// An empty batch is a no-op at the current frontier, including a full
	// context. No token may start at the full-context frontier.
	if position != r.pos || position < 0 || position > r.ctx || len(tokens) > r.ctx-position {
		return errors.New("Metal runtime: prefill exceeds context")
	}
	for _, token := range tokens {
		if token < 0 || uint64(token) > uint64(1<<31-1) || uint64(token) >= r.output.embedding.dimensions[1] {
			return fmt.Errorf("Metal runtime: token %d out of range", token)
		}
	}
	return nil
}

// finishStreamingExecution is the terminal defer for scalar and layer-major
// streaming. Keep the poison classification created by retained commands.
func (r *metalRuntime) finishStreamingExecution(x *metalGroupedExecution, err error) error {
	err = x.finish(r, err)
	err = errors.Join(err, r.streaming.finish())
	if err != nil {
		r.failed = errors.Join(r.failed, err)
	}
	return err
}
