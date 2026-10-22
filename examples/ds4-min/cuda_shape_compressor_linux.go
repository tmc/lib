//go:build linux

package main

import (
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type shapeCompressorWeights struct {
	kv, gate, ape, norm cudaModelTensor
}

// shapeCompressorProgress records successfully enqueued updates. It is owned
// by the decode State, not by the borrowed cache views. An asynchronous error
// invalidates the State; counts do not certify device completion.
type shapeCompressorProgress struct {
	next uint32
	rows int
}

func validateShapeCompressorWeights(c dsv4shape.Config, binding *dsv4shape.Binding, layer int, indexer bool, w shapeCompressorWeights) error {
	if _, err := planShapeForward(c, binding, 1); err != nil {
		return err
	}
	return validateShapeCompressorWeightsAdmitted(c, binding, layer, indexer, w)
}

// The caller must have admitted the unchanged binding with planShapeForward.
func validateShapeCompressorWeightsAdmitted(c dsv4shape.Config, binding *dsv4shape.Binding, layer int, indexer bool, w shapeCompressorWeights) error {
	l, err := c.Layer(layer)
	if err != nil {
		return err
	}
	if l.CompressRatio == 0 || indexer && !l.Indexer {
		return errors.New("cuda shape compressor: compressor absent from layer")
	}
	prefix := "attn_compressor_"
	if indexer {
		prefix = "indexer_compressor_"
	}
	specs, err := c.Weights()
	if err != nil {
		return err
	}
	for _, e := range []struct {
		name string
		kind uint32
		t    cudaModelTensor
	}{{"kv.weight", 1, w.kv}, {"gate.weight", 1, w.gate}, {"ape.weight", 1, w.ape}, {"norm.weight", 0, w.norm}} {
		name := fmt.Sprintf("blk.%d.%s%s", layer, prefix, e.name)
		var view dsv4shape.WeightView
		found := false
		for _, v := range binding.Weights {
			if v.Name == name {
				view, found = v, true
				break
			}
		}
		var dims []uint64
		for _, spec := range specs {
			if spec.Name == name {
				dims = spec.Dimensions
				break
			}
		}
		if !found || dims == nil || view.Type != e.kind || e.t.name != name || e.t.device == 0 || e.t.typeID != e.kind || e.t.bytes != view.Bytes || e.t.fileOffset != view.Offset || !slices.Equal(e.t.dimensions, dims) {
			return fmt.Errorf("cuda shape compressor: invalid uploaded tensor %s", name)
		}
	}
	return nil
}

// encodeShapeCompressor consumes the actual normalized attention input. It
// reuses the executable 512/128-wide compressor consumers, whose widths and
// ratios are identical in Flash and Pro; it does not use Flash layer indices.
// The RoPE constants in those consumers admit only defaultShapeRope metadata.
func encodeShapeCompressor(state *cudaShapeForward, binding *dsv4shape.Binding, layer int, position uint32, indexer bool, w shapeCompressorWeights, progress *shapeCompressorProgress) error {
	if state == nil || state.closed || progress == nil || state.cache == nil || state.cache.r != state.r || state.cache.plan.layers != state.plan.config.Layers {
		return errors.New("cuda shape compressor: closed or mismatched state")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	c := state.plan.config
	if uint64(position) >= state.plan.context || position != progress.next {
		return errors.New("cuda shape compressor: invalid position or update order")
	}
	if err := validateShapeCompressorWeights(c, binding, layer, indexer, w); err != nil {
		return err
	}
	l, _ := c.Layer(layer)
	head, prefix := c.HeadDim, "attention"
	if indexer {
		head, prefix = c.IndexerHeadDim, "indexer"
	}
	width := head
	if l.CompressRatio == 4 {
		width *= 2
	}
	kv, gate := state.scratch["comp_kv_cur"], state.scratch["comp_sc_cur"]
	if kv.Len() < width || gate.Len() < width || state.scratch["attn_norm"].Len() != c.Embed {
		return errors.New("cuda shape compressor: truncated projection scratch")
	}
	kv, gate = kv.Slice(0, width), gate.Slice(0, width)
	var parts [4]cudamem.Slice[float32]
	for i, suffix := range []string{"cache", "kv", "score", "pooled"} {
		v, err := shapeCacheLayerBuffer(state.cache, layer, prefix+"-"+suffix)
		if err != nil {
			return err
		}
		parts[i] = v
	}
	compressed := cudaCompressedState{cache: parts[0], stateKV: parts[1], stateScore: parts[2], pooled: parts[3], headDim: head, width: width, ratio: l.CompressRatio, cap: parts[0].Len() / head, n: progress.rows}
	if err := state.r.encodeF16PairOrdered(w.kv, w.gate, state.scratch["attn_norm"], kv, gate); err != nil {
		return err
	}
	emit, err := state.r.encodeCompressor(kv, gate, w.ape, w.norm, &compressed, int(position), indexer)
	if err != nil {
		return err
	}
	if emit {
		progress.rows++
	}
	progress.next = position + 1
	return nil
}
