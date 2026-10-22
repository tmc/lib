//go:build linux

package main

import (
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type shapeAttentionWeights struct {
	qa, qb, kv, outA, outB cudaModelTensor
	qaNorm, kvNorm         cudaModelTensor
}

type shapeAttentionQuantBuffers struct {
	input, qr, heads, low cudamem.Slice[byte]
}

// This route admits raw Q8_0 projections and F32 norms only. Wider GGUF type
// support must provide its own executable kernels and arithmetic controls.
func validateShapeAttentionWeights(c dsv4shape.Config, b *dsv4shape.Binding, layer int, w shapeAttentionWeights) error {
	if _, err := planShapeForward(c, b, 1); err != nil {
		return err
	}
	return validateShapeAttentionWeightsAdmitted(c, b, layer, w)
}

// The caller must have admitted the unchanged binding with planShapeForward.
func validateShapeAttentionWeightsAdmitted(c dsv4shape.Config, b *dsv4shape.Binding, layer int, w shapeAttentionWeights) error {
	if _, err := c.Layer(layer); err != nil {
		return err
	}
	specs, err := c.Weights()
	if err != nil {
		return err
	}
	byName := make(map[string]dsv4shape.TensorSpec)
	for _, spec := range specs {
		byName[spec.Name] = spec
	}
	views := make(map[string]dsv4shape.WeightView)
	for _, v := range b.Weights {
		views[v.Name] = v
	}
	for _, entry := range []struct {
		name   string
		kind   uint32
		tensor cudaModelTensor
	}{
		{"attn_q_a.weight", 8, w.qa}, {"attn_q_b.weight", 8, w.qb}, {"attn_kv.weight", 8, w.kv}, {"attn_output_a.weight", 8, w.outA}, {"attn_output_b.weight", 8, w.outB}, {"attn_q_a_norm.weight", 0, w.qaNorm}, {"attn_kv_a_norm.weight", 0, w.kvNorm},
	} {
		name := fmt.Sprintf("blk.%d.%s", layer, entry.name)
		v, ok := views[name]
		t := entry.tensor
		if !ok || v.Type != entry.kind || t.name != name || t.typeID != v.Type || t.device == 0 || t.bytes != v.Bytes || t.fileOffset != v.Offset || !slices.Equal(t.dimensions, byName[name].Dimensions) {
			return fmt.Errorf("cuda shape attention: invalid uploaded projection %s", name)
		}
	}
	return nil
}

func checkShapeAttentionQuantExtents(c dsv4shape.Config, q shapeAttentionQuantBuffers) error {
	if q.input.Len() != c.Embed/32*36 || q.qr.Len() != c.QLoRA/32*36 || q.heads.Len() != c.Heads*c.HeadDim/32*36 || q.low.Len() != c.OutGroups*c.OutLoRA/32*36 {
		return errors.New("cuda shape attention: invalid quantized scratch extents")
	}
	var ranges []mxfp4DeviceRange
	for _, v := range []cudamem.Slice[byte]{q.input, q.qr, q.heads, q.low} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len())})
	}
	return checkMXFP4DeviceRanges(ranges...)
}

// encodeShapeQKV fills the actual State q and kv vectors. RoPE and raw/
// compressed cache emission are separate required consumers, not inferred here.
func encodeShapeQKV(state *cudaShapeForward, binding *dsv4shape.Binding, layer int, w shapeAttentionWeights, quant shapeAttentionQuantBuffers) error {
	if state == nil || state.closed {
		return errors.New("cuda shape attention: closed state")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	c := state.plan.config
	if err := validateShapeAttentionWeights(c, binding, layer, w); err != nil {
		return err
	}
	if err := checkShapeAttentionQuantExtents(c, quant); err != nil {
		return err
	}
	for _, v := range []struct {
		name string
		n    int
	}{{"attn_norm", c.Embed}, {"qr", c.QLoRA}, {"qr_norm", c.QLoRA}, {"kv_raw", c.HeadDim}, {"kv", c.HeadDim}, {"q", c.Heads * c.HeadDim}} {
		if state.scratch[v.name].Len() != v.n || state.scratch[v.name].Ptr() == 0 {
			return errors.New("cuda shape attention: invalid QKV State extent")
		}
	}
	r := state.r
	if err := r.encodeQuantQ81(state.scratch["attn_norm"], quant.input, c.Embed); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81(w.qa, quant.input, state.scratch["qr"]); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81(w.kv, quant.input, state.scratch["kv_raw"]); err != nil {
		return err
	}
	if err := r.encodeRMS(state.scratch["qr"], w.qaNorm, state.scratch["qr_norm"], 1, true); err != nil {
		return err
	}
	if err := r.encodeRMS(state.scratch["kv_raw"], w.kvNorm, state.scratch["kv"], 1, true); err != nil {
		return err
	}
	if err := r.encodeQuantQ81(state.scratch["qr_norm"], quant.qr, c.QLoRA); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81(w.qb, quant.qr, state.scratch["q"]); err != nil {
		return err
	}
	return r.encodeRMS(state.scratch["q"], cudaModelTensor{}, state.scratch["q"], c.Heads, false)
}

// encodeShapeAttentionOutput consumes actual heads, applies the source-derived
// group-diagonal A projection, then projects all low-rank groups back to Embed.
func encodeShapeAttentionOutput(state *cudaShapeForward, binding *dsv4shape.Binding, layer int, w shapeAttentionWeights, quant shapeAttentionQuantBuffers) error {
	if state == nil || state.closed {
		return errors.New("cuda shape attention: closed state")
	}
	if err := state.r.checkOpen(); err != nil {
		return err
	}
	c := state.plan.config
	if err := validateShapeAttentionWeights(c, binding, layer, w); err != nil {
		return err
	}
	if err := checkShapeAttentionQuantExtents(c, quant); err != nil {
		return err
	}
	for _, v := range []struct {
		name string
		n    int
	}{{"heads", c.Heads * c.HeadDim}, {"attn_low", c.OutGroups * c.OutLoRA}, {"attn_out", c.Embed}} {
		if state.scratch[v.name].Len() != v.n || state.scratch[v.name].Ptr() == 0 {
			return errors.New("cuda shape attention: invalid output State extent")
		}
	}
	r := state.r
	if err := r.encodeQuantQ81(state.scratch["heads"], quant.heads, c.Heads*c.HeadDim); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQuantizedBlockDiag(w.outA, quant.heads, state.scratch["attn_low"], c.OutGroups); err != nil {
		return err
	}
	if err := r.encodeQuantQ81(state.scratch["attn_low"], quant.low, c.OutGroups*c.OutLoRA); err != nil {
		return err
	}
	return r.encodeQ80MatvecQ81(w.outB, quant.low, state.scratch["attn_out"])
}
