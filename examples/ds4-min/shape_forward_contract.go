package main

import (
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/gguf"
)

type shapeForwardBuffer struct {
	name     string
	elements int
}

type shapeForwardPlan struct {
	config  dsv4shape.Config
	context uint64
	buffers []shapeForwardBuffer
	bytes   uint64
	norms   map[string]dsv4shape.WeightView
}

// planShapeForward covers the float scratch roles in cpu_decode_scratch_init
// at 11e8cad. Quantized CPU temporaries are not CUDA operand layouts and are
// deliberately excluded; a later consumer must admit its own quantized ABI.
func planShapeForward(c dsv4shape.Config, binding *dsv4shape.Binding, context uint64) (shapeForwardPlan, error) {
	if err := c.Validate(); err != nil {
		return shapeForwardPlan{}, err
	}
	if binding == nil || context == 0 || context > math.MaxUint32 {
		return shapeForwardPlan{}, fmt.Errorf("cuda shape forward: invalid binding or context")
	}
	p := shapeForwardPlan{config: c, context: context, norms: make(map[string]dsv4shape.WeightView)}
	weights := make(map[string]dsv4shape.WeightView)
	for _, w := range binding.Weights {
		if _, ok := weights[w.Name]; ok {
			return shapeForwardPlan{}, fmt.Errorf("cuda shape forward: duplicate weight %s", w.Name)
		}
		weights[w.Name] = w
	}
	specs, err := c.Weights()
	if err != nil {
		return shapeForwardPlan{}, err
	}
	for _, spec := range specs {
		w, ok := weights[spec.Name]
		if !ok && spec.Optional {
			continue
		}
		if !ok || !slices.Contains(spec.Types, w.Type) {
			return shapeForwardPlan{}, fmt.Errorf("cuda shape forward: incompatible binding %s", spec.Name)
		}
		t := gguf.Tensor{TypeID: w.Type, Dimensions: spec.Dimensions}
		n, err := t.Bytes()
		if err != nil {
			return shapeForwardPlan{}, err
		}
		t.Dimensions = spec.Dimensions[:1]
		row, err := t.Bytes()
		if err != nil {
			return shapeForwardPlan{}, err
		}
		expert := uint64(0)
		if len(spec.Dimensions) == 3 {
			expert = n / spec.Dimensions[2]
		}
		if w.Bytes != n || w.RowBytes != row || w.ExpertBytes != expert || w.Offset > math.MaxUint64-w.Bytes {
			return shapeForwardPlan{}, fmt.Errorf("cuda shape forward: incompatible binding extent %s", spec.Name)
		}
	}
	for i := 0; i < c.Layers; i++ {
		for _, phase := range []string{"attn", "ffn"} {
			name := fmt.Sprintf("blk.%d.%s_norm.weight", i, phase)
			w, ok := weights[name]
			if !ok || w.Type != 0 || w.Bytes != uint64(c.Embed)*4 || w.RowBytes != w.Bytes {
				return shapeForwardPlan{}, fmt.Errorf("cuda shape forward: missing or invalid norm %s", name)
			}
			p.norms[name] = w
		}
	}
	e, h, q, low := uint64(c.Embed), uint64(c.HC*c.Embed), uint64(c.Heads*c.HeadDim), uint64(c.OutGroups*c.OutLoRA)
	raw := context
	if raw > uint64(c.SlidingWindow) {
		raw = uint64(c.SlidingWindow)
	}
	comp := context/4 + 2
	for _, b := range []struct {
		name string
		n    uint64
	}{
		{"plain", e}, {"cur", h}, {"next", h}, {"attn_cur", e}, {"attn_norm", e}, {"attn_residual", h},
		{"q", q}, {"qr", uint64(c.QLoRA)}, {"qr_norm", uint64(c.QLoRA)}, {"kv_raw", uint64(c.HeadDim)}, {"kv", uint64(c.HeadDim)},
		{"heads", q}, {"attn_low", low}, {"attn_out", e}, {"after_attn_hc", h}, {"attn_score", raw + comp},
		{"comp", uint64(c.HeadDim)}, {"index_comp", uint64(c.IndexerHeadDim)}, {"comp_kv_cur", uint64(2 * c.HeadDim)}, {"comp_sc_cur", uint64(2 * c.HeadDim)}, {"comp_pooled", uint64(c.HeadDim)},
		{"index_q", uint64(c.IndexerHeads * c.IndexerHeadDim)}, {"index_weights", uint64(c.IndexerHeads)}, {"index_scores", comp},
		{"ffn_cur", e}, {"ffn_norm", e}, {"ffn_moe", e}, {"ffn_shared", e}, {"ffn_out", e},
		{"shared_gate", uint64(c.ExpertFF)}, {"shared_up", uint64(c.ExpertFF)}, {"shared_mid", uint64(c.ExpertFF)}, {"routed_mid_all", uint64(c.ExpertsUsed * c.ExpertFF)},
		{"hc_flat", h}, {"output_flat", h}, {"output_pre", uint64(c.HC)}, {"output_weights", uint64(c.HC)}, {"output_embd", e}, {"output_norm", e},
	} {
		if b.n > uint64(int(^uint(0)>>1))/4 || b.n*4 > math.MaxUint64-p.bytes {
			return shapeForwardPlan{}, fmt.Errorf("cuda shape forward: scratch overflow")
		}
		p.buffers = append(p.buffers, shapeForwardBuffer{b.name, int(b.n)})
		p.bytes += b.n * 4
	}
	return p, nil
}
