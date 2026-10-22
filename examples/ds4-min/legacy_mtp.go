package main

import (
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/gguf"
)

// legacyMTPModel holds a legacy Flash support directory. Its raw attention
// state belongs to the request, never to this immutable weight directory.
type legacyMTPModel struct{ weights *dsparkModel }

func readLegacyMTPModel(reader io.ReaderAt, size int64, sha string) (*legacyMTPModel, error) {
	if reader == nil || len(sha) != 64 {
		return nil, fmt.Errorf("legacy mtp: reader and full support sha256 required")
	}
	if err := gguf.CheckSHA256(reader, size, sha); err != nil {
		return nil, fmt.Errorf("legacy mtp support: %w", err)
	}
	g, err := gguf.Read(reader, size)
	if err != nil {
		return nil, fmt.Errorf("legacy mtp support: %w", err)
	}
	m, err := admitLegacyMTPModel(g)
	if err != nil {
		return nil, err
	}
	m.weights.reader, m.weights.sha256 = reader, sha
	return m, nil
}

func admitLegacyMTPModel(g *gguf.File) (*legacyMTPModel, error) {
	if g == nil {
		return nil, fmt.Errorf("legacy mtp: missing container")
	}
	m := &dsparkModel{dataOffset: g.DataOffset, tensors: make(map[string]gguf.Tensor)}
	for _, t := range g.Tensors {
		if _, exists := m.tensors[t.Name]; exists {
			return nil, fmt.Errorf("legacy mtp: duplicate tensor %s", t.Name)
		}
		if t.Name == "mtp.0.main_proj.weight" {
			return nil, fmt.Errorf("legacy mtp: dspark support is not legacy mtp")
		}
		t.Dimensions = append([]uint64(nil), t.Dimensions...)
		m.tensors[t.Name] = t
	}
	rules := dsparkRules(1, m)
	e, h := uint64(Flash.Embed), uint64(Flash.HC)
	rules = append(rules,
		dsparkRule{"hc_head_base.weight", 0, []uint64{h}}, dsparkRule{"hc_head_fn.weight", 1, []uint64{e * h, h}},
		dsparkRule{"hc_head_scale.weight", 0, []uint64{1}}, dsparkRule{"e_proj.weight", 2, []uint64{e, e}},
		dsparkRule{"h_proj.weight", 2, []uint64{e, e}}, dsparkRule{"enorm.weight", 0, []uint64{e}},
		dsparkRule{"hnorm.weight", 0, []uint64{e}}, dsparkRule{"norm.weight", 0, []uint64{e}})
	for _, r := range rules {
		name := "mtp.0." + r.name
		t, ok := m.tensors[name]
		typeOK := r.accepts(t.TypeID)
		if r.kind == 2 {
			typeOK = t.TypeID == 8
		}
		if r.name == "ffn_gate_inp.weight" {
			typeOK = t.TypeID == 0 || t.TypeID == 1
		}
		if !ok || !typeOK || len(t.Dimensions) != len(r.dims) {
			return nil, fmt.Errorf("legacy mtp: invalid layout for %s", name)
		}
		for i, n := range r.dims {
			if t.Dimensions[i] != n {
				return nil, fmt.Errorf("legacy mtp: invalid dimension for %s", name)
			}
		}
		n, err := t.Bytes()
		if err != nil {
			return nil, fmt.Errorf("legacy mtp %s: %w", name, err)
		}
		if g.DataOffset > g.Size || t.Offset > g.Size-g.DataOffset || n > g.Size-g.DataOffset-t.Offset {
			return nil, fmt.Errorf("legacy mtp: %s outside file", name)
		}
	}
	if m.tensors["mtp.0.ffn_gate_exps.weight"].TypeID != m.tensors["mtp.0.ffn_up_exps.weight"].TypeID {
		return nil, fmt.Errorf("legacy mtp: gate/up quant types differ")
	}
	return &legacyMTPModel{weights: m}, nil
}

// input implements metal_graph_eval_mtp_draft_from_hc's embedding/HC merge
// (ds4.c:35618). It does not borrow or mutate target hidden storage.
func (m *legacyMTPModel) input(embedding, hidden []float32) ([]float32, error) {
	if m == nil || m.weights == nil || len(embedding) != Flash.Embed || len(hidden) != Flash.Embed*Flash.HC || !allFinite(embedding) || !allFinite(hidden) {
		return nil, errShape
	}
	norm := func(x []float32, name string) ([]float32, error) {
		w, err := m.weights.denseRow(name, 0)
		if err != nil {
			return nil, err
		}
		var sum float64
		for _, v := range x {
			sum += float64(v) * float64(v)
		}
		scale := 1 / math.Sqrt(sum/float64(len(x))+float64(Flash.RMSEps))
		y := make([]float32, len(x))
		for i, v := range x {
			y[i] = float32(float64(v)*scale) * w[i]
		}
		return y, nil
	}
	e, err := norm(embedding, "mtp.0.enorm.weight")
	if err != nil {
		return nil, err
	}
	e, err = m.weights.denseMatvec("mtp.0.e_proj.weight", e)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(hidden))
	for row := 0; row < Flash.HC; row++ {
		h, err := norm(hidden[row*Flash.Embed:(row+1)*Flash.Embed], "mtp.0.hnorm.weight")
		if err != nil {
			return nil, err
		}
		h, err = m.weights.denseMatvec("mtp.0.h_proj.weight", h)
		if err != nil {
			return nil, err
		}
		for i := range h {
			out[row*Flash.Embed+i] = e[i] + h[i]
		}
	}
	if !allFinite(out) {
		return nil, fmt.Errorf("legacy mtp: nonfinite input")
	}
	return out, nil
}

func legacyMTPDraftLength(n int) int {
	if n <= 0 {
		return 1
	}
	if n > 16 {
		return 16
	}
	return n
}

// legacyMTPMayDraft follows the sampled dispatch at ds4.c:78785-78847.
// Opportunistic and exact stochastic support verification are DSpark-only;
// legacy MTP emits the normally sampled target token when temperature>0.
func legacyMTPMayDraft(temperature float32) (bool, error) {
	if temperature < 0 || math.IsNaN(float64(temperature)) || math.IsInf(float64(temperature), 0) {
		return false, errShape
	}
	return temperature == 0, nil
}

// headInput implements the legacy HC output head up to the shared target
// vocabulary projection (ds4.c:27593). Its sigmoid weights are independent;
// normalizing them with softmax would change the trained support model.
func (m *legacyMTPModel) headInput(hidden []float32) ([]float32, error) {
	if m == nil || m.weights == nil || len(hidden) != Flash.Embed*Flash.HC || !allFinite(hidden) {
		return nil, errShape
	}
	ones := make([]float32, len(hidden))
	for i := range ones {
		ones[i] = 1
	}
	flat, err := rmsNorm(hidden, ones, Flash.RMSEps)
	if err != nil {
		return nil, err
	}
	pre, err := m.weights.denseMatvec("mtp.0.hc_head_fn.weight", flat)
	if err != nil {
		return nil, err
	}
	scale, err := m.weights.denseRow("mtp.0.hc_head_scale.weight", 0)
	if err != nil {
		return nil, err
	}
	base, err := m.weights.denseRow("mtp.0.hc_head_base.weight", 0)
	if err != nil {
		return nil, err
	}
	if len(pre) != Flash.HC || len(base) != Flash.HC || len(scale) != 1 {
		return nil, errShape
	}
	weights := make([]float32, Flash.HC)
	for i, v := range pre {
		weights[i] = float32(1/(1+math.Exp(-float64(v*scale[0]+base[i])))) + Flash.HCEps
	}
	embedding, err := hcWeightedSum(hidden, weights, Flash.Embed)
	if err != nil {
		return nil, err
	}
	norm, err := m.weights.denseRow("mtp.0.norm.weight", 0)
	if err != nil {
		return nil, err
	}
	out, err := rmsNorm(embedding, norm, Flash.RMSEps)
	if err == nil && !allFinite(out) {
		err = fmt.Errorf("legacy mtp: nonfinite head")
	}
	return out, err
}

// legacyMTPMargin admits two-row verification only when the final support
// logits have sufficient top-two separation (ds4.c:78187).
func legacyMTPMargin(logits []float32, drafts int, threshold float32) (bool, error) {
	if math.IsNaN(float64(threshold)) || math.IsInf(float64(threshold), 0) || threshold < 0 {
		return false, errShape
	}
	if drafts != 2 || threshold == 0 {
		return true, nil
	}
	if len(logits) < 2 || !allFinite(logits) {
		return false, errShape
	}
	a, b := float32(math.Inf(-1)), float32(math.Inf(-1))
	for _, v := range logits {
		if v > a {
			b, a = a, v
		} else if v > b {
			b = v
		}
	}
	return a-b >= threshold, nil
}
