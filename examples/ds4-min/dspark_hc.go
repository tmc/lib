package main

import "fmt"

// hcPre executes the real support stage's HC split/collapse and weighted RMS
// using its admitted tensors (ds4.c:34162-34211). CPU reduction order is a
// correctness reference, not a GPU bit-identity or performance claim.
func (m *dsparkModel) hcPre(stage int, state []float32, ffn bool) (normalized, post, combine []float32, err error) {
	if m == nil || stage < 0 || stage >= m.stages || len(state) != Flash.Embed*Flash.HC || !allFinite(state) {
		return nil, nil, nil, errShape
	}
	prefix, normName := "hc_attn", "attn_norm.weight"
	if ffn {
		prefix, normName = "hc_ffn", "ffn_norm.weight"
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	ones := make([]float32, len(state))
	for i := range ones {
		ones[i] = 1
	}
	flat, err := rmsNorm(state, ones, Flash.RMSEps)
	if err != nil {
		return nil, nil, nil, err
	}
	mix, err := m.denseMatvec(name(prefix+"_fn.weight"), flat)
	if err != nil {
		return nil, nil, nil, err
	}
	scale, err := m.denseRow(name(prefix+"_scale.weight"), 0)
	if err != nil {
		return nil, nil, nil, err
	}
	base, err := m.denseRow(name(prefix+"_base.weight"), 0)
	if err != nil {
		return nil, nil, nil, err
	}
	pre, post, combine, err := hcSplit(mix, scale, base, Flash.HC, Flash.SinkhornIters, Flash.HCEps)
	if err != nil {
		return nil, nil, nil, err
	}
	collapsed, err := hcWeightedSum(state, pre, Flash.Embed)
	if err != nil {
		return nil, nil, nil, err
	}
	norm, err := m.denseRow(name(normName), 0)
	if err != nil {
		return nil, nil, nil, err
	}
	normalized, err = rmsNorm(collapsed, norm, Flash.RMSEps)
	if err == nil && !allFinite(normalized, post, combine) {
		err = fmt.Errorf("dspark HC: nonfinite output")
	}
	return normalized, post, combine, err
}
