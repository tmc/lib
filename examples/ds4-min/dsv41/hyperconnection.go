package dsv41

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
)

// hcInput computes a new sublayer control split, but collapses with the
// preceding sublayer's pre weights. This is the literal scalar ds41 schedule:
// before attention uses g.pre; before MoE uses g.attn_split, not g.ffn_split.
func (m *Model) hcInput(layer int, ffn bool, residual, pre []float32) (norm, split []float32, err error) {
	if m == nil || m.catalog == nil || layer < 0 || layer >= 40 || len(residual) != 4*5120 || len(pre) != 4 {
		return nil, nil, fmt.Errorf("invalid V4.1 HC input")
	}
	prefix := fmt.Sprintf("blk.%d.", layer)
	kind := "hc_attn"
	normName := "attn_norm"
	if ffn {
		kind = "hc_ffn"
		normName = "ffn_norm"
	}
	flat := make([]float32, len(residual))
	if err := dsv4cpu.RMSNorm(flat, residual, nil, 1e-20); err != nil {
		return nil, nil, err
	}
	matrix, err := m.catalog.Matrix(prefix+kind+"_fn.weight", 0)
	if err != nil {
		return nil, nil, err
	}
	mix := make([]float32, 24)
	if err := matrix.Mul(mix, flat); err != nil {
		return nil, nil, err
	}
	scale, err := m.vectorRows(prefix+kind+"_scale.weight", 1, 3)
	if err != nil {
		return nil, nil, err
	}
	base, err := m.vectorRows(prefix+kind+"_base.weight", 1, 24)
	if err != nil {
		return nil, nil, err
	}
	split = make([]float32, 24)
	if err := dsv4cpu.Split(split, mix, scale, base, 4, 20, 1e-6); err != nil {
		return nil, nil, err
	}
	x := make([]float32, 5120)
	if err := dsv4cpu.Collapse(x, residual, pre); err != nil {
		return nil, nil, err
	}
	if err := Quantize(x, 5120, BFloat16); err != nil {
		return nil, nil, err
	}
	weight, err := m.vectorRows(prefix+normName+".weight", 1, 5120)
	if err != nil {
		return nil, nil, err
	}
	norm = make([]float32, 5120)
	if err := dsv4cpu.RMSNorm(norm, x, weight, 1e-20); err != nil {
		return nil, nil, err
	}
	if err := Quantize(norm, 5120, BFloat16); err != nil {
		return nil, nil, err
	}
	return norm, split, nil
}

func hcExpand(dst, block, residual, split []float32) error {
	if len(split) != 24 {
		return fmt.Errorf("invalid V4.1 HC split")
	}
	if err := dsv4cpu.Combine(dst, block, residual, split[4:8], split[8:]); err != nil {
		return err
	}
	return Quantize(dst, len(block), BFloat16)
}
