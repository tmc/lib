package main

import (
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/lm/quant"
)

// expertRow binds the third tensor dimension to an expert, not a token or
// output row. Only formats with source-backed CPU decoders are executable.
func (m *dsparkModel) expertRow(name string, expert, row int) ([]float32, error) {
	t, ok := m.tensor(name)
	if !ok || len(t.Dimensions) != 3 || expert < 0 || expert >= int(t.Dimensions[2]) || row < 0 || row >= int(t.Dimensions[1]) {
		return nil, errShape
	}
	width := int(t.Dimensions[0])
	if t.TypeID == 8 {
		if width%32 != 0 {
			return nil, errShape
		}
		t.Offset += uint64(expert) * t.Dimensions[1] * uint64(width/32*34)
		t.Dimensions = t.Dimensions[:2]
		view := *m
		view.tensors = map[string]gguf.Tensor{name: t}
		return view.denseRow(name, row)
	}
	if width%256 != 0 || (t.TypeID != 10 && t.TypeID != 16) {
		return nil, fmt.Errorf("dspark expert: CPU decoder unavailable for type %d", t.TypeID)
	}
	blockBytes := 84
	if t.TypeID == 16 {
		blockBytes = 66
	}
	rowBytes := uint64(width / 256 * blockBytes)
	offset := m.dataOffset + t.Offset + (uint64(expert)*t.Dimensions[1]+uint64(row))*rowBytes
	if offset > math.MaxInt64 || m.reader == nil {
		return nil, errShape
	}
	bytes := make([]byte, rowBytes)
	n, err := m.reader.ReadAt(bytes, int64(offset))
	if err != nil {
		return nil, fmt.Errorf("dspark expert row: %w", err)
	}
	if n != len(bytes) {
		return nil, io.ErrUnexpectedEOF
	}
	var out []float32
	if t.TypeID == 10 {
		out, err = quant.DequantizeQ2_K(bytes, width)
	} else {
		out, err = quant.DequantizeIQ2_XXS(bytes, width)
	}
	if err != nil {
		return nil, err
	}
	if !allFinite(out) {
		return nil, fmt.Errorf("dspark expert: nonfinite weight")
	}
	return out, nil
}

func (m *dsparkModel) expertMatvec(name string, expert int, input []float32) ([]float32, error) {
	t, ok := m.tensor(name)
	if !ok || len(t.Dimensions) != 3 || len(input) != int(t.Dimensions[0]) || !allFinite(input) {
		return nil, errShape
	}
	out := make([]float32, int(t.Dimensions[1]))
	for row := range out {
		weights, err := m.expertRow(name, expert, row)
		if err != nil {
			return nil, err
		}
		out[row] = dot(weights, input)
	}
	if !allFinite(out) {
		return nil, fmt.Errorf("dspark expert: nonfinite projection")
	}
	return out, nil
}

// dsparkSelect follows ds4.c:11894-11999: bias selects the top six, but the
// normalized weights use unbiased sqrt(softplus(logit)). Lowest IDs win ties.
func dsparkSelect(logits, bias []float32, count int) ([]int, []float32, error) {
	if len(logits) != len(bias) || count < 1 || count > len(logits) || !allFinite(logits, bias) {
		return nil, nil, errShape
	}
	probs := make([]float32, len(logits))
	for i, x := range logits {
		v := float64(x)
		if x < -20 {
			v = math.Exp(v)
		} else if x <= 20 {
			v = math.Log1p(math.Exp(v))
		}
		probs[i] = float32(math.Sqrt(v))
	}
	ids := make([]int, count)
	used := make([]bool, len(logits))
	weights := make([]float32, count)
	var sum float32
	for slot := range ids {
		best := -1
		for i := range logits {
			if !used[i] && (best < 0 || probs[i]+bias[i] > probs[best]+bias[best]) {
				best = i
			}
		}
		ids[slot], used[best], weights[slot] = best, true, probs[best]
		sum += probs[best]
	}
	sum = max(sum, 6.103515625e-5)
	for i := range weights {
		weights[i] = weights[i] / sum * Flash.ExpertWeightScale
	}
	return ids, weights, nil
}

func dsparkSwiGLU(gate, up []float32, weight float32) ([]float32, error) {
	if len(gate) != len(up) || !allFinite(gate, up, []float32{weight}) {
		return nil, errShape
	}
	out := make([]float32, len(gate))
	for i, g := range gate {
		g = min(g, 10) // The negative gate is intentionally not clipped.
		u := min(max(up[i], -10), 10)
		out[i] = g * sigmoid(g) * u * weight
	}
	return out, nil
}

// stageFFN executes admitted routed and shared expert weights and the HC
// combine (ds4.c:34378 and 12760-12830). It is a float-activation CPU reference,
// not the CUDA quantized-activation production path or a performance arm.
func (s *dsparkState) stageFFN(stage int, input [][]float32) ([][]float32, error) {
	if s == nil || stage < 0 || stage >= s.model.stages || len(input) != s.model.block {
		return nil, errShape
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	bias, err := s.model.denseRow(name("exp_probs_b.bias"), 0)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(input))
	for row, residual := range input {
		norm, post, combine, err := s.model.hcPre(stage, residual, true)
		if err != nil {
			return nil, err
		}
		logits, err := s.model.denseMatvec(name("ffn_gate_inp.weight"), norm)
		if err != nil {
			return nil, err
		}
		ids, weights, err := dsparkSelect(logits, bias, Flash.ExpertsUsed)
		if err != nil {
			return nil, err
		}
		ffn := make([]float32, Flash.Embed)
		for slot, expert := range ids {
			gate, err := s.model.expertMatvec(name("ffn_gate_exps.weight"), expert, norm)
			if err != nil {
				return nil, err
			}
			up, err := s.model.expertMatvec(name("ffn_up_exps.weight"), expert, norm)
			if err != nil {
				return nil, err
			}
			mid, err := dsparkSwiGLU(gate, up, weights[slot])
			if err != nil {
				return nil, err
			}
			down, err := s.model.expertMatvec(name("ffn_down_exps.weight"), expert, mid)
			if err != nil {
				return nil, err
			}
			for i := range ffn {
				ffn[i] += down[i]
			}
		}
		gate, err := s.model.denseMatvec(name("ffn_gate_shexp.weight"), norm)
		if err != nil {
			return nil, err
		}
		up, err := s.model.denseMatvec(name("ffn_up_shexp.weight"), norm)
		if err != nil {
			return nil, err
		}
		mid, err := dsparkSwiGLU(gate, up, 1)
		if err != nil {
			return nil, err
		}
		shared, err := s.model.denseMatvec(name("ffn_down_shexp.weight"), mid)
		if err != nil {
			return nil, err
		}
		for i := range ffn {
			ffn[i] += shared[i]
		}
		out[row], err = hcPost(ffn, residual, post, combine)
		if err != nil {
			return nil, fmt.Errorf("dspark FFN HC output: %w", err)
		}
		if !allFinite(out[row]) {
			return nil, fmt.Errorf("dspark FFN: nonfinite HC output")
		}
	}
	return out, nil
}
