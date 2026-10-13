package main

import (
	"fmt"
	"math"
)

// ffnPlan describes one dense, single-token float32 FFN on the host. It does
// not allocate device memory or admit a kernel launch. All dimensions and
// storage counts are private so callers construct a plan with newFFNPlan.
type ffnPlan struct {
	hidden, intermediate      int
	epsilon                   float32
	weightWords, scratchWords int
}

// ffnBuffers contains read-only inputs. Matrices are output-major: each output
// owns a contiguous row of input coefficients. Exact lengths are required.
type ffnBuffers struct {
	x, norm, gate, up, down []float32
}

// newFFNPlan checks all element and byte counts before any allocation. Scratch
// accounts for separate normalized, gate, up, activation and down-result
// vectors (2*hidden+3*intermediate float32 words). Output is separately owned;
// it is not included in scratchBytes. Negative budgets are invalid.
func newFFNPlan(hidden, intermediate int, epsilon float32, scratchBytes int) (ffnPlan, error) {
	if hidden <= 0 || intermediate <= 0 {
		return ffnPlan{}, fmt.Errorf("ffn dimensions must be positive")
	}
	if epsilon <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return ffnPlan{}, fmt.Errorf("ffn epsilon must be finite and positive")
	}
	maxWords := int(^uint(0)>>1) / 4
	if hidden > maxWords || intermediate > maxWords || hidden > maxWords/intermediate {
		return ffnPlan{}, fmt.Errorf("ffn matrix byte size exceeds int range")
	}
	if hidden > maxWords/2 || intermediate > (maxWords-2*hidden)/3 {
		return ffnPlan{}, fmt.Errorf("ffn scratch byte size exceeds int range")
	}
	words := 2*hidden + 3*intermediate
	if scratchBytes < words*4 {
		return ffnPlan{}, fmt.Errorf("ffn scratch needs %d bytes, have %d", words*4, scratchBytes)
	}
	return ffnPlan{hidden: hidden, intermediate: intermediate, epsilon: epsilon,
		weightWords: hidden * intermediate, scratchWords: words}, nil
}

// stages returns a detached dependency description. Each stage waits for all
// listed predecessors. Gate and up share normalization but not each other's
// output. This is a logical host plan, not a GPU synchronization contract.
func (p ffnPlan) stages() []ffnStage {
	return []ffnStage{
		{name: "normalize"},
		{name: "gate", after: []string{"normalize"}},
		{name: "up", after: []string{"normalize"}},
		{name: "activate", after: []string{"gate", "up"}},
		{name: "down", after: []string{"activate"}},
		{name: "residual", after: []string{"down"}},
	}
}

type ffnStage struct {
	name  string
	after []string
}

func (p ffnPlan) validate(b ffnBuffers) error {
	// Reconstruct to refuse zero or internally inconsistent plans before slicing.
	checked, err := newFFNPlan(p.hidden, p.intermediate, p.epsilon, int(^uint(0)>>1))
	if err != nil {
		return err
	}
	if checked != p {
		return fmt.Errorf("ffn plan storage counts are inconsistent")
	}
	for _, v := range []struct {
		name      string
		got, want int
	}{
		{"input", len(b.x), p.hidden}, {"norm", len(b.norm), p.hidden},
		{"gate", len(b.gate), p.weightWords}, {"up", len(b.up), p.weightWords},
		{"down", len(b.down), p.weightWords},
	} {
		if v.got != v.want {
			return fmt.Errorf("ffn %s length is %d, want %d", v.name, v.got, v.want)
		}
	}
	return nil
}

// reference returns a new residual output without changing any input. All
// intermediate regions are private to this call and retained until it returns.
// This host reference makes no numerical or performance claim about a GPU.
func (p ffnPlan) reference(b ffnBuffers) ([]float32, error) {
	if err := p.validate(b); err != nil {
		return nil, err
	}
	scratch := make([]float32, p.scratchWords)
	normalized := scratch[:p.hidden]
	gate := scratch[p.hidden : p.hidden+p.intermediate]
	up := scratch[p.hidden+p.intermediate : p.hidden+2*p.intermediate]
	activation := scratch[p.hidden+2*p.intermediate : p.hidden+3*p.intermediate]
	down := scratch[p.hidden+3*p.intermediate:]
	var sum float32
	for _, v := range b.x {
		sum += v * v
	}
	scale := float32(1 / math.Sqrt(float64(sum/float32(p.hidden)+p.epsilon)))
	for i := range normalized {
		normalized[i] = b.x[i] * scale * b.norm[i]
	}
	for o := 0; o < p.intermediate; o++ {
		for i, x := range normalized {
			gate[o] += x * b.gate[o*p.hidden+i]
			up[o] += x * b.up[o*p.hidden+i]
		}
		activation[o] = gate[o] / (1 + float32(math.Exp(float64(-gate[o])))) * up[o]
	}
	out := make([]float32, p.hidden)
	for o := range out {
		for i, x := range activation {
			down[o] += x * b.down[o*p.intermediate+i]
		}
		out[o] = b.x[o] + down[o]
	}
	return out, nil
}
