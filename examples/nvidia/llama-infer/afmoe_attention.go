package main

import (
	"fmt"
	"math"
)

// afmoeAttentionShape describes one token's grouped-query attention. Activations
// use float32 storage, but each value represents a BF16 activation. The cache is
// position-major: [position, kv head, head dimension].
type afmoeAttentionShape struct {
	Heads   int
	KVHeads int
	HeadDim int
}

func (s afmoeAttentionShape) validate() error {
	if s.Heads <= 0 || s.KVHeads <= 0 || s.Heads%s.KVHeads != 0 || s.HeadDim <= 0 || s.HeadDim%2 != 0 || s.Heads > math.MaxInt32/s.HeadDim {
		return fmt.Errorf("afmoe attention: invalid heads %d/%d or head dimension %d", s.Heads, s.KVHeads, s.HeadDim)
	}
	return nil
}

// afmoeApplyRoPE applies the reference's split-half rotation in place. Call it
// only for sliding-attention layers; global-attention layers use no RoPE.
func afmoeApplyRoPE(q, k []float32, s afmoeAttentionShape, pos int, theta float32) error {
	if err := s.validate(); err != nil {
		return err
	}
	if len(q) != s.Heads*s.HeadDim || len(k) != s.KVHeads*s.HeadDim || pos < 0 || theta <= 0 || math.IsNaN(float64(theta)) || math.IsInf(float64(theta), 0) {
		return fmt.Errorf("afmoe rope: invalid dimensions, position or theta")
	}
	for _, x := range [][]float32{q, k} {
		for h := 0; h < len(x)/s.HeadDim; h++ {
			for d := 0; d < s.HeadDim/2; d++ {
				inv := float32(1 / math.Pow(float64(theta), float64(2*d)/float64(s.HeadDim)))
				angle := float32(pos) * inv
				c := afmoeBF16(float32(math.Cos(float64(angle))))
				sn := afmoeBF16(float32(math.Sin(float64(angle))))
				i, j := h*s.HeadDim+d, h*s.HeadDim+d+s.HeadDim/2
				a, b := x[i], x[j]
				// The two BF16 products precede the BF16 sum in eager PyTorch.
				x[i] = afmoeBF16(afmoeBF16(a*c) + afmoeBF16(-b*sn))
				x[j] = afmoeBF16(afmoeBF16(b*c) + afmoeBF16(a*sn))
			}
		}
	}
	return nil
}

// afmoeAttentionDecode is a CPU oracle for one eager-attention decode token.
// window zero means global causal attention. A local window includes the current
// token, matching the pinned reference mask. This is not a GPU timing path.
func afmoeAttentionDecode(q, keys, values []float32, s afmoeAttentionShape, pos, window int) ([]float32, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	if pos < 0 || window < 0 || len(q) != s.Heads*s.HeadDim {
		return nil, fmt.Errorf("afmoe attention: invalid query, position or window")
	}
	stride := s.KVHeads * s.HeadDim
	if len(keys) != len(values) || len(keys)%stride != 0 || pos >= len(keys)/stride {
		return nil, fmt.Errorf("afmoe attention: cache does not contain position %d", pos)
	}
	start := afmoeAttentionStart(pos, window)
	scores := make([]float32, pos-start+1)
	probs := make([]float32, len(scores))
	out := make([]float32, len(q))
	scale := float32(1 / math.Sqrt(float64(s.HeadDim)))
	for h := 0; h < s.Heads; h++ {
		kvh := h / (s.Heads / s.KVHeads)
		max := float32(math.Inf(-1))
		for t := start; t <= pos; t++ {
			var dot float32
			for d := 0; d < s.HeadDim; d++ {
				dot += q[h*s.HeadDim+d] * keys[t*stride+kvh*s.HeadDim+d]
			}
			x := afmoeBF16(afmoeBF16(dot) * scale)
			scores[t-start] = x
			if x > max {
				max = x
			}
		}
		var sum float32
		for i, x := range scores {
			probs[i] = float32(math.Exp(float64(x - max)))
			sum += probs[i]
		}
		for i := range probs {
			probs[i] = afmoeBF16(probs[i] / sum)
		}
		for d := 0; d < s.HeadDim; d++ {
			var x float32
			for t := start; t <= pos; t++ {
				x += probs[t-start] * values[t*stride+kvh*s.HeadDim+d]
			}
			out[h*s.HeadDim+d] = afmoeBF16(x)
		}
	}
	return out, nil
}

func afmoeAttentionStart(pos, window int) int {
	if window > 0 && pos >= window {
		return pos - window + 1
	}
	return 0
}

// afmoeGateAttention applies sigmoid(gate projection) before the output
// projection, retaining the reference's BF16 sigmoid and multiplication bounds.
func afmoeGateAttention(out, gate []float32) error {
	if len(out) != len(gate) {
		return fmt.Errorf("afmoe attention gate: output and gate lengths differ")
	}
	for i, x := range gate {
		sigmoid := afmoeBF16(float32(1 / (1 + math.Exp(-float64(x)))))
		out[i] = afmoeBF16(out[i] * sigmoid)
	}
	return nil
}
