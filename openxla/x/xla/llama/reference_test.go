package llama

import "math"

// reference evaluates one token at a time using ordinary float64 Go arithmetic.
// It neither emits graphs nor reuses the runtime's index/mask construction.
type reference struct {
	c            Config
	w            Weights
	keys, values [][][][]float64
}

func newReference(c Config, w Weights) *reference {
	r := &reference{c: c, w: w, keys: make([][][][]float64, c.BatchSize), values: make([][][][]float64, c.BatchSize)}
	for b := range r.keys {
		r.keys[b] = make([][][]float64, len(w.Layers))
		r.values[b] = make([][][]float64, len(w.Layers))
	}
	return r
}
func (r *reference) token(batch int, id int32) []float64 {
	c := r.c
	h, d, qw, kw := int(c.HiddenSize), int(c.HeadSize), int(c.QueryHeads*c.HeadSize), int(c.KVHeads*c.HeadSize)
	x := make([]float64, h)
	for i := range x {
		x[i] = float64(r.w.Embedding[int(id)*h+i])
	}
	norm := func(x []float64, w []float32) []float64 {
		sum := 0.0
		for _, v := range x {
			sum += v * v
		}
		scale := 1 / math.Sqrt(sum/float64(len(x))+float64(c.NormEpsilon))
		out := make([]float64, len(x))
		for i, v := range x {
			out[i] = v * scale * float64(w[i])
		}
		return out
	}
	linear := func(x []float64, w, bias []float32, n int) []float64 {
		out := make([]float64, n)
		for o := range out {
			for i, v := range x {
				out[o] += v * float64(w[o*len(x)+i])
			}
			if bias != nil {
				out[o] += float64(bias[o])
			}
		}
		return out
	}
	for layer, w := range r.w.Layers {
		pos := len(r.keys[batch][layer])
		normal := norm(x, w.AttentionNorm)
		q, k, v := linear(normal, w.Query, w.QueryBias, qw), linear(normal, w.Key, w.KeyBias, kw), linear(normal, w.Value, w.ValueBias, kw)
		for _, vector := range [][]float64{q, k} {
			for head := 0; head < len(vector)/d; head++ {
				for i := 0; i < d/2; i++ {
					angle := float64(pos) / math.Pow(c.RopeTheta, float64(2*i)/float64(d))
					s, co := math.Sincos(angle)
					a, b := vector[head*d+i], vector[head*d+i+d/2]
					vector[head*d+i] = a*co - b*s
					vector[head*d+i+d/2] = a*s + b*co
				}
			}
		}
		r.keys[batch][layer] = append(r.keys[batch][layer], k)
		r.values[batch][layer] = append(r.values[batch][layer], v)
		attention := make([]float64, qw)
		for head := 0; head < int(c.QueryHeads); head++ {
			kvhead := head / int(c.QueryHeads/c.KVHeads)
			scores := make([]float64, pos+1)
			maximum := math.Inf(-1)
			for past := range scores {
				for i := 0; i < d; i++ {
					scores[past] += q[head*d+i] * r.keys[batch][layer][past][kvhead*d+i] / math.Sqrt(float64(d))
				}
				maximum = math.Max(maximum, scores[past])
			}
			total := 0.0
			for i, s := range scores {
				scores[i] = math.Exp(s - maximum)
				total += scores[i]
			}
			for past, s := range scores {
				for i := 0; i < d; i++ {
					attention[head*d+i] += s / total * r.values[batch][layer][past][kvhead*d+i]
				}
			}
		}
		projected := linear(attention, w.AttentionOutput, nil, h)
		for i := range x {
			x[i] += projected[i]
		}
		normal = norm(x, w.MLPNorm)
		gate, up := linear(normal, w.Gate, nil, int(c.IntermediateSize)), linear(normal, w.Up, nil, int(c.IntermediateSize))
		for i, v := range gate {
			gate[i] = v / (1 + math.Exp(-v)) * up[i]
		}
		down := linear(gate, w.Down, nil, h)
		for i := range x {
			x[i] += down[i]
		}
	}
	head := r.w.LMHead
	if head == nil {
		head = r.w.Embedding
	}
	return linear(norm(x, r.w.Norm), head, nil, int(c.VocabSize))
}

func fixture(c Config, tied bool) Weights {
	seed := 0
	values := func(n int64, norm bool) []float32 {
		seed++
		out := make([]float32, n)
		for i := range out {
			if norm {
				out[i] = .75 + float32((i+seed)%5)/8
			} else {
				out[i] = float32((i*7+seed*3)%17-8) / 32
			}
		}
		return out
	}
	w := Weights{Embedding: values(c.VocabSize*c.HiddenSize, false)}
	q, k := c.QueryHeads*c.HeadSize, c.KVHeads*c.HeadSize
	for layer := 0; layer < 2; layer++ {
		p := LayerWeights{AttentionNorm: values(c.HiddenSize, true), MLPNorm: values(c.HiddenSize, true), Query: values(q*c.HiddenSize, false), Key: values(k*c.HiddenSize, false), Value: values(k*c.HiddenSize, false), AttentionOutput: values(c.HiddenSize*q, false), Gate: values(c.IntermediateSize*c.HiddenSize, false), Up: values(c.IntermediateSize*c.HiddenSize, false), Down: values(c.HiddenSize*c.IntermediateSize, false)}
		if layer == 1 {
			p.QueryBias = values(q, false)
			p.KeyBias = values(k, false)
			p.ValueBias = values(k, false)
		}
		w.Layers = append(w.Layers, p)
	}
	w.Norm = values(c.HiddenSize, true)
	if !tied {
		w.LMHead = values(c.VocabSize*c.HiddenSize, false)
	}
	return w
}
