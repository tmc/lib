package llama

import (
	"fmt"
	"math"

	"github.com/tmc/lib/x/stablehlo/builder"
)

// Config describes an unscaled, split-half rotary LLaMA decoder using f32
// weights and arithmetic. Dimensions must be positive. HeadSize must be even;
// QueryHeads must be divisible by KVHeads. HeadSize need not equal
// HiddenSize/QueryHeads. NormEpsilon and RopeTheta must be positive and finite.
type Config struct {
	BatchSize, Capacity, VocabSize int64
	HiddenSize, IntermediateSize   int64
	QueryHeads, KVHeads, HeadSize  int64
	NormEpsilon                    float32
	RopeTheta                      float64
}

// LayerWeights contains row-major [output, input] projection matrices and
// one-dimensional normalization scales. Nil query/key/value biases mean zero.
type LayerWeights struct {
	AttentionNorm, MLPNorm             []float32
	Query, Key, Value, AttentionOutput []float32
	Gate, Up, Down                     []float32
	QueryBias, KeyBias, ValueBias      []float32
}

// Weights contains the embedding [VocabSize, HiddenSize], decoder layers,
// final normalization scale [HiddenSize], and LMHead [VocabSize, HiddenSize].
// A nil LMHead shares the embedding weights. At least one layer is required.
// New copies the data to the device and does not retain these slices.
type Weights struct {
	Embedding, Norm, LMHead []float32
	Layers                  []LayerWeights
}

type parameter struct {
	name string
	typ  builder.Type
	data []float32
}
type layer struct{ norm, post, q, k, v, o, gate, up, down, qBias, kBias, vBias int }
type layout struct {
	parameters            []parameter
	layers                []layer
	embedding, norm, head int
	frequencies           []float64
}

func byteSize(width int64, dims ...int64) (int, error) {
	n := width
	for _, d := range dims {
		if d <= 0 || d > int64(int(^uint(0)>>1))/n {
			return 0, fmt.Errorf("llama: invalid or overflowing tensor dimensions")
		}
		n *= d
	}
	return int(n), nil
}

func describe(c Config, w Weights) (layout, error) {
	var out layout
	for _, d := range []int64{c.BatchSize, c.Capacity, c.VocabSize, c.HiddenSize, c.IntermediateSize, c.QueryHeads, c.KVHeads, c.HeadSize} {
		if d <= 0 {
			return out, fmt.Errorf("llama: dimensions must be positive")
		}
	}
	if c.HeadSize%2 != 0 || c.QueryHeads%c.KVHeads != 0 {
		return out, fmt.Errorf("llama: even head size and divisible query/KV head counts required")
	}
	if c.NormEpsilon <= 0 || math.IsNaN(float64(c.NormEpsilon)) || math.IsInf(float64(c.NormEpsilon), 0) || c.RopeTheta <= 0 || math.IsNaN(c.RopeTheta) || math.IsInf(c.RopeTheta, 0) {
		return out, fmt.Errorf("llama: epsilon and rope theta must be positive and finite")
	}
	if len(w.Layers) == 0 {
		return out, fmt.Errorf("llama: at least one layer required")
	}
	qbytes, err := byteSize(4, c.QueryHeads, c.HeadSize)
	if err != nil {
		return out, err
	}
	kbytes, err := byteSize(4, c.KVHeads, c.HeadSize)
	if err != nil {
		return out, err
	}
	qwidth, kvwidth := int64(qbytes/4), int64(kbytes/4)
	for _, shape := range [][]int64{{c.BatchSize, c.Capacity, c.VocabSize}, {c.BatchSize, c.QueryHeads, c.Capacity, c.Capacity}, {c.BatchSize, c.Capacity, c.HiddenSize}, {c.BatchSize, c.Capacity, c.IntermediateSize}, {c.BatchSize, c.Capacity, qwidth}, {int64(len(w.Layers)), 2, c.BatchSize, c.Capacity, kvwidth}} {
		if _, err := byteSize(4, shape...); err != nil {
			return out, err
		}
	}
	if _, err := byteSize(8, c.BatchSize, c.Capacity, c.Capacity); err != nil {
		return out, err
	}
	if _, err := byteSize(8, c.BatchSize, c.KVHeads, c.Capacity); err != nil {
		return out, err
	}
	// Validate every shape and value before any device allocation.
	var firstErr error
	add := func(name string, data []float32, optional bool, dims ...int64) int {
		if optional && data == nil {
			return -1
		}
		n, err := byteSize(4, dims...)
		if err == nil && len(data) != n/4 {
			err = fmt.Errorf("llama: %s has %d elements, want %d", name, len(data), n/4)
		}
		if err == nil {
			for _, v := range data {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					err = fmt.Errorf("llama: %s contains nonfinite weights", name)
					break
				}
			}
		}
		if firstErr == nil {
			firstErr = err
		}
		i := len(out.parameters)
		out.parameters = append(out.parameters, parameter{name, builder.Type{Shape: dims, ElementType: "f32"}, data})
		return i
	}
	out.embedding = add("embedding", w.Embedding, false, c.VocabSize, c.HiddenSize)
	for i, v := range w.Layers {
		prefix := fmt.Sprintf("layer %d ", i)
		p := layer{
			norm: add(prefix+"attention norm", v.AttentionNorm, false, c.HiddenSize), post: add(prefix+"MLP norm", v.MLPNorm, false, c.HiddenSize),
			q: add(prefix+"query", v.Query, false, qwidth, c.HiddenSize), k: add(prefix+"key", v.Key, false, kvwidth, c.HiddenSize), v: add(prefix+"value", v.Value, false, kvwidth, c.HiddenSize),
			o: add(prefix+"attention output", v.AttentionOutput, false, c.HiddenSize, qwidth), gate: add(prefix+"gate", v.Gate, false, c.IntermediateSize, c.HiddenSize), up: add(prefix+"up", v.Up, false, c.IntermediateSize, c.HiddenSize), down: add(prefix+"down", v.Down, false, c.HiddenSize, c.IntermediateSize),
			qBias: add(prefix+"query bias", v.QueryBias, true, qwidth), kBias: add(prefix+"key bias", v.KeyBias, true, kvwidth), vBias: add(prefix+"value bias", v.ValueBias, true, kvwidth),
		}
		out.layers = append(out.layers, p)
	}
	out.norm = add("final norm", w.Norm, false, c.HiddenSize)
	out.head = out.embedding
	if w.LMHead != nil {
		out.head = add("LM head", w.LMHead, false, c.VocabSize, c.HiddenSize)
	}
	if firstErr != nil {
		return layout{}, firstErr
	}
	out.frequencies = make([]float64, c.HeadSize/2)
	for i := range out.frequencies {
		f := math.Pow(c.RopeTheta, -float64(2*i)/float64(c.HeadSize))
		if math.IsInf(f, 0) || math.IsInf(float64(c.Capacity-1)*f, 0) || math.IsNaN(f) {
			return layout{}, fmt.Errorf("llama: rotary angles overflow")
		}
		out.frequencies[i] = f
	}
	return out, nil
}
