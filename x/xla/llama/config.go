package llama

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

// Config describes an unscaled, split-half rotary LLaMA decoder.
// Dimensions must be positive. HeadSize must be even;
// QueryHeads must be divisible by KVHeads. HeadSize need not equal
// HiddenSize/QueryHeads. NormEpsilon and RopeTheta must be positive and finite.
type Config struct {
	// ElementType selects activations, projections and KV history. Zero means
	// F32; F32, F16 and BF16 are supported. Normalization, rotary, activation
	// and attention intermediates use F32. Logits are returned as F32.
	ElementType                    xla.ElementType
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
	name  string
	typ   builder.Type
	value weightValue
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

func describeValues(c Config, values []weightValue, layerCount int) (layout, error) {
	var out layout
	if _, _, err := elementInfo(c.ElementType, true); err != nil {
		return out, err
	}
	if layerCount < 0 || layerCount > (len(values)-3)/12 || len(values) != 3+12*layerCount {
		return out, fmt.Errorf("llama: invalid parameter count")
	}
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
	if layerCount == 0 {
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
	for _, shape := range [][]int64{{c.BatchSize, c.Capacity, c.VocabSize}, {c.BatchSize, c.QueryHeads, c.Capacity, c.Capacity}, {c.BatchSize, c.Capacity, c.HiddenSize}, {c.BatchSize, c.Capacity, c.IntermediateSize}, {c.BatchSize, c.Capacity, qwidth}, {int64(layerCount), 2, c.BatchSize, c.Capacity, kvwidth}} {
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
	next := 0
	add := func(name string, optional bool, dims ...int64) int {
		value := values[next]
		next++
		if optional && value.absent() {
			return -1
		}
		element, width, err := elementInfo(value.kind, false)
		if err == nil {
			err = value.validate(name, width, dims)
		}
		if firstErr == nil {
			firstErr = err
		}
		i := len(out.parameters)
		out.parameters = append(out.parameters, parameter{name, builder.Type{Shape: dims, ElementType: element}, value})
		return i
	}
	out.embedding = add("embedding", false, c.VocabSize, c.HiddenSize)
	for i := 0; i < layerCount; i++ {
		prefix := fmt.Sprintf("layer %d ", i)
		p := layer{
			norm: add(prefix+"attention norm", false, c.HiddenSize), post: add(prefix+"MLP norm", false, c.HiddenSize),
			q: add(prefix+"query", false, qwidth, c.HiddenSize), k: add(prefix+"key", false, kvwidth, c.HiddenSize), v: add(prefix+"value", false, kvwidth, c.HiddenSize),
			o: add(prefix+"attention output", false, c.HiddenSize, qwidth), gate: add(prefix+"gate", false, c.IntermediateSize, c.HiddenSize), up: add(prefix+"up", false, c.IntermediateSize, c.HiddenSize), down: add(prefix+"down", false, c.HiddenSize, c.IntermediateSize),
			qBias: add(prefix+"query bias", true, qwidth), kBias: add(prefix+"key bias", true, kvwidth), vBias: add(prefix+"value bias", true, kvwidth),
		}
		out.layers = append(out.layers, p)
	}
	out.norm = add("final norm", false, c.HiddenSize)
	out.head = add("LM head", true, c.VocabSize, c.HiddenSize)
	if out.head < 0 {
		out.head = out.embedding
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

// weightValue is an internal input representation. Constructors set exactly one
// payload; raw input is never widened into a full host float32 copy.
type weightValue struct {
	kind xla.ElementType
	data []float32
	raw  []byte
}

func (v weightValue) absent() bool { return v.data == nil && v.raw == nil }
func (v weightValue) validate(name string, width int64, dims []int64) error {
	n, err := byteSize(width, dims...)
	if err != nil {
		return err
	}
	if v.data != nil {
		if len(v.data) != n/4 {
			return fmt.Errorf("llama: %s has %d elements, want %d", name, len(v.data), n/4)
		}
		for _, x := range v.data {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return fmt.Errorf("llama: %s contains nonfinite weights", name)
			}
		}
		return nil
	}
	if len(v.raw) != n {
		return fmt.Errorf("llama: %s has %d bytes, want %d", name, len(v.raw), n)
	}
	for i := 0; i < n; i += int(width) {
		finite := false
		switch v.kind {
		case xla.F32:
			finite = binary.LittleEndian.Uint32(v.raw[i:])&0x7f800000 != 0x7f800000
		case xla.F16:
			finite = binary.LittleEndian.Uint16(v.raw[i:])&0x7c00 != 0x7c00
		case xla.BF16:
			finite = binary.LittleEndian.Uint16(v.raw[i:])&0x7f80 != 0x7f80
		}
		if !finite {
			return fmt.Errorf("llama: %s contains nonfinite weights", name)
		}
	}
	return nil
}
func elementInfo(kind xla.ElementType, defaultF32 bool) (string, int64, error) {
	if kind == 0 && defaultF32 {
		kind = xla.F32
	}
	switch kind {
	case xla.F32:
		return "f32", 4, nil
	case xla.F16:
		return "f16", 2, nil
	case xla.BF16:
		return "bf16", 2, nil
	}
	return "", 0, fmt.Errorf("llama: element type must be F32, F16 or BF16")
}
func describe(c Config, w Weights) (layout, error) {
	value := func(data []float32) weightValue { return weightValue{kind: xla.F32, data: data} }
	values := []weightValue{value(w.Embedding)}
	for _, l := range w.Layers {
		for _, v := range [][]float32{l.AttentionNorm, l.MLPNorm, l.Query, l.Key, l.Value, l.AttentionOutput, l.Gate, l.Up, l.Down, l.QueryBias, l.KeyBias, l.ValueBias} {
			values = append(values, value(v))
		}
	}
	values = append(values, value(w.Norm), value(w.LMHead))
	return describeValues(c, values, len(w.Layers))
}
