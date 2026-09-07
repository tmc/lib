package llama

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/x/xla"
)

// The fixture uses multiples of 1/16, exactly representable in both half types.
// This avoids sharing a production conversion implementation with the oracle.
func encodedFixture(c Config, tied bool, kind xla.ElementType) (Weights, EncodedWeights) {
	w := fixture(c, tied)
	encode := func(data []float32) EncodedTensor {
		if data == nil {
			return EncodedTensor{}
		}
		width := 2
		if kind == xla.F32 {
			width = 4
		}
		raw := make([]byte, len(data)*width)
		for i, v := range data {
			v = float32(math.Round(float64(v)*16) / 16)
			data[i] = v
			bits := math.Float32bits(v)
			if kind == xla.F32 {
				binary.LittleEndian.PutUint32(raw[4*i:], bits)
				continue
			}
			word := uint16(bits >> 16)
			if kind == xla.F16 {
				word = uint16(bits>>16) & 0x8000
				if v != 0 {
					word |= uint16(((bits>>23)&255)-112)<<10 | uint16(bits>>13)&0x3ff
				}
			}
			binary.LittleEndian.PutUint16(raw[2*i:], word)
		}
		return EncodedTensor{ElementType: kind, Data: raw}
	}
	e := EncodedWeights{Embedding: encode(w.Embedding), Norm: encode(w.Norm), LMHead: encode(w.LMHead)}
	for _, l := range w.Layers {
		e.Layers = append(e.Layers, EncodedLayerWeights{
			AttentionNorm: encode(l.AttentionNorm), MLPNorm: encode(l.MLPNorm),
			Query: encode(l.Query), Key: encode(l.Key), Value: encode(l.Value), AttentionOutput: encode(l.AttentionOutput),
			Gate: encode(l.Gate), Up: encode(l.Up), Down: encode(l.Down),
			QueryBias: encode(l.QueryBias), KeyBias: encode(l.KeyBias), ValueBias: encode(l.ValueBias),
		})
	}
	return w, e
}

func TestEncodedValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*Config, *EncodedWeights)
		want   string
	}{
		{"compute type", func(c *Config, w *EncodedWeights) { c.ElementType = xla.S32 }, "element type"},
		{"missing source type", func(c *Config, w *EncodedWeights) { w.Embedding.ElementType = 0 }, "element type"},
		{"source type", func(c *Config, w *EncodedWeights) { w.Embedding.ElementType = xla.S32 }, "element type"},
		{"short", func(c *Config, w *EncodedWeights) { w.Embedding.Data = w.Embedding.Data[1:] }, "bytes"},
		{"long", func(c *Config, w *EncodedWeights) { w.Embedding.Data = append(w.Embedding.Data, 0) }, "bytes"},
		{"nonfinite", func(c *Config, w *EncodedWeights) { binary.LittleEndian.PutUint16(w.Embedding.Data, 0x7c00) }, "nonfinite"},
		{"empty head", func(c *Config, w *EncodedWeights) { w.LMHead = EncodedTensor{ElementType: xla.F16, Data: []byte{}} }, "bytes"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := testConfig()
			_, w := encodedFixture(c, true, xla.F16)
			tt.change(&c, &w)
			if _, err := NewEncoded(context.Background(), nil, c, w); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %s before engine access", err, tt.want)
			}
		})
	}
	// The same word is F16 infinity but a finite BF16 number.
	for _, kind := range []xla.ElementType{xla.F16, xla.BF16} {
		c := testConfig()
		_, w := encodedFixture(c, true, kind)
		binary.LittleEndian.PutUint16(w.Embedding.Data, 0x7c00)
		_, err := describeEncoded(c, w)
		if (err == nil) != (kind == xla.BF16) {
			t.Fatalf("type %v: %v", kind, err)
		}
	}
}

func TestNativeEncodedModel(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()
	for _, tt := range []struct {
		name             string
		compute, storage xla.ElementType
		tol              float64
	}{
		{"f16", xla.F16, xla.F16, 0.01}, {"bf16", xla.BF16, xla.BF16, 0.08},
		{"f32 from bf16", xla.F32, xla.BF16, 2e-5},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := testConfig()
			c.ElementType = tt.compute
			w, encoded := encodedFixture(c, true, tt.storage)
			// Mixed source types must not be inferred from execution precision.
			encoded.Norm = EncodedTensor{ElementType: xla.F32, Data: encodeFloats(w.Norm)}
			m, err := NewEncoded(context.Background(), engine, c, encoded)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := m.Close(); err != nil {
					t.Error(err)
				}
			}()
			if m.head != m.embedding {
				t.Fatal("tied head duplicated")
			}
			for i, p := range m.parameters {
				want := tt.storage
				if i == m.norm {
					want = xla.F32
				}
				typ, _, err := p.Shape()
				if err != nil || typ != want {
					t.Fatalf("parameter %d type %v, %v; want %v", i, typ, err, want)
				}
			}
			for _, h := range m.history {
				typ, _, err := h.Shape()
				if err != nil || typ != tt.compute {
					t.Fatalf("history type %v %v", typ, err)
				}
			}
			clear(encoded.Embedding.Data) // Constructors must not retain mutable input.
			ref := newReference(c, w)
			maxError := 0.0
			for _, step := range []struct {
				ids    []int32
				counts []int64
				reset  bool
			}{
				{[]int32{1, 2, 3, 4, 0, 0}, []int64{3, 1}, false},
				{[]int32{2, 5}, nil, false},
				{[]int32{6, 1}, nil, true},
			} {
				if step.reset {
					if err := m.Reset(); err != nil {
						t.Fatal(err)
					}
					ref = newReference(c, w)
				}
				before := m.Lengths()
				output, err := m.Forward(context.Background(), step.ids, step.counts)
				if err != nil {
					t.Fatal(err)
				}
				typ, shape, err := output.Shape()
				if err != nil || typ != xla.F32 {
					t.Fatalf("logits type %v %v", typ, err)
				}
				chunk := int64(len(step.ids)) / c.BatchSize
				if !slices.Equal(shape, []int64{c.BatchSize, chunk, c.VocabSize}) {
					t.Fatalf("logit shape %v", shape)
				}
				raw := make([]byte, len(step.ids)*int(c.VocabSize)*4)
				readErr := output.ToHost(raw)
				closeErr := output.Close()
				if readErr != nil {
					t.Fatal(readErr)
				}
				if closeErr != nil {
					t.Fatal(closeErr)
				}
				for batch := int64(0); batch < c.BatchSize; batch++ {
					count := chunk
					if step.counts != nil {
						count = step.counts[batch]
					}
					before[batch] += count
					for row := int64(0); row < count; row++ {
						index := batch*chunk + row
						want := ref.token(int(batch), step.ids[index])
						for i, v := range want {
							got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[(index*c.VocabSize+int64(i))*4:])))
							delta := math.Abs(got - v)
							maxError = math.Max(maxError, delta)
							if math.IsNaN(got) || math.IsInf(got, 0) || delta > tt.tol*(1+math.Abs(v)) {
								t.Fatalf("batch %d row %d logit %d: %g, want %g", batch, row, i, got, v)
							}
						}
					}
				}
				if !slices.Equal(m.Lengths(), before) {
					t.Fatalf("lengths %v, want %v", m.Lengths(), before)
				}
			}
			t.Logf("maximum absolute error against float64 reference: %g", maxError)
		})
	}
}

func TestEncodedNonfinite(t *testing.T) {
	for _, tt := range []struct {
		kind  xla.ElementType
		words []uint32
	}{
		{xla.F32, []uint32{0x7f800000, 0xff800000, 0x7fc00000}},
		{xla.F16, []uint32{0x7c00, 0xfc00, 0x7e00}},
		{xla.BF16, []uint32{0x7f80, 0xff80, 0x7fc0}},
	} {
		for _, word := range tt.words {
			c := testConfig()
			_, w := encodedFixture(c, true, tt.kind)
			if tt.kind == xla.F32 {
				binary.LittleEndian.PutUint32(w.Norm.Data, word)
			} else {
				binary.LittleEndian.PutUint16(w.Norm.Data, uint16(word))
			}
			if _, err := NewEncoded(context.Background(), nil, c, w); err == nil || !strings.Contains(err.Error(), "nonfinite") {
				t.Fatalf("type %v word %x: %v", tt.kind, word, err)
			}
		}
	}
}
