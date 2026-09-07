package kvcache

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestNativeGroupedCache(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, tt := range []struct {
		name      string
		typ       xla.ElementType
		kvheads   int64
		tolerance float64
	}{
		{"gqa_f32", xla.F32, 2, 1e-6}, {"gqa_f16", xla.F16, 2, 5e-4}, {"gqa_bf16", xla.BF16, 2, 4e-3}, {"mha", xla.F32, 4, 1e-6}, {"mqa", xla.F32, 1, 1e-6},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := Config{BatchShape: []int64{2}, Capacity: 4, KeySize: 2, ValueSize: 3, ElementType: tt.typ, QueryHeads: 4, KVHeads: tt.kvheads}
			c, err := New(engine, cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := c.Close(); err != nil {
					t.Error(err)
				}
			})
			_, shape, err := c.keys.Shape()
			if err != nil || !slices.Equal(shape, []int64{2, tt.kvheads, 4, 2}) {
				t.Fatalf("K history shape = %v, %v", shape, err)
			}
			_, shape, err = c.values.Shape()
			if err != nil || !slices.Equal(shape, []int64{2, tt.kvheads, 4, 3}) {
				t.Fatalf("V history shape = %v, %v", shape, err)
			}
			query := func(batch, head, pos, col int64) float32 { return float32((batch+head+pos+col)%4-1) / 2 }
			key := func(batch, head, pos, col int64) float32 { return float32(1+batch+2*head+pos+col) / 2 }
			value := func(batch, head, pos, col int64) float32 { return float32(10*batch+3*head+2*pos+col-4) / 2 }
			position := int64(0)
			for _, chunk := range []int64{2, 1} {
				upload := func(heads, features int64, at func(int64, int64, int64, int64) float32) *xla.Buffer {
					t.Helper()
					var values []float32
					for batch := int64(0); batch < 2; batch++ {
						for head := int64(0); head < heads; head++ {
							for row := int64(0); row < chunk; row++ {
								for col := int64(0); col < features; col++ {
									values = append(values, at(batch, head, position+row, col))
								}
							}
						}
					}
					b, err := engine.NewBufferFromHost(gqaEncode(t, values, tt.typ), tt.typ, []int64{2, heads, chunk, features}, c.device)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := b.Close(); err != nil {
							t.Error(err)
						}
					})
					return b
				}
				out, err := c.Append(context.Background(), upload(4, 2, query), upload(tt.kvheads, 2, key), upload(tt.kvheads, 3, value))
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := out.Close(); err != nil {
						t.Error(err)
					}
				})
				_, shape, err := out.Shape()
				if err != nil || !slices.Equal(shape, []int64{2, 4, chunk, 3}) {
					t.Fatalf("output shape %v, %v", shape, err)
				}
				width := 2
				if tt.typ == xla.F32 {
					width = 4
				}
				data := make([]byte, int(2*4*chunk*3)*width)
				if err := out.ToHost(data); err != nil {
					t.Fatal(err)
				}
				index := 0
				for batch := int64(0); batch < 2; batch++ {
					for head := int64(0); head < 4; head++ {
						for row := int64(0); row < chunk; row++ {
							pos := position + row
							kvhead := head / (4 / tt.kvheads)
							weights := make([]float64, pos+1)
							maximum := math.Inf(-1)
							for past := int64(0); past <= pos; past++ {
								score := 0.0
								for col := int64(0); col < 2; col++ {
									score += float64(query(batch, head, pos, col)) * float64(key(batch, kvhead, past, col))
								}
								weights[past] = score / math.Sqrt(2)
								maximum = math.Max(maximum, weights[past])
							}
							total := 0.0
							for i, w := range weights {
								weights[i] = math.Exp(w - maximum)
								total += weights[i]
							}
							for col := int64(0); col < 3; col++ {
								want := 0.0
								for past, w := range weights {
									want += w / total * float64(value(batch, kvhead, int64(past), col))
								}
								got := gqaDecode(t, data[index*width:], tt.typ)
								index++
								if math.IsNaN(got) || math.Abs(got-want) > 1e-5+tt.tolerance*math.Abs(want) {
									t.Errorf("batch %d head %d position %d column %d = %g, want %g", batch, head, pos, col, got, want)
								}
							}
						}
					}
				}
				position += chunk
			}
			wrong, err := engine.NewBufferFromHost(gqaEncode(t, make([]float32, 12), tt.typ), tt.typ, []int64{2, 3, 1, 2}, c.device)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := wrong.Close(); err != nil {
					t.Error(err)
				}
			})
			if _, err := c.Append(context.Background(), wrong, wrong, wrong); err == nil {
				t.Fatal("accepted wrong query-head count")
			}
			if c.Len() != 3 || len(c.plans) != 2 {
				t.Fatal("grouped cache length or program retention incorrect")
			}
		})
	}
}

// Fixtures are exactly representable in all three types; reject accidental
// fixture rounding rather than quietly altering the reference inputs.
func gqaEncode(t *testing.T, values []float32, typ xla.ElementType) []byte {
	t.Helper()
	width := 2
	if typ == xla.F32 {
		width = 4
	}
	data := make([]byte, width*len(values))
	for i, v := range values {
		bits := math.Float32bits(v)
		switch typ {
		case xla.F32:
			binary.LittleEndian.PutUint32(data[4*i:], bits)
		case xla.BF16:
			if bits&0xffff != 0 {
				t.Fatalf("inexact bf16 fixture %g", v)
			}
			binary.LittleEndian.PutUint16(data[2*i:], uint16(bits>>16))
		case xla.F16:
			var word uint16
			if v != 0 {
				exp := int(bits>>23&255) - 127 + 15
				if exp <= 0 || exp >= 31 || bits&0x1fff != 0 {
					t.Fatalf("inexact f16 fixture %g", v)
				}
				word = uint16(bits>>16&0x8000) | uint16(exp<<10) | uint16(bits>>13&1023)
			}
			binary.LittleEndian.PutUint16(data[2*i:], word)
		}
	}
	return data
}

func gqaDecode(t *testing.T, data []byte, typ xla.ElementType) float64 {
	t.Helper()
	if typ == xla.F32 {
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(data)))
	}
	word := binary.LittleEndian.Uint16(data)
	if typ == xla.BF16 {
		return float64(math.Float32frombits(uint32(word) << 16))
	}
	sign := 1.0
	if word&0x8000 != 0 {
		sign = -1
	}
	exp := int(word >> 10 & 31)
	fraction := float64(word & 1023)
	if exp == 31 {
		t.Fatalf("nonfinite half output %04x", word)
	}
	if exp == 0 {
		return sign * math.Ldexp(fraction, -24)
	}
	return sign * math.Ldexp(1+fraction/1024, exp-15)
}
