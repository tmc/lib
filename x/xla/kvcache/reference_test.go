package kvcache

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestNativeCacheReference(t *testing.T) {
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
	devices, err := engine.Devices()
	if err != nil || len(devices) == 0 {
		t.Fatalf("devices: %v %v", devices, err)
	}
	queries := [][]float32{{.5, -1}, {1, .5}, {-1, 1}}
	keys := [][]float32{{1, 0}, {0, 1}, {1, 1}}
	values := [][]float32{{1, -1, 2}, {3, 2, 0}, {-1, 1, 4}}
	half := map[float32]uint16{-1: 0xbc00, 0: 0, .5: 0x3800, 1: 0x3c00, 2: 0x4000, 3: 0x4200, 4: 0x4400}
	for _, tt := range []struct {
		name      string
		typ       xla.ElementType
		tolerance float64
	}{{"f32", xla.F32, 1e-6}, {"f16", xla.F16, 5e-4}, {"bf16", xla.BF16, 4e-3}} {
		t.Run(tt.name, func(t *testing.T) {
			c, err := New(engine, Config{Capacity: 4, KeySize: 2, ValueSize: 3, ElementType: tt.typ})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := c.Close(); err != nil {
					t.Error(err)
				}
			})
			upload := func(input []float32) *xla.Buffer {
				t.Helper()
				width := 2
				if tt.typ == xla.F32 {
					width = 4
				}
				data := make([]byte, width*len(input))
				for i, v := range input {
					switch tt.typ {
					case xla.F32:
						binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
					case xla.BF16:
						binary.LittleEndian.PutUint16(data[2*i:], uint16(math.Float32bits(v)>>16))
					case xla.F16:
						bits, ok := half[v]
						if !ok {
							t.Fatalf("missing fixture bits for %g", v)
						}
						binary.LittleEndian.PutUint16(data[2*i:], bits)
					}
				}
				b, err := engine.NewBufferFromHost(data, tt.typ, []int64{1, int64(len(input))}, c.device)
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
			for step, q := range queries {
				out, err := c.Append(context.Background(), upload(q), upload(keys[step]), upload(values[step]))
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := out.Close(); err != nil {
						t.Error(err)
					}
				})
				width := 2
				if tt.typ == xla.F32 {
					width = 4
				}
				data := make([]byte, 3*width)
				if err := out.ToHost(data); err != nil {
					t.Fatal(err)
				}
				// Full-prefix reference with nonzero, distinct keys: dropping K updates
				// changes attention weights, independently of V-history correctness.
				scores := make([]float64, step+1)
				maximum := math.Inf(-1)
				for j := range scores {
					scores[j] = (float64(q[0])*float64(keys[j][0]) + float64(q[1])*float64(keys[j][1])) / math.Sqrt(2)
					maximum = math.Max(maximum, scores[j])
				}
				total := 0.0
				for j := range scores {
					scores[j] = math.Exp(scores[j] - maximum)
					total += scores[j]
				}
				for col := 0; col < 3; col++ {
					want := 0.0
					for j, weight := range scores {
						want += weight / total * float64(values[j][col])
					}
					var got float64
					switch tt.typ {
					case xla.F32:
						got = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[4*col:])))
					case xla.BF16:
						got = float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(data[2*col:])) << 16))
					case xla.F16:
						word := binary.LittleEndian.Uint16(data[2*col:])
						sign := 1.0
						if word&0x8000 != 0 {
							sign = -1
						}
						exp := int(word >> 10 & 31)
						fraction := float64(word & 1023)
						if exp == 31 {
							t.Fatalf("nonfinite half result %04x", word)
						}
						if exp == 0 {
							got = sign * math.Ldexp(fraction, -24)
						} else {
							got = sign * math.Ldexp(1+fraction/1024, exp-15)
						}
					}
					if math.IsNaN(got) || math.Abs(got-want) > 1e-5+tt.tolerance*math.Abs(want) {
						t.Errorf("step %d output %d = %g, want %g", step, col, got, want)
					}
				}
			}
		})
	}
}
