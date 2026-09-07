package xla_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/x/stablehlo"
	"github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

func TestNativeAttention(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	e, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	}()
	devices, err := e.Devices()
	if err != nil {
		t.Fatal(err)
	}
	if len(devices) == 0 {
		t.Fatal("no addressable devices")
	}
	for _, tt := range []struct {
		name                            string
		typ                             xla.ElementType
		one, two, half, negative, large uint16
		rtol                            float64
	}{
		{"f16", xla.F16, 0x3c00, 0x4000, 0x3800, 0xbc00, 0x5c00, 0.0005},
		{"bf16", xla.BF16, 0x3f80, 0x4000, 0x3f00, 0xbf80, 0x4380, 0.004},
	} {
		t.Run(tt.name, func(t *testing.T) {
			typ := builder.Type{Shape: []int64{2, 2}, ElementType: tt.name}
			b := builder.NewFunc("main", typ, typ, typ)
			out, err := b.ScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2))
			if err != nil {
				t.Fatal(err)
			}
			b.Return(out)
			var program bytes.Buffer
			if err := (&stablehlo.Module{Funcs: []stablehlo.Emitter{b}}).Emit(&program); err != nil {
				t.Fatal(err)
			}
			executable, err := e.Compile(program.Bytes(), "mlir", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := executable.Close(); err != nil {
					t.Error(err)
				}
			}()
			for _, large := range []bool{false, true} {
				t.Run(map[bool]string{false: "moderate", true: "f16_dot_overflow"}[large], func(t *testing.T) {
					one := tt.one
					p := 1 / (1 + math.Exp(-1/math.Sqrt(2)))
					if large {
						one, p = tt.large, 1
					}
					var args []*xla.Buffer
					for _, words := range [][]uint16{{one, 0, 0, one}, {one, 0, 0, one}, {tt.two, tt.negative, 0, tt.half}} {
						data := make([]byte, 2*len(words))
						for i, w := range words {
							binary.LittleEndian.PutUint16(data[2*i:], w)
						}
						buf, err := e.NewBufferFromHost(data, tt.typ, []int64{2, 2}, devices[0])
						if err != nil {
							t.Fatal(err)
						}
						defer func() {
							if err := buf.Close(); err != nil {
								t.Error(err)
							}
						}()
						args = append(args, buf)
					}
					outputs, err := executable.Execute(context.Background(), args)
					if err != nil {
						t.Fatal(err)
					}
					defer func() {
						for _, out := range outputs {
							if err := out.Close(); err != nil {
								t.Error(err)
							}
						}
					}()
					if len(outputs) != 1 {
						t.Fatalf("got %d outputs", len(outputs))
					}
					var data bytes.Buffer
					if _, err := outputs[0].WriteTo(&data); err != nil {
						t.Fatal(err)
					}
					if data.Len() != 8 {
						t.Fatalf("got %d bytes, want 8", data.Len())
					}
					for i, want := range []float64{2 * p, .5 - 1.5*p, 2 * (1 - p), -1 + 1.5*p} {
						word := binary.LittleEndian.Uint16(data.Bytes()[2*i:])
						var got float64
						if tt.name == "bf16" {
							got = float64(math.Float32frombits(uint32(word) << 16))
						} else {
							sign := 1.0
							if word&0x8000 != 0 {
								sign = -1
							}
							exp, fraction := int(word>>10&31), float64(word&1023)
							if exp == 31 {
								t.Fatalf("nonfinite f16 output: %04x", word)
							}
							if exp == 0 {
								got = sign * math.Ldexp(fraction, -24)
							} else {
								got = sign * math.Ldexp(1+fraction/1024, exp-15)
							}
						}
						if math.IsNaN(got) || math.Abs(got-want) > 1e-5+tt.rtol*math.Abs(want) {
							t.Errorf("output %d = %g, want %g", i, got, want)
						}
					}
				})
			}
		})
	}
}
