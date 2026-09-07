package xla_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/openxla"
	"github.com/tmc/lib/x/stablehlo"
	"github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

func TestNativeMaskedAttention(t *testing.T) {
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
		name            string
		queries, offset int64
		padding, causal bool
		want            []float32
	}{
		{"causal", 2, 0, false, true, []float32{1, 2, 2, 3}},
		{"padding", 2, 0, true, false, []float32{2, 2, 0, 0}},
		{"combined", 2, 0, true, true, []float32{1, 2, 0, 0}},
		{"decode", 1, 2, false, true, []float32{103.0 / 3, 106.0 / 3}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			typ := func(shape ...int64) builder.Type { return builder.Type{Shape: shape, ElementType: "f32"} }
			args := []builder.Type{typ(2, tt.queries, 1), typ(2, 3, 1), typ(2, 3, 1)}
			if tt.padding {
				args = append(args, builder.Type{Shape: []int64{2, 1, 3}, ElementType: "i1"})
			}
			b := builder.NewFunc("main", args...)
			var mask builder.Value
			if tt.causal {
				mask, err = b.CausalMask(tt.queries, 3, tt.offset)
				if err != nil {
					t.Fatal(err)
				}
			}
			if tt.padding {
				padding := b.Arg(3)
				if tt.causal {
					full := builder.Type{Shape: []int64{2, tt.queries, 3}, ElementType: "i1"}
					mask, err = b.BroadcastInDim(mask, []int64{1, 2}, full)
					if err != nil {
						t.Fatal(err)
					}
					padding, err = b.BroadcastInDim(padding, []int64{0, 1, 2}, full)
					if err != nil {
						t.Fatal(err)
					}
					mask, err = b.And(mask, padding)
					if err != nil {
						t.Fatal(err)
					}
				} else {
					mask = padding
				}
			}
			out, err := b.MaskedScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2), mask)
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
			var inputs []*xla.Buffer
			for i, values := range [][]float32{make([]float32, 2*tt.queries), make([]float32, 6), {1, 3, 99, 2, 4, 100}} {
				data := make([]byte, len(values)*4)
				for j, v := range values {
					binary.LittleEndian.PutUint32(data[4*j:], math.Float32bits(v))
				}
				buf, err := e.NewBufferFromHost(data, xla.F32, args[i].Shape, devices[0])
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := buf.Close(); err != nil {
						t.Error(err)
					}
				}()
				inputs = append(inputs, buf)
			}
			if tt.padding {
				buf, err := e.NewBufferFromHost([]byte{1, 1, 0, 0, 0, 0}, openxla.Buffer_Type_PRED, args[3].Shape, devices[0])
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := buf.Close(); err != nil {
						t.Error(err)
					}
				}()
				inputs = append(inputs, buf)
			}
			outputs, err := executable.Execute(context.Background(), inputs)
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
			if data.Len() != 4*len(tt.want) {
				t.Fatalf("got %d output bytes", data.Len())
			}
			for i, want := range tt.want {
				got := math.Float32frombits(binary.LittleEndian.Uint32(data.Bytes()[4*i:]))
				if math.IsNaN(float64(got)) || math.Abs(float64(got-want)) > 1e-5 {
					t.Errorf("output %d = %g, want %g", i, got, want)
				}
			}
		})
	}
}
