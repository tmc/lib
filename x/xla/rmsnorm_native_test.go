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

func TestNativeRMSNormHalf(t *testing.T) {
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
	devices, err := engine.Devices()
	if err != nil {
		t.Fatal(err)
	}
	if len(devices) == 0 {
		t.Fatal("no addressable devices")
	}
	for _, tt := range []struct {
		name       string
		typ        xla.ElementType
		large, one uint16
		tol        float64
	}{
		{"f16", xla.F16, 0x63d0, 0x3c00, 0.0008},
		{"bf16", xla.BF16, 0x447a, 0x3f80, 0.004},
	} {
		t.Run(tt.name, func(t *testing.T) {
			typ := builder.Type{Shape: []int64{2, 3}, ElementType: tt.name}
			gamma := builder.Type{Shape: []int64{3}, ElementType: tt.name}
			b := builder.NewFunc("main", typ, gamma)
			out, err := b.RMSNorm(b.Arg(0), b.Arg(1), 1, 1e-8)
			if err != nil {
				t.Fatal(err)
			}
			out, err = b.Convert(out, "f32")
			if err != nil {
				t.Fatal(err)
			}
			b.Return(out)
			var module bytes.Buffer
			if err := (&stablehlo.Module{Funcs: []stablehlo.Emitter{b}}).Emit(&module); err != nil {
				t.Fatal(err)
			}
			program, err := engine.Compile(module.Bytes(), "mlir", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := program.Close(); err != nil {
					t.Error(err)
				}
			}()
			var args []*xla.Buffer
			for i, words := range [][]uint16{{tt.large, 0, tt.large | 0x8000, 0, 0, 0}, {tt.one, tt.one, tt.one}} {
				raw := make([]byte, len(words)*2)
				for j, w := range words {
					binary.LittleEndian.PutUint16(raw[j*2:], w)
				}
				shape := typ.Shape
				if i == 1 {
					shape = gamma.Shape
				}
				buf, err := engine.NewBufferFromHost(raw, tt.typ, shape, devices[0])
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
			outputs, err := program.Execute(context.Background(), args)
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
				t.Fatalf("outputs %d, want 1", len(outputs))
			}
			var raw bytes.Buffer
			if _, err := outputs[0].WriteTo(&raw); err != nil {
				t.Fatal(err)
			}
			if raw.Len() != 24 {
				t.Fatalf("bytes %d, want 24", raw.Len())
			}
			scale := 1000 / math.Sqrt(2e6/3+1e-8)
			for i, want := range []float64{scale, 0, -scale, 0, 0, 0} {
				got := float64(math.Float32frombits(binary.LittleEndian.Uint32(raw.Bytes()[4*i:])))
				if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > tt.tol {
					t.Errorf("element %d: %g, want %g", i, got, want)
				}
			}
		})
	}
}
