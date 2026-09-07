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

// This exercises model primitives through the actual Go PJRT transport. It is
// a composed operation test, not a complete model or accelerator benchmark.
func TestNativeModelOps(t *testing.T) {
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
		t.Fatalf("devices: %v", err)
	}
	f := func(shape ...int64) builder.Type { return builder.Type{Shape: shape, ElementType: "f32"} }
	b := builder.NewFunc("main", f(5, 6), builder.Type{Shape: []int64{2, 3}, ElementType: "i64"}, f(2, 3, 2), f(2, 3, 2), f(2, 3, 6))
	x, err := b.GatherRows(b.Arg(0), b.Arg(1))
	if err != nil {
		t.Fatal(err)
	}
	r, err := b.RotaryEmbedding(x, b.Arg(2), b.Arg(3))
	if err != nil {
		t.Fatal(err)
	}
	out, err := b.SwiGLU(r, b.Arg(4))
	if err != nil {
		t.Fatal(err)
	}
	b.Return(x, r, out)
	var text bytes.Buffer
	if err := (&stablehlo.Module{Funcs: []stablehlo.Emitter{b}}).Emit(&text); err != nil {
		t.Fatal(err)
	}
	executable, err := engine.Compile(text.Bytes(), "mlir", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := executable.Close(); err != nil {
			t.Error(err)
		}
	})
	weights := make([]float32, 30)
	for i := range weights {
		weights[i] = float32(i-13) / 4
	}
	ids := []int64{4, 0, 2, 1, 3, 1}
	cosine, sine := make([]float32, 12), make([]float32, 12)
	for i := range cosine {
		angle := float64(i) / 7
		cosine[i] = float32(math.Cos(angle))
		sine[i] = float32(math.Sin(angle))
	}
	up := make([]float32, 36)
	for i := range up {
		up[i] = float32(i%7-3) / 2
	}
	data := make([][]byte, 5)
	for i, values := range [][]float32{weights, nil, cosine, sine, up} {
		data[i] = make([]byte, 4*len(values))
		for j, v := range values {
			binary.LittleEndian.PutUint32(data[i][4*j:], math.Float32bits(v))
		}
	}
	data[1] = make([]byte, 8*len(ids))
	for i, v := range ids {
		binary.LittleEndian.PutUint64(data[1][8*i:], uint64(v))
	}
	args := make([]*xla.Buffer, 5)
	for i := range args {
		typ := xla.F32
		if i == 1 {
			typ = openxla.Buffer_Type_S64
		}
		buffer, err := engine.NewBufferFromHost(data[i], typ, b.Arg(i).Type.Shape, devices[0])
		if err != nil {
			t.Fatal(err)
		}
		args[i] = buffer
		t.Cleanup(func() {
			if err := buffer.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	outputs, err := executable.Execute(context.Background(), args)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, out := range outputs {
			if err := out.Close(); err != nil {
				t.Error(err)
			}
		}
	})
	if len(outputs) != 3 {
		t.Fatalf("output count %d", len(outputs))
	}
	wants := make([][]float64, 3)
	for i := range wants {
		wants[i] = make([]float64, 36)
	}
	for row, id := range ids {
		for col := 0; col < 6; col++ {
			v := float64(weights[int(id)*6+col])
			wants[0][row*6+col] = v
			wants[1][row*6+col] = v
		}
		for col := 0; col < 2; col++ {
			a, c := wants[0][row*6+col], float64(cosine[row*2+col])
			v, s := wants[0][row*6+col+2], float64(sine[row*2+col])
			wants[1][row*6+col] = a*c - v*s
			wants[1][row*6+col+2] = a*s + v*c
		}
	}
	for i, v := range wants[1] {
		wants[2][i] = v / (1 + math.Exp(-v)) * float64(up[i])
	}
	for i, out := range outputs {
		dst := make([]byte, 4*36)
		if err := out.ToHost(dst); err != nil {
			t.Fatal(err)
		}
		for j, want := range wants[i] {
			got := float64(math.Float32frombits(binary.LittleEndian.Uint32(dst[4*j:])))
			if math.IsNaN(got) || math.Abs(got-want) > 2e-6*(1+math.Abs(want)) {
				t.Fatalf("output %d element %d = %g, want %g", i, j, got, want)
			}
		}
	}
}
