package xla_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/x/xla"
)

// TestNativePJRT validates the actual Go-to-PJRT path with a caller-supplied
// CPU plugin. It does not download a plugin or select accelerator hardware.
func TestNativePJRT(t *testing.T) {
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
		t.Fatal("CPU plugin has no addressable devices")
	}
	program := `module {
  func.func @main(%a: tensor<2x4xf32>, %b: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}`
	executable, err := engine.Compile([]byte(program), "mlir", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := executable.Close(); err != nil {
			t.Error(err)
		}
	}()
	encode := func(values []float32) []byte {
		// Exercise a sub-slice that does not start at the backing allocation.
		storage := make([]byte, 4+4*len(values))
		data := storage[4:]
		for i, v := range values {
			binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
		}
		return data
	}
	a, err := engine.NewBufferFromHost(encode([]float32{1, 2, 3, 4, 5, 6, 7, 8}), xla.F32, []int64{2, 4}, devices[0])
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := a.Close(); err != nil {
			t.Error(err)
		}
	}()
	b, err := engine.NewBufferFromHost(encode([]float32{1, .5, 2, 0, 1.5, 1, 2, 0, 1, 1, 2, .5}), xla.F32, []int64{4, 3}, devices[0])
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	outputs, err := executable.Execute(context.Background(), []*xla.Buffer{a, b})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, b := range outputs {
			if err := b.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	if len(outputs) != 1 {
		t.Fatalf("got %d outputs, want 1", len(outputs))
	}
	var data bytes.Buffer
	if _, err := outputs[0].WriteTo(&data); err != nil {
		t.Fatal(err)
	}
	if data.Len() != 24 {
		t.Fatalf("got %d output bytes, want 24", data.Len())
	}
	for i, want := range []float32{11, 11.5, 9, 27, 27.5, 27} {
		got := math.Float32frombits(binary.LittleEndian.Uint32(data.Bytes()[4*i:]))
		if got != want {
			t.Errorf("output %d = %g, want %g", i, got, want)
		}
	}
}
