package xla

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"
)

func TestNativeSharedBuffer(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN")
	}
	e, err := NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	devices, err := e.Devices()
	if err != nil {
		t.Fatal(err)
	}
	p, err := e.Compile([]byte(`module { func.func @main(%x: tensor<2x3xf32> {tf.aliasing_output = 0 : i32}) -> tensor<2x3xf32> { %y = stablehlo.add %x, %x : tensor<2x3xf32> return %y : tensor<2x3xf32> } }`), "mlir", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	b, data, err := e.NewSharedBuffer(F32, []int64{2, 3}, devices[0])
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	ptr := unsafe.Pointer(unsafe.SliceData(data))
	if uintptr(ptr)%64 != 0 {
		t.Fatal("unaligned allocation")
	}
	for _, value := range []float32{1, 7, -3} {
		for i := 0; i < 6; i++ {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value+float32(i)))
		}
		runtime.GC()
		outputs, err := p.Execute(context.Background(), []*Buffer{b})
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, 6)
		err = outputs[0].ReadFloats(got)
		if _, err := outputs[0].SharedData(); err == nil {
			t.Error("native output exposed as owned storage")
		}
		if closeErr := outputs[0].Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 6; i++ {
			if v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:])); v != value+float32(i) {
				t.Fatalf("shared input mutated by executable: %v", v)
			}
		}
		for i, v := range got {
			if v != 2*(value+float32(i)) {
				t.Fatalf("element %d = %v", i, v)
			}
		}
		alias, err := b.SharedData()
		if err != nil || unsafe.SliceData(alias) != unsafe.SliceData(data) {
			t.Fatalf("alias: %v", err)
		}
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := b.SharedData(); err == nil {
		t.Fatal("closed data accepted")
	}
	if _, ok := sharedAllocations.Load(ptr); ok {
		t.Fatal("CPU plugin did not release owned allocation on close")
	}
	for _, dims := range [][]int64{nil, {0}, {2, 0, 3}} {
		b, data, err := e.NewSharedBuffer(U8, dims, devices[0])
		if err != nil {
			t.Fatal(err)
		}
		n, _ := bufferSize(U8, dims)
		if len(data) != n {
			t.Fatalf("empty/scalar size %d want %d", len(data), n)
		}
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}
}
