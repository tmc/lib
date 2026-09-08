package gomlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"os"
	"testing"
)

func TestShapeBoundaries(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name  string
		dims  []int
		valid bool
	}{
		{"scalar", nil, true}, {"negative", []int{-1}, false}, {"negative after zero", []int{0, -1}, false},
		{"overflow", []int{maxInt, 2}, false}, {"zero first", []int{0, maxInt, 2}, true}, {"zero last", []int{maxInt, 2, 0}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validate(shapes.Shape{DType: dtypes.Float32, Dimensions: tc.dims})
			if (err == nil) != tc.valid {
				t.Fatalf("validate(%v): %v", tc.dims, err)
			}
		})
	}
	f := (&Backend{}).Builder("invalid").Main()
	if _, err := f.Constant([]float32{1}, -1); err == nil {
		t.Fatal("negative constant accepted")
	}
	v, err := f.Constant([]float32{1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Reshape(v, -1); err == nil {
		t.Fatal("negative reshape accepted")
	}
}
func TestNativeBackendFinalize(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("close")
	f := builder.Main()
	s := shapes.Make(dtypes.Float32, 1)
	p, err := f.Parameter("x", s, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{p}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	in, err := b.BufferFromFlatData(0, []float32{7}, s)
	if err != nil {
		t.Fatal(err)
	}
	out, err := e.Execute([]compute.Buffer{in}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.buffers) != 2 || len(b.executables) != 1 {
		t.Fatalf("tracked buffers %d, executables %d", len(b.buffers), len(b.executables))
	}
	b.Finalize()
	if b.closeErr != nil {
		t.Fatal(b.closeErr)
	}
	if len(b.buffers) != 0 || len(b.executables) != 0 {
		t.Fatal("resources remain tracked")
	}
	for _, v := range append(out, in) {
		if _, err := v.Shape(); err == nil {
			t.Fatal("live buffer after backend close")
		}
		if err := v.ToFlatData(make([]float32, 1)); err == nil {
			t.Fatal("readback after backend close succeeded")
		}
		if err := v.Finalize(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := e.Execute(nil, nil, 0); err == nil {
		t.Fatal("execute after backend close succeeded")
	}
	e.Finalize()
	b.Finalize()
	t.Log("backend close consumed 2 buffers and 1 executable; repeated closes and closed-resource checks pass")
}

func TestNativeReviewControls(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	builder := b.Builder("empty")
	f := builder.Main()
	v, err := f.Constant([]float32{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{v}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	out, err := e.Execute(nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("empty constant outputs: %d", len(out))
	}
	defer out[0].Finalize()
	if err := out[0].ToFlatData([]float32{}); err != nil {
		t.Fatal(err)
	}
	t.Log("dense<> zero-sized constant compiles and executes")
	builder = b.Builder("invalid donation")
	f = builder.Main()
	p, err := f.Parameter("x", shapes.Make(dtypes.Float32, 1), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{p}, nil); err != nil {
		t.Fatal(err)
	}
	e, err = builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer e.Finalize()
	if _, err := e.Execute([]compute.Buffer{nil}, []bool{true}, 0); err == nil {
		t.Fatal("accepted invalid donated input")
	}
	t.Log("invalid donated input returns error before donation defer is registered")
}
