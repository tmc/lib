package gomlx

import (
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestNativeSharedInput(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	if b.HasSharedBuffers() {
		t.Fatal("unqualified native output sharing advertised")
	}
	shape := shapes.Make(dtypes.Float32, 2, 3)
	input, flat, err := b.NewSharedBuffer(0, shape)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Finalize()
	build := b.Builder("shared")
	f := build.Main()
	x, err := f.Parameter("x", shape, nil)
	if err != nil {
		t.Fatal(err)
	}
	y, err := f.Add(x, x)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{y}, nil); err != nil {
		t.Fatal(err)
	}
	exe, err := build.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer exe.Finalize()
	if _, err := exe.Execute([]compute.Buffer{input}, []bool{true}, 0); err == nil || !strings.Contains(err.Error(), "cannot donate shared") {
		t.Fatalf("donation: %v", err)
	}
	for _, start := range []float32{2, 7, -1} {
		values := flat.([]float32)
		for i := range values {
			values[i] = start + float32(i)
		}
		outputs, err := exe.Execute([]compute.Buffer{input}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]float32, 6)
		err = outputs[0].ToFlatData(got)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := outputs[0].Data(); err == nil {
			t.Error("native output claims shared data")
		}
		if err := outputs[0].Finalize(); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v != values[i]*2 {
				t.Fatalf("value %d = %v", i, v)
			}
		}
		alias, err := input.Data()
		if err != nil || &alias.([]float32)[0] != &values[0] {
			t.Fatalf("data alias: %v", err)
		}
	}
	if err := input.Finalize(); err != nil {
		t.Fatal(err)
	}
	if _, err := input.Data(); err == nil {
		t.Fatal("closed data accepted")
	}
	for _, dtype := range []dtypes.DType{dtypes.Bool, dtypes.Int8, dtypes.Int16, dtypes.Int32, dtypes.Int64, dtypes.Float32, dtypes.Float64, dtypes.Complex64, dtypes.Complex128} {
		t.Run(dtype.String(), func(t *testing.T) {
			for _, n := range []int{0, 3} {
				buf, data, err := b.NewSharedBuffer(0, shapes.Make(dtype, n))
				if err != nil {
					t.Fatal(err)
				}
				if reflect.TypeOf(data).Elem() != dtype.GoType() || reflect.ValueOf(data).Len() != n {
					t.Fatalf("typed data %T", data)
				}
				if err := buf.Finalize(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func ExampleBackend_NewSharedBuffer() {
	var backend Backend
	_, _, err := backend.NewSharedBuffer(0, shapes.Make(dtypes.Float32, 2))
	fmt.Println(err)
	// Output: backend closed
}
