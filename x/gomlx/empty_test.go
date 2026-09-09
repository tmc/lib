package gomlx

import (
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestNativeEmptyBuffers(t *testing.T) {
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
	for _, dtype := range []dtypes.DType{dtypes.Bool, dtypes.Float16, dtypes.BFloat16, dtypes.Float32, dtypes.Float64, dtypes.Int8, dtypes.Int32, dtypes.Int64, dtypes.Uint8, dtypes.Uint32, dtypes.Uint64} {
		for _, dims := range [][]int{{0}, {2, 0}, {0, 3}} {
			t.Run(fmt.Sprintf("%s/%v", dtype, dims), func(t *testing.T) {
				shape := shapes.Make(dtype, dims...)
				flat := newFlat(dtype, 0)
				in, err := b.BufferFromFlatData(0, flat, shape)
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := in.Finalize(); err != nil {
						t.Error(err)
					}
				}()
				host := newFlat(dtype, 0)
				if err := in.ToFlatData(host); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(host, flat) {
					t.Fatal("empty roundtrip differs")
				}
				builder := b.Builder("empty")
				f := builder.Main()
				x, err := f.Parameter("x", shape, nil)
				if err != nil {
					t.Fatal(err)
				}
				y := x
				if dtype != dtypes.Bool {
					y, err = f.CumSum(x, 0, compute.CumSumOptions{Exclusive: true, Reverse: true})
					if err != nil {
						t.Fatal(err)
					}
				}
				if err := f.Return([]compute.Value{y}, nil); err != nil {
					t.Fatal(err)
				}
				e, err := builder.Compile()
				if err != nil {
					t.Fatal(err)
				}
				defer e.Finalize()
				out, err := e.Execute([]compute.Buffer{in}, nil, 0)
				if err != nil {
					t.Fatal(err)
				}
				for _, v := range out {
					defer func() {
						if err := v.Finalize(); err != nil {
							t.Error(err)
						}
					}()
					got, err := v.Shape()
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(got.Dimensions, dims) || got.DType != dtype {
						t.Fatalf("shape %v want %v", got, shape)
					}
					if err := v.ToFlatData(host); err != nil {
						t.Fatal(err)
					}
				}
			})
		}
	}
}
