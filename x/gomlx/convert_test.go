package gomlx

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestNativeBoolConversion(t *testing.T) {
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
	for _, tc := range []struct {
		name        string
		input, want any
		dtype       dtypes.DType
	}{
		{"bool to int32", []bool{false, true, true, false}, []int32{0, 1, 1, 0}, dtypes.Int32},
		{"bool to float32", []bool{false, true}, []float32{0, 1}, dtypes.Float32},
		{"float32 to bool", []float32{0, math.Float32frombits(0x80000000), -2, 3, float32(math.Inf(1)), float32(math.NaN())}, []bool{false, false, true, true, true, true}, dtypes.Bool},
		{"int64 to bool", []int64{0, -1, 1, 9007199254740993}, []bool{false, true, true, true}, dtypes.Bool},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node { return graph.ConvertDType(x, tc.dtype) })
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			out, err := e.Call(tc.input)
			if err != nil {
				t.Fatal(err)
			}
			defer out[0].FinalizeAll()
			if got := out[0].Value(); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}
