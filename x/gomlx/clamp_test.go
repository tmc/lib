package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestNativeClamp(t *testing.T) {
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
		name            string
		lo, x, hi, want any
	}{
		{"scalar bounds", float32(-1), []float32{-2, 0, 4}, float32(2), []float32{-1, 0, 2}},
		{"tensor bounds", []int64{1<<53 + 1, 5}, []int64{1 << 53, 9}, []int64{1<<53 + 3, 8}, []int64{1<<53 + 1, 8}},
		{"unsigned mixed", uint64(1 << 63), []uint64{1<<63 - 1, 1<<63 + 3}, []uint64{1<<63 + 1, 1<<63 + 2}, []uint64{1 << 63, 1<<63 + 2}},
		{"scalar", int8(-2), int8(5), int8(3), int8(3)},
		{"reversed bounds", float64(4), []float64{-1, 3, 8}, float64(2), []float64{2, 2, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{tc.lo, tc.x, tc.hi}, func(f compute.Function, v []compute.Value) (compute.Value, error) { return f.Clamp(v[0], v[1], v[2]) })
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestClampValidation(t *testing.T) {
	for _, bound := range []shapes.Shape{shapes.Make(dtypes.Float64), shapes.Make(dtypes.Float32, 1), shapes.Make(dtypes.Float32, 3, 2)} {
		f := (&Backend{}).Builder("clamp").Main()
		x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
		if err != nil {
			t.Fatal(err)
		}
		limit, err := f.Parameter("limit", bound, nil)
		if err != nil {
			t.Fatal(err)
		}
		raw := f.(*function).raw
		n := len(raw.Ops)
		if _, err := f.Clamp(limit, x, limit); err == nil {
			t.Fatalf("accepted bound %v", bound)
		}
		if len(raw.Ops) != n {
			t.Fatal("invalid clamp emitted IR")
		}
	}
}
