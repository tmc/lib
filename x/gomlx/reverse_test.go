package gomlx

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"os"
	"reflect"
	"testing"
)

func TestNativeReverse(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	input := [][]int64{{9007199254740993, 2, 3}, {4, 5, 6}}
	for _, tc := range []struct {
		name string
		axes []int
		want [][]int64
	}{
		{"none", nil, input}, {"rows", []int{0}, [][]int64{{4, 5, 6}, {9007199254740993, 2, 3}}}, {"both", []int{0, 1}, [][]int64{{6, 5, 4}, {3, 2, 9007199254740993}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, v []compute.Value) (compute.Value, error) { return f.Reverse(v[0], tc.axes...) })
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestReverseValidation(t *testing.T) {
	for _, axes := range [][]int{{-1}, {2}, {0, 0}} {
		f := (&Backend{}).Builder("reverse").Main()
		x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 2, 3), nil)
		if err != nil {
			t.Fatal(err)
		}
		raw := f.(*function).raw
		n := len(raw.Ops)
		if _, err := f.Reverse(x, axes...); err == nil {
			t.Fatalf("accepted axes %v", axes)
		}
		if len(raw.Ops) != n {
			t.Fatal("invalid reverse emitted IR")
		}
	}
}
