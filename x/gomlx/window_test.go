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

func TestNativeReduceWindow(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	for _, tc := range []struct {
		name                           string
		input, want                    any
		op                             compute.ReduceOpType
		window, stride, base, dilation []int
		padding                        [][2]int
	}{
		{name: "default stride", input: []float32{1, 2, 3, 4}, want: []float32{3, 7}, op: compute.ReduceOpSum, window: []int{2}},
		{name: "product padding", input: []float32{2, 3, 4}, want: []float32{2, 6, 12, 4}, op: compute.ReduceOpProduct, window: []int{2}, stride: []int{1}, padding: [][2]int{{1, 1}}},
		{name: "input dilation", input: []float32{1, 2, 3}, want: []float32{3, 2, 5}, op: compute.ReduceOpSum, window: []int{3}, stride: []int{1}, base: []int{2}},
		{name: "window dilation", input: []float32{1, 9, 3, 8, 5}, want: []float32{4, 17, 8}, op: compute.ReduceOpSum, window: []int{2}, stride: []int{1}, dilation: []int{2}},
		{name: "matrix max", input: [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, want: [][]float32{{5, 6}, {8, 9}}, op: compute.ReduceOpMax, window: []int{2, 2}, stride: []int{1, 1}},
		{name: "signed max padding", input: []int64{-9007199254740993, -9007199254740995}, want: []int64{-9007199254740993, -9007199254740993, -9007199254740995}, op: compute.ReduceOpMax, window: []int{2}, stride: []int{1}, padding: [][2]int{{1, 1}}},
		{name: "unsigned min padding", input: []uint64{9223372036854775811, 9223372036854775813}, want: []uint64{9223372036854775811, 9223372036854775811, 9223372036854775813}, op: compute.ReduceOpMin, window: []int{2}, stride: []int{1}, padding: [][2]int{{1, 1}}},
		{name: "scalar", input: float64(3), want: float64(3), op: compute.ReduceOpSum},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
				return f.ReduceWindow(v[0], tc.op, tc.window, tc.stride, tc.base, tc.dilation, tc.padding)
			})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestReduceWindowValidation(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		window, stride, base []int
	}{
		{"zero window", []int{0}, nil, nil},
		{"stride rank", []int{2}, []int{1, 1}, nil},
		{"zero dilation", []int{2}, nil, []int{0}},
		{"oversized window", []int{5}, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := (&Backend{}).Builder("window").Main()
			x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 4), nil)
			if err != nil {
				t.Fatal(err)
			}
			raw := f.(*function).raw
			before := len(raw.Ops)
			if _, err := f.ReduceWindow(x, compute.ReduceOpSum, tc.window, tc.stride, tc.base, nil, nil); err == nil {
				t.Fatal("accepted invalid window")
			}
			if len(raw.Ops) != before {
				t.Fatal("invalid window emitted IR")
			}
		})
	}
}
