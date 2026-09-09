package gomlx

import (
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestNativeCumSum(t *testing.T) {
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
	input := [][]int64{{1<<53 + 1, 2, -3}, {4, -5, 6}}
	for axis := 0; axis < 2; axis++ {
		for _, reverse := range []bool{false, true} {
			for _, exclusive := range []bool{false, true} {
				options := compute.CumSumOptions{Reverse: reverse, Exclusive: exclusive}
				t.Run(fmt.Sprintf("axis%d/reverse%v/exclusive%v", axis, reverse, exclusive), func(t *testing.T) {
					want := [][]int64{{0, 0, 0}, {0, 0, 0}}
					for row := range input {
						for col := range input[row] {
							pos, n := col, 3
							if axis == 0 {
								pos, n = row, 2
							}
							for k := 0; k < n; k++ {
								include := k <= pos
								if reverse {
									include = k >= pos
								}
								if exclusive && k == pos {
									include = false
								}
								if include {
									if axis == 0 {
										want[row][col] += input[k][col]
									} else {
										want[row][col] += input[row][k]
									}
								}
							}
						}
					}
					got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
						return f.CumSum(v[0], axis, options)
					})
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("got %v want %v", got, want)
					}
				})
			}
		}
	}
	for _, tc := range []struct {
		name        string
		input, want any
	}{
		{"float", []float32{1, .5, -2}, []float32{1, 1.5, -.5}},
		{"unsigned", []uint64{1<<63 + 1, 2, 3}, []uint64{1<<63 + 1, 1<<63 + 3, 1<<63 + 6}},
		{"overflow", []uint8{255, 2, 3}, []uint8{255, 1, 4}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
				return f.CumSum(v[0], 0, compute.CumSumOptions{})
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

func TestCumSumValidation(t *testing.T) {
	for _, tc := range []struct {
		shape shapes.Shape
		axis  int
	}{
		{shapes.Make(dtypes.Bool, 3), 0}, {shapes.Make(dtypes.Float32), 0},
		{shapes.Make(dtypes.Float32, 3), -1}, {shapes.Make(dtypes.Float32, 3), 1},
	} {
		f := (&Backend{}).Builder("cumsum").Main()
		v, err := f.Parameter("x", tc.shape, nil)
		if err != nil {
			t.Fatal(err)
		}
		raw := f.(*function).raw
		n := len(raw.Ops)
		if _, err := f.CumSum(v, tc.axis, compute.CumSumOptions{}); err == nil {
			t.Fatalf("accepted %v axis%d", tc.shape, tc.axis)
		}
		if len(raw.Ops) != n {
			t.Fatal("invalid cumulative sum emitted IR")
		}
	}
}
