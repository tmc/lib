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

func TestNativeScatterSum(t *testing.T) {
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
		name    string
		indices any
		axis    int
	}{
		{"implicit", []int32{2, 0, 2, -1, 99}, 1},
		{"leading", [][]int32{{2, 0, 2, -1, 99}}, 0},
		{"trailing", [][]int64{{2}, {0}, {2}, {-1}, {99}}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{[][]int64{{9007199254740993, 2}, {3, 4}, {5, 6}}, tc.indices, [][]int64{{1, 2}, {3, 4}, {5, 6}, {100, 100}, {1000, 1000}}}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
				return f.ScatterSum(v[0], v[1], v[2], tc.axis, []int{1}, []int{0}, []int{0}, false, false)
			})
			if err != nil {
				t.Fatal(err)
			}
			want := [][]int64{{9007199254740996, 6}, {3, 4}, {11, 14}}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("got %v want %v", got, want)
			}
		})
	}
}

func TestScatterValidation(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		axis                      int
		window, inserted, mapping []int
		updates                   []int
	}{
		{"axis", 3, []int{1}, []int{0}, []int{0}, []int{2, 2}},
		{"implicit map", 1, []int{1}, []int{0}, []int{}, []int{2, 2}},
		{"duplicate window", 1, []int{1, 1}, nil, []int{0}, []int{2, 2}},
		{"duplicate inserted", 1, nil, []int{0, 0}, []int{0}, []int{2}},
		{"batch size", 1, []int{1}, []int{0}, []int{0}, []int{3, 2}},
		{"window size", 1, []int{1}, []int{0}, []int{0}, []int{2, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := (&Backend{}).Builder("scatter").Main()
			param := func(name string, s shapes.Shape) compute.Value {
				v, err := f.Parameter(name, s, nil)
				if err != nil {
					t.Fatal(err)
				}
				return v
			}
			x := param("x", shapes.Make(dtypes.Float32, 3, 2))
			idx := param("idx", shapes.Make(dtypes.Int32, 2))
			u := param("u", shapes.Make(dtypes.Float32, tc.updates...))
			raw := f.(*function).raw
			before := len(raw.Ops)
			if _, err := f.ScatterSum(x, idx, u, tc.axis, tc.window, tc.inserted, tc.mapping, false, false); err == nil {
				t.Fatal("accepted invalid scatter")
			}
			if len(raw.Ops) != before {
				t.Fatal("invalid scatter emitted IR")
			}
		})
	}
}
