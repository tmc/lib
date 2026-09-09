package gomlx

import (
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestNativeCompute(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN to run native acceptance")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		b.Finalize()
		if b.closeErr != nil {
			t.Error(b.closeErr)
		}
	}()
	for _, tc := range []struct {
		name                string
		left, right         []float32
		leftDims, rightDims []int
		want                []float32
		dot                 bool
		axes                [4][]int
	}{
		{"Add", []float32{1, 2, 3}, []float32{4, 5, 6}, []int{3}, []int{3}, []float32{5, 7, 9}, false, [4][]int{}},
		{"DotGeneral", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{2, 4}, []int{4, 3}, []float32{70, 80, 90, 158, 184, 210}, true, [4][]int{{1}, nil, {0}, nil}},

		{"AddBroadcast", []float32{1, 2}, []float32{10, 20, 30}, []int{2, 1}, []int{3}, []float32{11, 21, 31, 12, 22, 32}, false, [4][]int{}},
		{"AddScalar", []float32{2}, []float32{1, 2, 3}, nil, []int{3}, []float32{3, 4, 5}, false, [4][]int{}},
		{"DotBatched", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{5, 6, 7, 3}, []int{2, 2, 2}, []int{2, 2, 1}, []float32{17, 39, 53, 73}, true, [4][]int{{2}, {0}, {1}, {0}}},
		{"DotVector", []float32{1, 2, 3}, []float32{1, 2, 3}, []int{3}, []int{3}, []float32{14}, true, [4][]int{{0}, nil, {0}, nil}},
		{"DotOuter", []float32{1, 2}, []float32{3, 4}, []int{2}, []int{2}, []float32{3, 4, 6, 8}, true, [4][]int{}},
		{"DotPermuted", []float32{1, 2, 3, 4}, []float32{5, 6}, []int{2, 2}, []int{2}, []float32{23, 34}, true, [4][]int{{0}, nil, {0}, nil}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			builder := b.Builder(tc.name)
			f := builder.Main()
			var params []compute.Value
			var inputs []compute.Buffer
			for i, data := range [][]float32{tc.left, tc.right} {
				dims := tc.leftDims
				if i == 1 {
					dims = tc.rightDims
				}
				shape := shapes.Make(dtypes.Float32, dims...)
				v, err := b.BufferFromFlatData(0, data, shape)
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := v.Finalize(); err != nil {
						t.Error(err)
					}
				}()
				inputs = append(inputs, v)
				name := "left"
				if i == 1 {
					name = "right"
				}
				p, err := f.Parameter(name, shape, nil)
				if err != nil {
					t.Fatal(err)
				}
				params = append(params, p)
			}
			var out compute.Value
			var err error
			if tc.dot {
				out, err = f.DotGeneral(params[0], tc.axes[0], tc.axes[1], params[1], tc.axes[2], tc.axes[3], compute.DotGeneralConfig{})
			} else {
				out, err = f.Add(params[0], params[1])
			}
			if err != nil {
				t.Fatal(err)
			}
			if err = f.Return([]compute.Value{out}, nil); err != nil {
				t.Fatal(err)
			}
			if _, err = f.Add(params[0], params[0]); err == nil {
				t.Fatal("accepted operation after return")
			}
			exe, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				exe.Finalize()
				if err := exe.(*executable).closeErr; err != nil {
					t.Error(err)
				}
			}()

			outputs, err := exe.Execute(inputs, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range outputs {
				defer func() {
					if err := v.Finalize(); err != nil {
						t.Error(err)
					}
				}()
			}
			if len(outputs) != 1 {
				t.Fatalf("outputs: %d", len(outputs))
			}
			got := make([]float32, len(tc.want))
			if err := outputs[0].ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			wrong := slices.Clone(tc.want)
			wrong[0]++
			if slices.Equal(got, wrong) {
				t.Fatal("wrong-result control accepted")
			}
			t.Logf("native result %v; wrong-result control rejected", got)
		})
	}
}

func TestRejectedContracts(t *testing.T) {
	b := &Backend{}
	f := b.Builder("contracts").Main()
	if _, err := f.Parameter("x", shapes.Make(dtypes.Complex64, 2), nil); !errors.Is(err, compute.ErrNotImplemented) {
		t.Fatalf("dtype: %v", err)
	}
	if err := f.Builder().DeviceAssignment(); !errors.Is(err, compute.ErrNotImplemented) {
		t.Fatalf("device: %v", err)
	}
	p, err := f.Parameter("x", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	integer, err := f.Parameter("integer", shapes.Make(dtypes.Int32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Tanh(integer); !errors.Is(err, compute.ErrNotImplemented) {
		t.Fatalf("tanh: %v", err)
	}
	foreign := b.Builder("foreign").Main()
	q, err := foreign.Parameter("y", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Add(p, q); err == nil {
		t.Fatal("accepted foreign value")
	}
}

func TestInvalidContractions(t *testing.T) {
	for _, tc := range []struct {
		name           string
		ac, ab, bc, bb []int
	}{
		{"negative", []int{-1}, nil, []int{0}, nil},
		{"out of range", []int{2}, nil, []int{0}, nil},
		{"duplicate", []int{1, 1}, nil, []int{0, 1}, nil},
		{"batch overlap", []int{1}, []int{1}, []int{0}, []int{1}},
		{"count mismatch", []int{1}, nil, nil, nil},
		{"size mismatch", []int{0}, nil, []int{0}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := contractionShape([]int64{2, 3}, []int64{3, 4}, tc.ac, tc.ab, tc.bc, tc.bb); err == nil {
				t.Fatal("accepted invalid contraction")
			}
		})
	}
}

func TestInvalidBroadcast(t *testing.T) {
	f := (&Backend{}).Builder("broadcast").Main()
	a, err := f.Parameter("a", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := f.Parameter("b", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Add(a, b); err == nil {
		t.Fatal("accepted incompatible shapes")
	}
}
