package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeDynamicRuntimeOperations(t *testing.T) {
	for _, operation := range []string{"reshape", "broadcast", "iota", "pad"} {
		t.Run(operation, func(t *testing.T) {
			path := os.Getenv("TMC_COMPUTE_PLUGIN")
			if path == "" {
				t.Skip("set TMC_COMPUTE_PLUGIN")
			}
			b, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			defer b.Finalize()
			builder := b.Builder(operation)
			f := builder.Main()
			must := func(v compute.Value, err error) compute.Value {
				t.Helper()
				if err != nil {
					t.Fatal(err)
				}
				return v
			}
			xshape := shapes.Make(dtypes.Int32, 1)
			xdata := []int32{7}
			if operation == "reshape" {
				xshape = shapes.Make(dtypes.Int32, 6)
				xdata = []int32{1, 2, 3, 4, 5, 6}
			}
			if operation == "pad" {
				xshape = shapes.Make(dtypes.Int32, 2)
				xdata = []int32{4, 5}
			}
			x := must(f.Parameter("x", xshape, nil))
			n := must(f.Parameter("n", shapes.Make(dtypes.Int32), nil))
			// Exercise an actual runtime expression, not just direct parameter reads.
			one := must(f.Constant([]int32{1}))
			size := must(f.Mul(n, one))
			spec := compute.DynamicDimensionSpec{Name: "size", Value: size}
			var y compute.Value
			switch operation {
			case "reshape":
				y = must(f.DynamicReshape(x, spec, compute.DynamicDimensionSpec{Name: "rest"}))
			case "broadcast":
				y = must(f.DynamicBroadcastInDim(x, []int{0}, spec))
			case "iota":
				y = must(f.DynamicIota(dtypes.Int32, 0, spec))
			case "pad":
				y = must(f.DynamicPad(x, one, compute.DynamicPadAxis{StartValue: size, End: 1, Interior: 1, TargetAxisName: "padded"}))
			}
			ds := must(f.DynamicShape(y))
			if err := f.Return([]compute.Value{y, ds}, nil); err != nil {
				t.Fatal(err)
			}
			ev, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			e := ev.(*executable)
			input, err := b.BufferFromFlatData(0, xdata, xshape)
			if err != nil {
				t.Fatal(err)
			}
			for run, nv := range []int32{2, 3, 2} {
				nb, err := b.BufferFromFlatData(0, []int32{nv}, shapes.Make(dtypes.Int32))
				if err != nil {
					t.Fatal(err)
				}
				out, err := e.Execute([]compute.Buffer{input, nb}, nil, 0)
				_ = nb.Finalize()
				if err != nil {
					t.Fatal(err)
				}
				dims := []int{int(nv)}
				want := make([]int32, int(nv))
				switch operation {
				case "reshape":
					dims = []int{int(nv), 6 / int(nv)}
					want = xdata
				case "broadcast":
					for i := range want {
						want[i] = 7
					}
				case "iota":
					for i := range want {
						want[i] = int32(i)
					}
				case "pad":
					dims = []int{int(nv) + 4}
					want = make([]int32, dims[0])
					for i := range want {
						want[i] = 1
					}
					want[nv] = 4
					want[nv+2] = 5
				}
				got := make([]int32, len(want))
				if err := out[0].ToFlatData(got); err != nil {
					t.Fatal(err)
				}
				gotDims := make([]int32, len(dims))
				if err := out[1].ToFlatData(gotDims); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("n=%d: got %v want %v", nv, got, want)
				}
				for i, d := range dims {
					if gotDims[i] != int32(d) {
						t.Fatalf("shape=%v want=%v", gotDims, dims)
					}
				}
				for _, v := range out {
					_ = v.Finalize()
				}
				if wantCalls := []int{1, 2, 2}[run]; e.compileCalls != wantCalls {
					t.Fatalf("compileCalls=%d want=%d", e.compileCalls, wantCalls)
				}
			}
			bad := int32(-1)
			if operation == "reshape" {
				bad = 4
			}
			if operation == "pad" {
				bad = -9
			}
			nb, err := b.BufferFromFlatData(0, []int32{bad}, shapes.Make(dtypes.Int32))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := e.Execute([]compute.Buffer{input, nb}, nil, 0); err == nil {
				t.Fatal("invalid runtime size accepted")
			}
			if e.compileCalls != 2 {
				t.Fatal("invalid size reached final compilation")
			}
		})
	}
}

func TestDynamicOperationValidation(t *testing.T) {
	b := (&Backend{}).Builder("validation")
	f := b.Main().(*function)
	x, err := f.Parameter("x", shapes.MakeDynamic(dtypes.Int32, []int{-1}, []string{"batch"}), nil)
	if err != nil {
		t.Fatal(err)
	}
	n, err := f.Parameter("n", shapes.Make(dtypes.Float32), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, build := range []func() (compute.Value, error){
		func() (compute.Value, error) {
			return f.DynamicReshape(x, compute.DynamicDimensionSpec{Name: "a"}, compute.DynamicDimensionSpec{Name: "b"})
		},
		func() (compute.Value, error) {
			return f.DynamicReshape(x, compute.DynamicDimensionSpec{Name: "a", Value: n})
		},
		func() (compute.Value, error) {
			return f.DynamicBroadcastInDim(x, []int{0}, compute.DynamicDimensionSpec{Name: "different"})
		},
		func() (compute.Value, error) {
			return f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: "unknown"})
		},
		func() (compute.Value, error) { return f.DynamicPad(x, x) },
	} {
		before := len(f.raw.Ops)
		if _, err := build(); err == nil {
			t.Fatal("invalid dynamic operation accepted")
		}
		if len(f.raw.Ops) != before {
			t.Fatal("invalid operation emitted IR")
		}
	}
}

func TestNativeDynamicGraphOperations(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	exec, err := graph.NewExec(b, func(x, n *graph.Node) []*graph.Node {
		spec := graph.NamedDynamicDim("size", n)
		reshaped := graph.DynamicReshape(x, spec, graph.NamedInferredDim("rest"))
		iota := graph.DynamicIota(x.Graph(), dtypes.Int32, 0, spec)
		broadcast := graph.DynamicBroadcastInDim(graph.Const(x.Graph(), []int32{9}), []int{0}, spec)
		padded := graph.DynamicPad(x, graph.Const(x.Graph(), int32(0)), graph.DynamicPadAxisSpec{StartNode: n, TargetAxisName: "padded"})
		return []*graph.Node{reshaped, iota, broadcast, padded}
	})
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()
	for _, n := range []int32{2, 3} {
		out, err := exec.Call([]int32{1, 2, 3, 4, 5, 6}, n)
		if err != nil {
			t.Fatal(err)
		}
		if got := out[0].Shape().Dimensions; !reflect.DeepEqual(got, []int{int(n), 6 / int(n)}) {
			t.Fatalf("shape=%v", got)
		}
		wantIota := make([]int32, n)
		wantBroadcast := make([]int32, n)
		for i := range wantIota {
			wantIota[i] = int32(i)
			wantBroadcast[i] = 9
		}
		if !reflect.DeepEqual(out[1].Value(), wantIota) || !reflect.DeepEqual(out[2].Value(), wantBroadcast) {
			t.Fatal("graph values mismatch")
		}
		wantPad := make([]int32, int(n)+6)
		copy(wantPad[n:], []int32{1, 2, 3, 4, 5, 6})
		if !reflect.DeepEqual(out[3].Value(), wantPad) {
			t.Fatal("graph padding mismatch")
		}
		for _, v := range out {
			v.FinalizeAll()
		}
	}
}

func TestNativeDynamicDataDimension(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("data-dimension")
	f := builder.Main()
	x, err := f.Parameter("lengths", shapes.Make(dtypes.Int32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	size, err := f.ReduceSum(x)
	if err != nil {
		t.Fatal(err)
	}
	value, err := f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: "sum", Value: size})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{value}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range [][]int32{{1, 2}, {2, 3}, {-1, 2}} {
		input, err := b.BufferFromFlatData(0, data, shapes.Make(dtypes.Int32, 2))
		if err != nil {
			t.Fatal(err)
		}
		out, err := e.Execute([]compute.Buffer{input}, nil, 0)
		_ = input.Finalize()
		if err != nil {
			t.Fatal(err)
		}
		want := make([]int32, data[0]+data[1])
		for i := range want {
			want[i] = int32(i)
		}
		got := make([]int32, len(want))
		if err := out[0].ToFlatData(got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got=%v want=%v", got, want)
		}
		_ = out[0].Finalize()
	}
}

func TestNativeDynamicNamedAxes(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	builder := b.Builder("named-dimensions")
	f := builder.Main()
	must := func(v compute.Value, err error) compute.Value {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	x := must(f.Parameter("x", shapes.MakeDynamic(dtypes.Int32, []int{-1, 3}, []string{"batch", ""}), nil))
	batch := must(f.DynamicDimensionSize(x, 0))
	three := must(f.Constant([]int32{3}))
	count := must(f.Mul(batch, three))
	flat := must(f.DynamicReshape(x, compute.DynamicDimensionSpec{Name: "count", Value: count}))
	iota := must(f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: "count"}))
	broadcast := must(f.DynamicBroadcastInDim(three, nil, compute.DynamicDimensionSpec{Name: "batch"}))
	padded := must(f.DynamicPad(flat, three, compute.DynamicPadAxis{Start: -1, End: 1, TargetAxisName: "padded"}))
	shape, err := f.Shape(flat)
	if err != nil {
		t.Fatal(err)
	}
	if shape.AxisName(0) != "count" || !shape.IsDynamic() {
		t.Fatalf("lost reshape axis metadata: %v", shape)
	}
	if err := f.Return([]compute.Value{flat, iota, broadcast, padded}, nil); err != nil {
		t.Fatal(err)
	}
	e, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	for _, batch := range []int{2, 4} {
		data := make([]int32, batch*3)
		for i := range data {
			data[i] = int32(i)
		}
		input, err := b.BufferFromFlatData(0, data, shapes.Make(dtypes.Int32, batch, 3))
		if err != nil {
			t.Fatal(err)
		}
		out, err := e.Execute([]compute.Buffer{input}, nil, 0)
		_ = input.Finalize()
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range out {
			want := data
			if i == 2 {
				want = make([]int32, batch)
				for j := range want {
					want[j] = 3
				}
			}
			if i == 3 {
				want = append(append([]int32(nil), data[1:]...), 3)
			}
			got := make([]int32, len(want))
			if err := v.ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("output %d got=%v want=%v", i, got, want)
			}
			_ = v.Finalize()
		}
	}
}
