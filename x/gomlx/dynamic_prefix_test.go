package gomlx

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

type prefixExecutable struct {
	nativeExecutable
	mode     string
	closeErr error
	closes   int
	outputs  []*xla.Buffer
}

func (e *prefixExecutable) Close() error {
	e.closes++
	return errors.Join(e.nativeExecutable.Close(), e.closeErr)
}

func (e *prefixExecutable) Execute(ctx context.Context, inputs []*xla.Buffer) ([]*xla.Buffer, error) {
	if e.mode == "execute" {
		return nil, errors.New("injected prefix execute failure")
	}
	out, err := e.nativeExecutable.Execute(ctx, inputs)
	if err != nil {
		return nil, err
	}
	if e.mode == "count" {
		extra, err := e.nativeExecutable.Execute(ctx, inputs)
		if err != nil {
			for _, v := range out {
				_ = v.Close()
			}
			return nil, err
		}
		out = append(out, extra...)
	}
	if e.mode == "readback" {
		if err := out[0].Close(); err != nil {
			return nil, err
		}
	}
	e.outputs = out
	return out, nil
}

func TestNativeDynamicPrefixFailures(t *testing.T) {
	for _, mode := range []string{"compile", "execute", "count", "readback"} {
		t.Run(mode, func(t *testing.T) {
			path := os.Getenv("TMC_COMPUTE_PLUGIN")
			if path == "" {
				t.Skip("set TMC_COMPUTE_PLUGIN")
			}
			b, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			defer b.Finalize()
			builder := b.Builder("prefix-failure")
			f := builder.Main()
			n, err := f.Parameter("n", shapes.Make(dtypes.Int32), nil)
			if err != nil {
				t.Fatal(err)
			}
			y, err := f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: "size", Value: n})
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Return([]compute.Value{y}, nil); err != nil {
				t.Fatal(err)
			}
			ev, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			e := ev.(*executable)
			if e.template == nil || e.raw != nil {
				t.Fatal("all-static input graph did not defer compilation")
			}
			input, err := b.BufferFromFlatData(0, []int32{3}, shapes.Make(dtypes.Int32))
			if err != nil {
				t.Fatal(err)
			}
			realCompile := b.compile
			compileErr, closeErr := errors.New("injected prefix compile failure"), errors.New("injected prefix close failure")
			var prefix *prefixExecutable
			b.compile = func(code []byte, format string, opts *xla.CompileOptions) (nativeExecutable, error) {
				if mode == "compile" {
					return nil, compileErr
				}
				raw, err := realCompile(code, format, opts)
				if err != nil {
					return nil, err
				}
				prefix = &prefixExecutable{nativeExecutable: raw, mode: mode, closeErr: closeErr}
				return prefix, nil
			}
			_, err = e.Execute([]compute.Buffer{input}, nil, 0)
			if err == nil {
				t.Fatal("injected prefix failure succeeded")
			}
			if mode == "compile" {
				if !errors.Is(err, compileErr) {
					t.Fatalf("lost compile error: %v", err)
				}
			} else {
				if !errors.Is(err, closeErr) || prefix.closes != 1 {
					t.Fatalf("prefix closes=%d err=%v", prefix.closes, err)
				}
				want := "injected prefix execute"
				if mode == "count" {
					want = "output count mismatch"
				}
				if mode == "readback" {
					want = "buffer is closed"
				}
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("missing primary failure %q: %v", want, err)
				}
				for _, output := range prefix.outputs {
					if err := output.ToHost(make([]byte, 4)); err == nil {
						t.Fatal("prefix output was not closed")
					}
				}
			}
			if e.compileCalls != 0 || len(e.specializations) != 0 {
				t.Fatal("prefix failure populated final cache")
			}
			b.compile = realCompile
			out, err := e.Execute([]compute.Buffer{input}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]int32, 3)
			if err := out[0].ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			if got[0] != 0 || got[1] != 1 || got[2] != 2 || e.compileCalls != 1 || len(e.specializations) != 1 {
				t.Fatalf("retry result=%v compiles=%d", got, e.compileCalls)
			}
			if prefix != nil && prefix.closes != 1 {
				t.Fatal("retry closed failed prefix again")
			}
		})
	}
}

type prefixBuffer struct {
	*xla.Buffer
	closes   int
	closeErr error
}

func (b *prefixBuffer) Close() error {
	b.closes++
	return errors.Join(b.Buffer.Close(), b.closeErr)
}

func TestNativeDynamicPrefixCleanupErrors(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()
	raw := hlo.NewFunc("main")
	value, err := raw.IntegerConstant(3, hlo.Type{ElementType: "i32"})
	if err != nil {
		t.Fatal(err)
	}
	raw.Return(value)
	program, err := emitDynamicProgram(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, count := range []int{1, 2} {
		native, err := b.compileProgram([]byte(program), "mlir", nil)
		if err != nil {
			t.Fatal(err)
		}
		executableErr := errors.New("prefix executable close")
		compiled := &prefixExecutable{nativeExecutable: native, closeErr: executableErr}
		var outputs []dimensionBuffer
		var tracked []*prefixBuffer
		for i := 0; i < count; i++ {
			out, err := compiled.Execute(context.Background(), nil)
			if err != nil {
				t.Fatal(err)
			}
			buffer := &prefixBuffer{Buffer: out[0], closeErr: errors.New("prefix output close")}
			tracked = append(tracked, buffer)
			outputs = append(outputs, buffer)
		}
		result, err := readDimension(value, outputs, compiled.Close)
		if !errors.Is(err, executableErr) || compiled.closes != 1 {
			t.Fatalf("compiled closes=%d error=%v", compiled.closes, err)
		}
		for _, buffer := range tracked {
			if !errors.Is(err, buffer.closeErr) || buffer.closes != 1 {
				t.Fatalf("buffer closes=%d error=%v", buffer.closes, err)
			}
			if err := buffer.ToHost(make([]byte, 4)); err == nil {
				t.Fatal("buffer still readable")
			}
		}
		if count == 1 && result != 3 {
			t.Fatalf("result=%d", result)
		}
		if count == 2 && !strings.Contains(err.Error(), "output count mismatch") {
			t.Fatalf("lost count error: %v", err)
		}
	}
}

func TestDynamicSnapshotCopiesRecords(t *testing.T) {
	f := (&Backend{}).Builder("snapshot").Main().(*function)
	value := hlo.Value{ID: "x", Type: hlo.Type{Shape: []int64{2}, ElementType: "i32"}}
	f.raw.Args = []hlo.Value{value}
	f.raw.Returns = []hlo.Value{value}
	f.dynamicOps = []dynamicOp{{operand: value, result: value, operands: []hlo.Value{value}, axes: []int{1}, dimensions: []dimensionSpec{{static: 2, value: value}}, pads: []padSpec{{start: dimensionSpec{static: 1, value: value}}}}}
	saved := f.snapshot()
	f.raw.Args[0].Type.Shape[0] = 9
	f.dynamicOps[0].axes[0] = 7
	f.dynamicOps[0].dimensions[0].static = 8
	f.dynamicOps[0].pads[0].start.static = 4
	op := saved.dynamicOps[0]
	if saved.raw.Args[0].Type.Shape[0] != 2 || saved.raw.Returns[0].Type.Shape[0] != 2 || op.operand.Type.Shape[0] != 2 || op.result.Type.Shape[0] != 2 || op.operands[0].Type.Shape[0] != 2 || op.axes[0] != 1 || op.dimensions[0].static != 2 || op.dimensions[0].value.Type.Shape[0] != 2 || op.pads[0].start.static != 1 || op.pads[0].start.value.Type.Shape[0] != 2 {
		t.Fatal("snapshot shares mutable operation storage")
	}
}

func TestNativeDynamicAxisBindingConflict(t *testing.T) {
	for _, duplicate := range []bool{false, true} {
		name := "input-conflict"
		if duplicate {
			name = "duplicate-result-axes"
		}
		t.Run(name, func(t *testing.T) {
			path := os.Getenv("TMC_COMPUTE_PLUGIN")
			if path == "" {
				t.Skip("set TMC_COMPUTE_PLUGIN")
			}
			b, err := New(path)
			if err != nil {
				t.Fatal(err)
			}
			defer b.Finalize()
			builder := b.Builder(name)
			f := builder.Main()
			inputShape := shapes.MakeDynamic(dtypes.Int32, []int{-1}, []string{"batch"})
			if duplicate {
				inputShape = shapes.Make(dtypes.Int32, 2)
			}
			if _, err := f.Parameter("x", inputShape, nil); err != nil {
				t.Fatal(err)
			}
			n, err := f.Parameter("n", shapes.Make(dtypes.Int32), nil)
			if err != nil {
				t.Fatal(err)
			}
			axis := "batch"
			if duplicate {
				axis = "same"
			}
			specs := []compute.DynamicDimensionSpec{{Name: axis, Value: n}}
			if duplicate {
				two, err := f.Constant([]int32{2})
				if err != nil {
					t.Fatal(err)
				}
				specs = append(specs, compute.DynamicDimensionSpec{Name: axis, Value: two})
			}
			if _, err := f.DynamicIota(dtypes.Int32, 0, specs...); err != nil {
				t.Fatal(err)
			}
			later, err := f.DynamicIota(dtypes.Int32, 0, compute.DynamicDimensionSpec{Name: axis})
			if err != nil {
				t.Fatal(err)
			}
			fill, err := f.Constant([]int32{1})
			if err != nil {
				t.Fatal(err)
			}
			broadcast, err := f.DynamicBroadcastInDim(fill, nil, compute.DynamicDimensionSpec{Name: axis})
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Return([]compute.Value{later, broadcast}, nil); err != nil {
				t.Fatal(err)
			}
			ev, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			e := ev.(*executable)
			input, err := b.BufferFromFlatData(0, []int32{1, 2}, shapes.Make(dtypes.Int32, 2))
			if err != nil {
				t.Fatal(err)
			}
			for _, size := range []int32{3, 2} {
				nb, err := b.BufferFromFlatData(0, []int32{size}, shapes.Make(dtypes.Int32))
				if err != nil {
					t.Fatal(err)
				}
				out, err := e.Execute([]compute.Buffer{input, nb}, nil, 0)
				_ = nb.Finalize()
				if size == 3 {
					if err == nil || !strings.Contains(err.Error(), "conflicting values") || e.compileCalls != 0 || len(e.specializations) != 0 {
						t.Fatalf("conflict err=%v compiles=%d cache=%d", err, e.compileCalls, len(e.specializations))
					}
				} else {
					if err != nil {
						t.Fatal(err)
					}
					for i, v := range out {
						got := make([]int32, 2)
						if err := v.ToFlatData(got); err != nil {
							t.Fatal(err)
						}
						want := []int32{0, 1}
						if i == 1 {
							want = []int32{1, 1}
						}
						if got[0] != want[0] || got[1] != want[1] {
							t.Fatalf("later named operation %d=%v", i, got)
						}
					}
				}
			}
		})
	}
}
