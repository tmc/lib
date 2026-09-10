package gomlx

import (
	"context"
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/tmc/lib/x/xla"
)

func TestDynamicShapeValidation(t *testing.T) {
	b := (&Backend{}).Builder("invalid")
	f := b.Main()
	for _, shape := range []shapes.Shape{
		{DType: dtypes.Float32, Dimensions: []int{shapes.DynamicDim}},
		{DType: dtypes.Float32, Dimensions: []int{3}, AxisNames: []string{"batch"}},
	} {
		if _, err := f.Parameter("x", shape, nil); err == nil {
			t.Fatalf("Parameter(%s) succeeded", shape)
		}
	}
}

func TestDynamicSpecializationStructure(t *testing.T) {
	b := (&Backend{}).Builder("structure")
	f := b.Main().(*function)
	template := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
	x, err := f.Parameter("x", template, nil)
	if err != nil {
		t.Fatal(err)
	}
	dim, err := f.DynamicDimensionSize(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	shape, err := f.DynamicShape(x)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{dim, shape}, nil); err != nil {
		t.Fatal(err)
	}
	program, inputs, outputs, err := f.specialize(shapes.AxisBindings{"batch": 2})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(program, "?") || !strings.Contains(program, "tensor<2x3xf32>") || !strings.Contains(program, "dense<[2, 3]>") {
		t.Fatalf("specialization is not concrete:\n%s", program)
	}
	if got := inputs[0].Dimensions; got[0] != 2 || got[1] != 3 {
		t.Fatalf("inputs=%v", inputs)
	}
	if len(outputs) != 2 || outputs[0].Rank() != 0 || outputs[1].Dimensions[0] != 2 {
		t.Fatalf("outputs=%v", outputs)
	}
}

func TestDynamicUnsupportedOperationRejectsBeforeCompile(t *testing.T) {
	b := (&Backend{}).Builder("unsupported")
	f := b.Main().(*function)
	template := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{"batch"})
	x, err := f.Parameter("x", template, nil)
	if err != nil {
		t.Fatal(err)
	}
	value, err := f.Add(x, x)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{value}, nil); err != nil {
		t.Fatal(err)
	}
	if err := f.checkDynamicSlice(); err == nil {
		t.Fatal("unsupported dynamic add was accepted")
	}
}

func TestDynamicAxisConflictRejectsBeforeIR(t *testing.T) {
	b := (&Backend{}).Builder("axis-conflict")
	f := b.Main().(*function)
	a, err := f.Parameter("a", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{"batch"}), nil)
	if err != nil {
		t.Fatal(err)
	}
	bv, err := f.Parameter("b", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{"items"}), nil)
	if err != nil {
		t.Fatal(err)
	}
	before := len(f.raw.Ops)
	if _, err := f.Sub(a, bv); err == nil {
		t.Fatal("conflicting names accepted")
	}
	if len(f.raw.Ops) != before {
		t.Fatal("conflict emitted IR")
	}
}

func TestNativeDynamicRecompiling(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN to run native acceptance")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Finalize()

	builder := b.Builder("dynamic")
	f := builder.Main()
	template := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
	x, err := f.Parameter("x", template, nil)
	if err != nil {
		t.Fatal(err)
	}
	y, err := f.Parameter("y", template, nil)
	if err != nil {
		t.Fatal(err)
	}
	weightShape := shapes.Make(dtypes.Float32)
	weight, err := f.Parameter("weight", weightShape, nil)
	if err != nil {
		t.Fatal(err)
	}
	dim, err := f.DynamicDimensionSize(x, 0)
	if err != nil {
		t.Fatal(err)
	}
	shape, err := f.DynamicShape(x)
	if err != nil {
		t.Fatal(err)
	}
	staticDim, err := f.DynamicDimensionSize(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := f.Sub(x, y)
	if err != nil {
		t.Fatal(err)
	}
	square, err := f.Mul(delta, delta)
	if err != nil {
		t.Fatal(err)
	}
	sum, err := f.ReduceSum(square, 1)
	if err != nil {
		t.Fatal(err)
	}
	distance, err := f.Sqrt(sum)
	if err != nil {
		t.Fatal(err)
	}
	total, err := f.ReduceSum(square)
	if err != nil {
		t.Fatal(err)
	}
	weighted, err := f.Mul(total, weight)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{dim, shape, staticDim, distance, weighted}, nil); err != nil {
		t.Fatal(err)
	}
	execValue, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	exec := execValue.(*executable)
	exec.maxSpecializations = 2
	defer exec.Finalize()
	if exec.compileCalls != 0 {
		t.Fatalf("compile calls after graph compile = %d, want 0", exec.compileCalls)
	}
	weightBuffer, err := b.BufferFromFlatData(0, []float32{2}, weightShape)
	if err != nil {
		t.Fatal(err)
	}
	defer weightBuffer.Finalize()
	realCompile := b.compile
	failed := false
	b.compile = func(code []byte, format string, opts *xla.CompileOptions) (nativeExecutable, error) {
		if !failed {
			failed = true
			return nil, errors.New("injected compile failure")
		}
		return realCompile(code, format, opts)
	}
	probeShape := shapes.Make(dtypes.Float32, 2, 3)
	probeA, _ := b.BufferFromFlatData(0, make([]float32, 6), probeShape)
	probeB, _ := b.BufferFromFlatData(0, ones(6), probeShape)
	if _, err := exec.Execute([]compute.Buffer{probeA, probeB, weightBuffer}, nil, 0); err == nil {
		t.Fatal("injected compile succeeded")
	}
	_ = probeA.Finalize()
	_ = probeB.Finalize()
	if exec.compileCalls != 1 || len(exec.specializations) != 0 {
		t.Fatalf("failed compile retained state")
	}

	wantCompiles := []int{2, 3, 3, 4, 5}
	for run, batch := range []int{2, 4, 2, 6, 4} {
		concrete := shapes.Make(dtypes.Float32, batch, 3)
		input, err := b.BufferFromFlatData(0, make([]float32, batch*3), concrete)
		if err != nil {
			t.Fatal(err)
		}
		other, err := b.BufferFromFlatData(0, ones(batch*3), concrete)
		if err != nil {
			t.Fatal(err)
		}
		outputs, err := exec.Execute([]compute.Buffer{input, other, weightBuffer}, nil, 0)
		_ = input.Finalize()
		_ = other.Finalize()
		if err != nil {
			t.Fatalf("run %d: %v", run, err)
		}
		gotDim := make([]int32, 1)
		if err := outputs[0].ToFlatData(gotDim); err != nil {
			t.Fatal(err)
		}
		gotShape := make([]int32, 2)
		if err := outputs[1].ToFlatData(gotShape); err != nil {
			t.Fatal(err)
		}
		gotStaticDim := make([]int32, 1)
		if err := outputs[2].ToFlatData(gotStaticDim); err != nil {
			t.Fatal(err)
		}
		gotDistance := make([]float32, batch)
		if err := outputs[3].ToFlatData(gotDistance); err != nil {
			t.Fatal(err)
		}
		gotWeighted := make([]float32, 1)
		if err := outputs[4].ToFlatData(gotWeighted); err != nil {
			t.Fatal(err)
		}
		for _, output := range outputs {
			_ = output.Finalize()
		}
		if gotDim[0] != int32(batch) || gotShape[0] != int32(batch) || gotShape[1] != 3 {
			t.Fatalf("run %d: dim=%d shape=%v", run, gotDim, gotShape)
		}
		if gotStaticDim[0] != 3 {
			t.Fatalf("static dimension=%v", gotStaticDim)
		}
		if gotWeighted[0] != float32(batch*6) {
			t.Fatalf("weighted=%v want=%d", gotWeighted, batch*6)
		}
		for _, value := range gotDistance {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value)-math.Sqrt(3)) > 1e-5 {
				t.Fatalf("run %d: distance=%v", run, gotDistance)
			}
		}
		wantCalls := wantCompiles[run]
		if exec.compileCalls != wantCalls {
			t.Fatalf("run %d: compile calls=%d want=%d", run, exec.compileCalls, wantCalls)
		}
	}
	if len(exec.specializations) != 2 {
		t.Fatalf("cache size=%d want=2", len(exec.specializations))
	}

	bad, err := b.BufferFromFlatData(0, make([]float32, 8), shapes.Make(dtypes.Float32, 2, 4))
	if err != nil {
		t.Fatal(err)
	}
	good, err := b.BufferFromFlatData(0, make([]float32, 6), shapes.Make(dtypes.Float32, 2, 3))
	if err != nil {
		t.Fatal(err)
	}
	_, err = exec.Execute([]compute.Buffer{good, bad, weightBuffer}, nil, 0)
	_ = good.Finalize()
	_ = bad.Finalize()
	if err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("bad static dimension: %v", err)
	}
	if exec.compileCalls != 5 {
		t.Fatalf("invalid input compiled: calls=%d", exec.compileCalls)
	}
}

func ones(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

type trackedExecutable struct {
	nativeExecutable
	closeErr error
	closes   int
	entered  chan struct{}
	release  chan struct{}
}

func (e *trackedExecutable) Close() error {
	e.closes++
	return errors.Join(e.nativeExecutable.Close(), e.closeErr)
}

func (e *trackedExecutable) Execute(ctx context.Context, inputs []*xla.Buffer) ([]*xla.Buffer, error) {
	if e.entered != nil {
		close(e.entered)
		<-e.release
	}
	return e.nativeExecutable.Execute(ctx, inputs)
}

func nativeDynamicIdentity(t *testing.T) (*Backend, *executable) {
	t.Helper()
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN to run native acceptance")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Finalize)
	builder := b.Builder("dynamic-lifecycle")
	f := builder.Main()
	x, err := f.Parameter("x", shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim}, []string{"batch"}), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{x}, nil); err != nil {
		t.Fatal(err)
	}
	value, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	return b, value.(*executable)
}

func dynamicInput(t *testing.T, b *Backend, batch int) compute.Buffer {
	t.Helper()
	input, err := b.BufferFromFlatData(0, ones(batch), shapes.Make(dtypes.Float32, batch))
	if err != nil {
		t.Fatal(err)
	}
	return input
}

func TestDynamicEvictionCloseErrorPolicy(t *testing.T) {
	b, e := nativeDynamicIdentity(t)
	e.maxSpecializations = 1
	oldErr, newErr := errors.New("old close"), errors.New("new close")
	realCompile := b.compile
	var compiled []*trackedExecutable
	b.compile = func(code []byte, format string, opts *xla.CompileOptions) (nativeExecutable, error) {
		raw, err := realCompile(code, format, opts)
		if err != nil {
			return nil, err
		}
		tracked := &trackedExecutable{nativeExecutable: raw}
		switch len(compiled) {
		case 0:
			tracked.closeErr = oldErr
		case 1:
			tracked.closeErr = newErr
		}
		compiled = append(compiled, tracked)
		return tracked, nil
	}
	a, c := dynamicInput(t, b, 2), dynamicInput(t, b, 3)
	out, err := e.Execute([]compute.Buffer{a}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		if err := v.Finalize(); err != nil {
			t.Fatal(err)
		}
	}
	out, err = e.Execute([]compute.Buffer{c}, nil, 0)
	if !errors.Is(err, oldErr) || !errors.Is(err, newErr) || len(out) != 0 {
		t.Fatalf("eviction outputs=%v error=%v", out, err)
	}
	if len(e.specializations) != 0 || len(compiled) != 2 || compiled[0].closes != 1 || compiled[1].closes != 1 {
		t.Fatalf("failed eviction retained cache or incorrect cleanup: cache=%d compiled=%v", len(e.specializations), compiled)
	}
	out, err = e.Execute([]compute.Buffer{c}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range out {
		if err := v.Finalize(); err != nil {
			t.Fatal(err)
		}
	}
	if len(compiled) != 3 || len(e.specializations) != 1 {
		t.Fatal("retry did not compile and cache fresh executable")
	}
	compiled[2].closeErr = newErr
	e.Finalize()
	retained := b.closeErr
	e.Finalize()
	if b.closeErr != retained || !errors.Is(e.closeErr, newErr) || !errors.Is(b.closeErr, newErr) {
		t.Fatalf("repeated finalize changed retained errors: executable=%v backend=%v", e.closeErr, b.closeErr)
	}
	for i, raw := range compiled {
		if raw.closes != 1 {
			t.Fatalf("compiled[%d] closed %d times", i, raw.closes)
		}
	}
	if len(e.specializations) != 0 || len(b.executables) != 0 {
		t.Fatal("finalize retained executable")
	}
	if err := b.Close(); !errors.Is(err, newErr) {
		t.Fatalf("backend close lost error: %v", err)
	}
}

func TestDynamicExecuteConcurrentClose(t *testing.T) {
	for _, backendClose := range []bool{false, true} {
		name := "finalize"
		if backendClose {
			name = "backend-close"
		}
		t.Run(name, func(t *testing.T) {
			b, e := nativeDynamicIdentity(t)
			realCompile := b.compile
			entered, release := make(chan struct{}), make(chan struct{})
			var raw *trackedExecutable
			b.compile = func(code []byte, format string, opts *xla.CompileOptions) (nativeExecutable, error) {
				compiled, err := realCompile(code, format, opts)
				if err != nil {
					return nil, err
				}
				raw = &trackedExecutable{nativeExecutable: compiled, entered: entered, release: release}
				return raw, nil
			}
			input := dynamicInput(t, b, 2)
			executed := make(chan error, 1)
			go func() { _, err := e.Execute([]compute.Buffer{input}, nil, 0); executed <- err }()
			select {
			case <-entered:
			case err := <-executed:
				t.Fatalf("execution ended before native gate: %v", err)
			}
			if b.mu.TryLock() {
				b.mu.Unlock()
				close(release)
				t.Fatal("Execute released backend lock during native execution")
			}
			started, closed := make(chan struct{}), make(chan error, 1)
			go func() {
				close(started)
				if backendClose {
					closed <- b.Close()
				} else {
					e.Finalize()
					closed <- e.closeErr
				}
			}()
			<-started
			select {
			case err := <-closed:
				close(release)
				t.Fatalf("close returned during execution: %v", err)
			default:
			}
			close(release)
			if err := <-executed; err != nil {
				t.Fatal(err)
			}
			if err := <-closed; err != nil {
				t.Fatal(err)
			}
			if raw.closes != 1 || len(e.specializations) != 0 {
				t.Fatal("close did not clean specialization exactly once")
			}
			if _, err := e.Execute([]compute.Buffer{input}, nil, 0); err == nil {
				t.Fatal("execution succeeded after close")
			}
			e.Finalize()
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
			if raw.closes != 1 {
				t.Fatal("repeated cleanup closed specialization twice")
			}
		})
	}
}
