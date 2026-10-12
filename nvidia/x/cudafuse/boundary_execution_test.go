package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// sumExecutionProbe consumes emitted source and the runner's actual pointer
// lists. It performs host simulation only; it never calls a CUDA API.
type sumExecutionProbe struct {
	storage                            map[driver.DevicePtr][]float32
	allocs, launches, compiles, frees  int
	syncs, closes, releases            int
	failCompile, failAlloc, failLaunch int
	failSync, failFree                 bool
	pending                            bool
	sources                            []FusedKernel
	arguments                          [][]driver.DevicePtr
	counts                             []int
}

var sumProbeError = errors.New("injected sum execution failure")

func (p *sumExecutionProbe) ops() executionOps {
	return executionOps{
		alloc: func(n int) (*executionBuffer, error) {
			p.allocs++
			ptr := driver.DevicePtr(100000 + p.allocs*8192)
			p.storage[ptr] = make([]float32, n)
			b := &executionBuffer{ptr: ptr, read: func(out []float32) error {
				if p.pending {
					return fmt.Errorf("read before drain")
				}
				copy(out, p.storage[ptr])
				return nil
			}, free: func() error {
				if p.pending {
					return fmt.Errorf("free before drain")
				}
				if p.failFree {
					return sumProbeError
				}
				p.frees++
				delete(p.storage, ptr)
				return nil
			}}
			if p.allocs == p.failAlloc {
				return b, sumProbeError
			}
			return b, nil
		},
		compile: func(f FusedKernel) (*executionKernel, error) {
			p.compiles++
			p.sources = append(p.sources, f)
			k := &executionKernel{close: func() error { p.closes++; return nil }, launch: func(ptrs []driver.DevicePtr, count int) error {
				p.pending = true
				p.launches++
				p.arguments = append(p.arguments, slices.Clone(ptrs))
				p.counts = append(p.counts, count)
				if p.launches == p.failLaunch {
					return sumProbeError
				}
				if len(ptrs) != len(f.Inputs)+len(f.Outputs) {
					return fmt.Errorf("wrong pointer ABI")
				}
				var inputs [][]float32
				for _, ptr := range ptrs[:len(f.Inputs)] {
					inputs = append(inputs, p.storage[ptr])
				}
				out, err := sourceModel(f, inputs, count)
				if err != nil {
					return err
				}
				for i, ptr := range ptrs[len(f.Inputs):] {
					if len(p.storage[ptr]) != len(out[i]) {
						return fmt.Errorf("wrong output extent")
					}
					copy(p.storage[ptr], out[i])
				}
				return nil
			}}
			if p.compiles == p.failCompile {
				return k, sumProbeError
			}
			return k, nil
		},
		sync: func(ctx context.Context) error {
			p.syncs++
			if ctx != nil && ctx.Err() != nil {
				return ctx.Err()
			}
			if p.failSync {
				return sumProbeError
			}
			p.pending = false
			return nil
		},
	}
}

func prepareSumProbe(t *testing.T, root *LazyNode, eager bool, data map[*LazyNode][]float32) (*executionPlan, *sumExecutionProbe, map[*LazyNode]driver.DevicePtr) {
	t.Helper()
	p, err := CompileDeferredGraph(nil, root)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := snapshotExecution(p, eager)
	if err != nil {
		t.Fatal(err)
	}
	probe := &sumExecutionProbe{storage: make(map[driver.DevicePtr][]float32)}
	bound := make(map[*LazyNode]driver.DevicePtr)
	for n, values := range data {
		ptr := driver.DevicePtr(n.ID * 8192)
		probe.storage[ptr] = slices.Clone(values)
		bound[plan.originals[n]] = ptr
	}
	return plan, probe, bound
}

func TestSumExecutionMultiRegionSourceAndPointers(t *testing.T) {
	for _, eager := range []bool{false, true} {
		t.Run(fmt.Sprint(eager), func(t *testing.T) {
			ctx := NewLazyContext()
			x := NewLazyArray(ctx, []int{3}, cudair.Float32)
			y := NewLazyArray(ctx, []int{3}, cudair.Float32)
			z := NewLazyArray(ctx, []int{1}, cudair.Float32)
			expr := x.Add(y).Map("neg")
			sum := sumNode(ctx, expr.Node)
			root := &LazyNode{ID: ctx.newID(), Op: "add", Inputs: []*LazyNode{sum, z.Node}, Shape: []int{1}, DType: cudair.Float32}
			plan, probe, bound := prepareSumProbe(t, root, eager, map[*LazyNode][]float32{x.Node: {16777216, 1, -16777216}, y.Node: {2, 3, 4}, z.Node: {7}})
			// Mutating caller metadata after snapshot cannot change execution.
			root.Shape[0] = 99
			sum.Attrs["axis"] = 7
			x.Node.Shape[0] = 0
			e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), []func(){func() { probe.releases++ }})
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Readback(context.Background())
			if err != nil || !slices.Equal(got, []float32{-3}) || !slices.Equal(e.Shape(), []int{1}) {
				t.Fatal(got, err, e.Shape())
			}
			want := 2
			if eager {
				want = 4
			}
			if e.Stats() != (ExecutionStats{Allocations: want, Launches: want}) || probe.syncs != 1 {
				t.Fatal(e.Stats(), probe)
			}
			last := probe.arguments[len(probe.arguments)-1]
			previous := probe.arguments[len(probe.arguments)-2]
			if last[0] != previous[len(previous)-1] || last[1] != bound[plan.originals[z.Node]] || probe.counts[len(probe.counts)-1] != 1 {
				t.Fatal("boundary result or external operand pointer changed")
			}
			shape := e.Shape()
			shape[0] = 100
			if e.Shape()[0] != 1 {
				t.Fatal("shape aliases execution")
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if probe.frees != want || probe.closes != want || probe.releases != 1 {
				t.Fatal("owned temporary or module leak")
			}
		})
	}
}

func TestSumExecutionSecondaryOutputsAndSharedProducer(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{3}, cudair.Float32)
	y := NewLazyArray(ctx, []int{3}, cudair.Float32)
	a := x.Add(y)
	b := a.Mul(y)
	s1, s2 := sumNode(ctx, b.Node), sumNode(ctx, a.Node)
	root := &LazyNode{ID: ctx.newID(), Op: "add", Inputs: []*LazyNode{s1, s2}, Shape: []int{1}, DType: cudair.Float32}
	plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{x.Node: {1, 2, 3}, y.Node: {2, 3, 4}})
	e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(context.Background())
	if err != nil || !slices.Equal(got, []float32{64}) || len(probe.sources[0].Outputs) != 2 || e.Stats().Allocations != 5 || e.Stats().Launches != 4 {
		t.Fatal(got, err, e.Stats(), probe.sources)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSumExecutionEmptyWritesAndBindingDomains(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{0}, cudair.Float32)
	p, err := x.Sum()
	if err != nil {
		t.Fatal(err)
	}
	plan, err := snapshotExecution(p, false)
	if err != nil {
		t.Fatal(err)
	}
	probe := &sumExecutionProbe{storage: make(map[driver.DevicePtr][]float32)}
	e, err := runExecutionPlan(context.Background(), plan, nil, probe.ops(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(context.Background())
	if err != nil || !slices.Equal(got, []float32{0}) || e.Stats().Allocations != 1 || e.Stats().Launches != 1 {
		t.Fatal(got, err, e.Stats())
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	dev := &cuda.Device{}
	if err := validateExecutionBindings(plan.nodes, 1, dev, map[string]executionBindingDescription{x.Node.Name(): {shape: []int{0}, dtype: cudair.Float32}}); err != nil {
		t.Fatal("empty leaf incorrectly required scalar root storage", err)
	}
	if result, err := p.Execute(context.Background(), nil, map[string]ExecutionBinding{x.Node.Name(): {Shape: []int{0}, DType: cudair.Float32}}); err == nil || result != nil {
		t.Fatal("empty sum silently skipped its required real device write")
	}
}

func TestSumExecutionFailureDrainAndPartialOwners(t *testing.T) {
	for _, failure := range []string{"compile2", "allocation2", "launch2", "sync", "free"} {
		t.Run(failure, func(t *testing.T) {
			ctx := NewLazyContext()
			x := NewLazyArray(ctx, []int{3}, cudair.Float32)
			sum := sumNode(ctx, x.Node)
			root := &LazyNode{ID: ctx.newID(), Op: "neg", Inputs: []*LazyNode{sum}, Shape: []int{1}, DType: cudair.Float32}
			plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{x.Node: {1, 2, 3}})
			switch failure {
			case "compile2":
				probe.failCompile = 2
			case "allocation2":
				probe.failAlloc = 2
			case "launch2":
				probe.failLaunch = 2
			case "sync":
				probe.failSync = true
			case "free":
				probe.failFree = true
			}
			e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), []func(){func() { probe.releases++ }})
			if failure == "free" {
				err = e.Close()
			}
			if !errors.Is(err, sumProbeError) || probe.frees != 0 || probe.releases != 0 {
				t.Fatal(err, probe)
			}
			probe.failSync = true
			if failure != "free" {
				if err := e.Close(); !errors.Is(err, sumProbeError) || probe.frees != 0 || probe.releases != 0 {
					t.Fatal("failed drain discarded owners", err)
				}
			}
			probe.failSync, probe.failFree = false, false
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if probe.frees != probe.allocs || probe.closes != probe.compiles || probe.releases != 1 {
				t.Fatal("partial owners did not retire once", probe)
			}
			if err := e.Close(); err != nil || probe.releases != 1 {
				t.Fatal("close was not idempotent", err)
			}
		})
	}
}

func TestSumExecutionRefusesUnsupportedContracts(t *testing.T) {
	for _, mode := range []string{"axis", "rank", "output", "broadcast", "filter", "cycle", "region", "name"} {
		t.Run(mode, func(t *testing.T) {
			ctx := NewLazyContext()
			x := NewLazyArray(ctx, []int{3}, cudair.Float32)
			root := sumNode(ctx, x.Node)
			p, err := CompileDeferredGraph(ctx, root)
			if err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "axis":
				root.Attrs["axis"] = 1
			case "rank":
				x.Node.Shape = []int{1, 3}
			case "output":
				root.Shape = []int{}
			case "broadcast":
				root.Op, root.Attrs = "add", nil
				root.Inputs = []*LazyNode{x.Node, NewLazyArray(ctx, []int{1}, cudair.Float32).Node}
			case "filter":
				root.Op = "filter"
			case "cycle":
				root.Inputs[0] = root
			case "region":
				p.Regions = p.Regions[:1]
			case "name":
				root.ID = -1
			}
			if _, err := snapshotExecution(p, false); err == nil {
				t.Fatal("invalid contract admitted by snapshot before native calls")
			}
			if e, err := p.Execute(context.Background(), nil, nil); err == nil || e != nil {
				t.Fatal("unsupported contract reached device path")
			}
		})
	}
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{3}, cudair.Float32)
	x.Node.Inputs = []*LazyNode{x.Node}
	if _, err := x.Materialize(); err == nil || !strings.Contains(err.Error(), "cyclic") {
		t.Fatal("deferred compile accepted cycle", err)
	}
}

func TestSumExecutionFiveOperationChainHasOneOwnedResult(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{1003}, cudair.Float32)
	y := NewLazyArray(ctx, []int{1003}, cudair.Float32)
	expr := x.Add(y).Mul(y).Map("neg").Add(x)
	if ctx.Tracker.Snapshot().Allocations != 0 {
		t.Fatal("lazy chain allocated before its boundary")
	}
	root := sumNode(ctx, expr.Node)
	a, b := make([]float32, 1003), make([]float32, 1003)
	var want float64
	for i := range a {
		a[i], b[i] = float32(i%17-8)/4, float32(i%11-5)/8
		v := a[i] + b[i]
		v *= b[i]
		v = -v
		v += a[i]
		want += float64(v)
	}
	plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{x.Node: a, y.Node: b})
	e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(context.Background())
	if err != nil || !slices.Equal(got, []float32{float32(want)}) || e.Stats() != (ExecutionStats{Allocations: 1, Launches: 1}) {
		t.Fatal(got, err, want, e.Stats())
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}

func ExampleExecution_Shape() {
	// Shape is safe to inspect on a nil owner. Actual results require Execute
	// on a live stream; this example performs no device work.
	var execution *Execution
	fmt.Println(execution.Shape())
	// Output: []
}

func TestSumExecutionIndexedTransposeBeforeReduction(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	y := NewLazyArray(ctx, []int{3, 2}, cudair.Float32)
	expr := x.Transpose([]int{1, 0}).Mul(y).Reshape([]int{6})
	root := sumNode(ctx, expr.Node)
	plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{x.Node: {1, 2, 3, 4, 5, 6}, y.Node: {1, 2, 3, 4, 5, 6}})
	// The snapshot owns permutation storage as well as shapes.
	transpose := expr.Node.Inputs[0].Inputs[0]
	transpose.Attrs["permutation"].([]int)[0] = 0
	// The independent row-major transpose is [1,4,2,5,3,6].
	want := float32(1 + 8 + 6 + 20 + 15 + 36)
	e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(context.Background())
	if err != nil || !slices.Equal(got, []float32{want}) || e.Stats().Allocations != 1 {
		t.Fatal(got, err, e.Stats())
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSumExecutionPreservesScalarAndRankNPointwise(t *testing.T) {
	for _, scalar := range []bool{false, true} {
		t.Run(fmt.Sprint(scalar), func(t *testing.T) {
			ctx := NewLazyContext()
			shape := []int{2, 3}
			if scalar {
				shape = []int{}
			}
			x := NewLazyArray(ctx, shape, cudair.Float32)
			var y, root *LazyArray
			var data map[*LazyNode][]float32
			var want []float32
			if scalar {
				y = NewLazyArray(ctx, shape, cudair.Float32)
				root = x.Add(y)
				data = map[*LazyNode][]float32{x.Node: {3}, y.Node: {4}}
				want = []float32{7}
			} else {
				y = NewLazyArray(ctx, []int{3, 2}, cudair.Float32)
				root = x.Transpose([]int{1, 0}).Mul(y)
				data = map[*LazyNode][]float32{x.Node: {1, 2, 3, 4, 5, 6}, y.Node: {1, 2, 3, 4, 5, 6}}
				want = []float32{1, 8, 6, 20, 15, 36}
			}
			plan, probe, bound := prepareSumProbe(t, root.Node, false, data)
			e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Readback(context.Background())
			if err != nil || !slices.Equal(got, want) || !slices.Equal(e.Shape(), root.Shape) || e.Stats().Allocations != 1 {
				t.Fatal(got, err, e.Shape())
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestSumExecutionChecksOwnedIntermediatesAfterCompletion(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{3}, cudair.Float32)
	sum := sumNode(ctx, x.Node)
	root := &LazyNode{ID: ctx.newID(), Op: "neg", Inputs: []*LazyNode{sum}, Shape: []int{1}, DType: cudair.Float32}
	plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{x.Node: {1, 2, 3}})
	ops := probe.ops()
	allocate := ops.alloc
	ops.alloc = func(n int) (*executionBuffer, error) {
		b, err := allocate(n)
		if probe.allocs == 1 {
			b.check = func() error {
				if probe.pending {
					return fmt.Errorf("intermediate check before completion")
				}
				return ErrOutputGuard
			}
		}
		return b, err
	}
	e, err := runExecutionPlan(context.Background(), plan, bound, ops, nil)
	if !errors.Is(err, ErrOutputGuard) {
		t.Fatal("corrupt intermediate did not fail execution", err)
	}
	if out, err := e.Readback(context.Background()); !errors.Is(err, ErrOutputGuard) || out != nil {
		t.Fatal("corrupt intermediate exposed plausible final result", out, err)
	}
	if err := e.Close(); err != nil || probe.frees != 2 || probe.syncs != 1 {
		t.Fatal("completed ownership did not retire", err, probe)
	}
}

func TestSumExecutionSharedInputDistinctIndexMappings(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{2, 2}, cudair.Float32)
	root := x.Transpose([]int{1, 0}).Add(x)
	plan, probe, bound := prepareSumProbe(t, root.Node, false, map[*LazyNode][]float32{x.Node: {1, 2, 3, 4}})
	e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(context.Background())
	if err != nil || !slices.Equal(got, []float32{2, 5, 5, 8}) {
		t.Fatal("shared input index mappings were conflated", got, err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSumExecutionFailedLaunchPreservesOriginalError(t *testing.T) {
	ctx := NewLazyContext()
	x := NewLazyArray(ctx, []int{3}, cudair.Float32)
	sum := sumNode(ctx, x.Node)
	root := &LazyNode{ID: ctx.newID(), Op: "neg", Inputs: []*LazyNode{sum}, Shape: []int{1}, DType: cudair.Float32}
	plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{x.Node: {1, 2, 3}})
	probe.failLaunch = 2
	ops := probe.ops()
	allocate := ops.alloc
	ops.alloc = func(n int) (*executionBuffer, error) {
		b, err := allocate(n)
		b.check = func() error { return ErrOutputUnwritten }
		return b, err
	}
	e, err := runExecutionPlan(context.Background(), plan, bound, ops, nil)
	if !errors.Is(err, sumProbeError) {
		t.Fatal(err)
	}
	if out, err := e.Readback(context.Background()); out != nil || !errors.Is(err, sumProbeError) || errors.Is(err, ErrOutputUnwritten) {
		t.Fatal("failed launch replaced by meaningless poison evidence", out, err)
	}
	if err := e.Close(); err != nil || probe.frees != 2 {
		t.Fatal(err, probe)
	}
}

func TestSumExecutionPublicEntryRequiresRealStream(t *testing.T) {
	x := NewLazyArray(NewLazyContext(), []int{3}, cudair.Float32)
	p, err := x.Sum()
	if err != nil {
		t.Fatal(err)
	}
	e, err := p.Execute(context.Background(), nil, nil)
	if e != nil || err == nil || !strings.Contains(err.Error(), "nil execution stream or device") {
		t.Fatal("public entry did not admit sum before requiring actual device resources", e, err)
	}
}
