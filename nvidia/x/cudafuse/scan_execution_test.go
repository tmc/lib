package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestScanPublicAdmissionAndEmpty(t *testing.T) {
	for _, n := range []int{0, 3} {
		ctx := NewLazyContext()
		a := NewLazyArray(ctx, []int{n}, cudair.Float32)
		plan, err := a.Scan("sum")
		if err != nil {
			t.Fatal(err)
		}
		if _, err = snapshotExecution(plan, false); err != nil {
			t.Fatal(err)
		}
		e, err := plan.Execute(context.Background(), nil, map[string]ExecutionBinding{a.Node.Name(): {Shape: []int{n}, DType: cudair.Float32}})
		if n == 0 {
			if err != nil || e == nil || !slices.Equal(e.Shape(), []int{0}) {
				t.Fatal(e, err)
			}
			got, err := e.Readback(context.Background())
			if err != nil || len(got) != 0 || e.Stats() != (ExecutionStats{}) {
				t.Fatal(got, err)
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
		} else if err == nil || e != nil {
			t.Fatal("nonempty scan fabricated device output")
		}
	}
}

func TestScanExecutionSourceSchedule(t *testing.T) {
	for _, eager := range []bool{false, true} {
		t.Run(fmt.Sprint(eager), func(t *testing.T) {
			ctx := NewLazyContext()
			a := NewLazyArray(ctx, []int{3}, cudair.Float32)
			b := NewLazyArray(ctx, []int{3}, cudair.Float32)
			expr := a.Add(b).Map("neg").Mul(b).Add(b).Map("neg")
			root := scanNode(ctx, expr.Node)
			data := map[*LazyNode][]float32{a.Node: {1, 2, 3}, b.Node: {2, 3, 4}}
			plan, probe, bound := prepareSumProbe(t, root, eager, data)
			e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), []func(){func() { probe.releases++ }})
			if err != nil {
				t.Fatal(err)
			}
			got, err := e.Readback(context.Background())
			want := ordinaryScan([]float32{4, 12, 24})
			if err != nil || !sameScan(got, want) || !slices.Equal(e.Shape(), []int{3}) {
				t.Fatal(got, err, want)
			}
			allocs := 1
			if eager {
				allocs = 6
			}
			if e.Stats().Allocations != allocs || e.Stats().Launches != allocs || probe.counts[len(probe.counts)-1] != 3 {
				t.Fatal(e.Stats(), probe.counts)
			}
			root.Shape[0] = 99
			shape := e.Shape()
			shape[0] = 44
			if e.Shape()[0] != 3 {
				t.Fatal("shape alias")
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if probe.frees != allocs || probe.releases != 1 {
				t.Fatal("owned result or leaf release leak")
			}
		})
	}
}

func TestScanExecutionSharedProducerAndSum(t *testing.T) {
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{3}, cudair.Float32)
	b := NewLazyArray(ctx, []int{3}, cudair.Float32)
	shared := a.Add(b)
	scan := scanNode(ctx, shared.Node)
	root := &LazyNode{ID: ctx.newID(), Op: "add", Inputs: []*LazyNode{scan, shared.Node}, Shape: []int{3}, DType: cudair.Float32}
	plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{a.Node: {1, 2, 3}, b.Node: {2, 3, 4}})
	e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.Readback(context.Background())
	if err != nil || !sameScan(got, []float32{6, 13, 22}) || e.Stats().Allocations != 3 {
		t.Fatal(got, err, e.Stats())
	}
	if probe.arguments[1][0] != probe.arguments[0][len(probe.arguments[0])-1] || probe.arguments[2][0] != probe.arguments[1][len(probe.arguments[1])-1] || probe.arguments[2][1] != probe.arguments[0][len(probe.arguments[0])-1] {
		t.Fatal("shared pointer schedule")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	ctx = NewLazyContext()
	a = NewLazyArray(ctx, []int{3}, cudair.Float32)
	scan = scanNode(ctx, a.Node)
	sum := sumNode(ctx, scan)
	plan, probe, bound = prepareSumProbe(t, sum, false, map[*LazyNode][]float32{a.Node: {16777216, 1, -16777216}})
	e, err = runExecutionPlan(context.Background(), plan, bound, probe.ops(), nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err = e.Readback(context.Background())
	if err != nil || !sameScan(got, []float32{33554432}) || e.Stats().Allocations != 2 {
		t.Fatal(got, err, e.Stats())
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestScanExecutionDrainAndPartialOwners(t *testing.T) {
	for _, failure := range []string{"allocation", "launch", "sync"} {
		t.Run(failure, func(t *testing.T) {
			ctx := NewLazyContext()
			a := NewLazyArray(ctx, []int{3}, cudair.Float32)
			root := scanNode(ctx, a.Node)
			plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{a.Node: {1, 2, 3}})
			switch failure {
			case "allocation":
				probe.failAlloc = 1
			case "launch":
				probe.failLaunch = 1
			case "sync":
				probe.failSync = true
			}
			e, err := runExecutionPlan(context.Background(), plan, bound, probe.ops(), []func(){func() { probe.releases++ }})
			if err == nil || e == nil {
				t.Fatal("failure lost")
			}
			if failure != "allocation" {
				if probe.frees != 0 || probe.releases != 0 {
					t.Fatal("freed before drain")
				}
				probe.failSync = true
				if err := e.Close(); err == nil || probe.frees != 0 {
					t.Fatal("failed drain freed owner")
				}
				probe.failSync = false
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if probe.frees != 1 || probe.releases != 1 {
				t.Fatal("owner leak or duplicate release")
			}
			if err := e.Close(); err != nil || probe.frees != 1 {
				t.Fatal("nonidempotent close")
			}
		})
	}
}

func TestScanExecutionRefusals(t *testing.T) {
	for _, op := range []string{"mul", "exclusive_sum"} {
		ctx := NewLazyContext()
		a := NewLazyArray(ctx, []int{3}, cudair.Float32)
		p, err := a.Scan(op)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = snapshotExecution(p, false); err == nil {
			t.Fatal("unsupported scan admitted", op)
		}
	}
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	p, err := a.Scan("sum")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = snapshotExecution(p, false); err == nil {
		t.Fatal("rank2 scan admitted")
	}
}

func ExampleLazyArray_Scan() {
	ctx := NewLazyContext()
	input := NewLazyArray(ctx, []int{0}, cudair.Float32)
	plan, err := input.Scan("sum")
	if err != nil {
		fmt.Println(err)
		return
	}
	execution, err := plan.Execute(context.Background(), nil, map[string]ExecutionBinding{input.Node.Name(): {Shape: []int{0}, DType: cudair.Float32}})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer execution.Close()
	values, err := execution.Readback(context.Background())
	fmt.Println(execution.Shape(), len(values), err, execution.Stats())
	// Output: [0] 0 <nil> {0 0 0}
}

func TestScanCanceledWaitAndPoisonRestriction(t *testing.T) {
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{3}, cudair.Float32)
	root := scanNode(ctx, a.Node)
	plan, probe, bound := prepareSumProbe(t, root, false, map[*LazyNode][]float32{a.Node: {1, 2, 3}})
	wait, cancel := context.WithCancel(context.Background())
	ops := probe.ops()
	sync := ops.sync
	ops.sync = func(ctx context.Context) error { cancel(); return sync(ctx) }
	e, err := runExecutionPlan(wait, plan, bound, ops, []func(){func() { probe.releases++ }})
	if err == nil || e == nil || probe.frees != 0 || probe.releases != 0 {
		t.Fatal("canceled wait lost ownership", err)
	}
	probe.failSync = true
	if err := e.Close(); err == nil || probe.frees != 0 {
		t.Fatal("failed close drain released owner")
	}
	probe.failSync = false
	if err := e.Close(); err != nil || probe.frees != 1 || probe.releases != 1 {
		t.Fatal("retry cleanup", err)
	}
	ctx = NewLazyContext()
	a = NewLazyArray(ctx, []int{1}, cudair.Float32)
	root = scanNode(ctx, a.Node)
	poison := math.Float32frombits(0x7fc01234)
	plan, probe, bound = prepareSumProbe(t, root, false, map[*LazyNode][]float32{a.Node: {poison}})
	ops = probe.ops()
	allocate := ops.alloc
	ops.alloc = func(n int) (*executionBuffer, error) {
		b, err := allocate(n)
		if b != nil {
			b.check = func() error {
				all := append([]float32{poison}, probe.storage[b.ptr]...)
				all = append(all, poison)
				return checkExecutionOutput(all, n)
			}
		}
		return b, err
	}
	e, err = runExecutionPlan(context.Background(), plan, bound, ops, nil)
	if !errors.Is(err, ErrOutputUnwritten) {
		t.Fatal("reserved NaN payload silently admitted", err)
	}
	if _, err := e.Readback(context.Background()); !errors.Is(err, ErrOutputUnwritten) {
		t.Fatal("poison result exposed", err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}
