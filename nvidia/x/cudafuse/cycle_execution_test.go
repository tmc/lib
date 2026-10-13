package cudafuse

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudair"
	"slices"
	"testing"
)

func TestCycleExecutionOwnedFanout(t *testing.T) {
	for _, eager := range []bool{false, true} {
		t.Run(fmt.Sprint(eager), func(t *testing.T) {
			ctx := NewLazyContext()
			a := NewLazyArray(ctx, []int{3}, cudair.Float32)
			scalar := NewLazyArray(ctx, nil, cudair.Float32)
			cycle := a.Cycle([]int{6})
			root := cycle.Add(scalar).Add(cycle)
			e, probe, plan := runShapeProbe(t, root.Node, map[*LazyNode][]float32{a.Node: {2, 5, 9}, scalar.Node: {10}}, eager)
			got, err := e.Readback(context.Background())
			if err != nil || !slices.Equal(got, []float32{14, 20, 28, 14, 20, 28}) {
				t.Fatal(got, err)
			}
			t.Logf("host Cycle eager=%t values=%v step-counts=%v source=%x", eager, got, probe.counts, sha256.Sum256([]byte(probe.sources[0].Source)))
			var owned bool
			for i, step := range plan.steps {
				if slices.Contains(step.outputs, plan.originals[cycle.Node]) {
					owned = true
					if step.count != 6 || probe.counts[i] != 6 {
						t.Fatal("Cycle output extent")
					}
				}
			}
			if !owned || probe.allocs < 2 {
				t.Fatal("Cycle boundary storage erased")
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if probe.frees != probe.allocs || probe.releases != 1 {
				t.Fatal("Cycle ownership leak")
			}
		})
	}
}
func TestCycleExecutionTransposedSource(t *testing.T) {
	for _, eager := range []bool{false, true} {
		ctx := NewLazyContext()
		a := NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
		root := a.Transpose([]int{1, 0}).Cycle([]int{8})
		e, probe, _ := runShapeProbe(t, root.Node, map[*LazyNode][]float32{a.Node: {0, 1, 2, 3, 4, 5}}, eager)
		got, err := e.Readback(context.Background())
		if err != nil || !slices.Equal(got, []float32{0, 3, 1, 4, 2, 5, 0, 3}) {
			t.Fatal(got, err)
		}
		if err := e.Close(); err != nil || probe.frees != probe.allocs {
			t.Fatal(err)
		}
	}
}
func TestCycleExecutionFailureDrain(t *testing.T) {
	for _, failure := range []string{"compile2", "allocation2", "launch2", "sync", "free"} {
		t.Run(failure, func(t *testing.T) {
			ctx := NewLazyContext()
			a := NewLazyArray(ctx, []int{3}, cudair.Float32)
			root := a.Cycle([]int{6}).Map("neg")
			plan, probe, bound := prepareSumProbe(t, root.Node, false, map[*LazyNode][]float32{a.Node: {1, 2, 3}})
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
				t.Fatal("discarded attempted owners", err, probe.frees, probe.releases)
			}
			probe.failSync = true
			if err := e.Close(); !errors.Is(err, sumProbeError) || probe.frees != 0 {
				t.Fatal("failed drain lost storage", err)
			}
			probe.failSync = false
			probe.failFree = false
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if probe.frees != probe.allocs || probe.closes != probe.compiles || probe.releases != 1 {
				t.Fatal("leaked partial owners")
			}
		})
	}
}
