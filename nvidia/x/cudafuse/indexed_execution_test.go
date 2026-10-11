package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// This probe checks host ownership and dispatch bookkeeping. Emitted-source
// numerical controls are separate; the probe does not execute device code.
func TestIndexedExecutionOwnership(t *testing.T) {
	for _, eager := range []bool{false, true} {
		for _, canceled := range []bool{false, true} {
			t.Run(fmt.Sprintf("eager=%v/canceled=%v", eager, canceled), func(t *testing.T) {
				ctx := NewLazyContext()
				x := NewLazyArray(ctx, []int{17, 59}, cudair.Float32)
				y := NewLazyArray(ctx, []int{59, 17}, cudair.Float32)
				root := x.Transpose([]int{1, 0}).Map("neg").Add(y)
				plan, err := root.Materialize()
				if err != nil {
					t.Fatal(err)
				}
				nodes, count, err := executionNodes(plan)
				if err != nil {
					t.Fatal(err)
				}
				probe := &executionProbe{cancel: canceled}
				runctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				result, err := runExecution(runctx, nodes, count, executionPointers(nodes), eager, probe.ops(), []func(){func() { probe.released++ }})
				want := 1
				if eager {
					want = 3
				}
				if probe.allocs != want || probe.launches != want || probe.syncs != 1 || result.Stats() != (ExecutionStats{Allocations: want, Launches: want}) {
					t.Fatalf("indexed host dispatch differs: %+v, %+v", probe, result.Stats())
				}
				if canceled {
					if !errors.Is(err, context.Canceled) || probe.frees != 0 || probe.released != 0 {
						t.Fatal("cancellation released unfinished indexed work", err)
					}
					failure := errors.New("indexed drain failed")
					probe.syncErr = failure
					if err := result.Close(); !errors.Is(err, failure) {
						t.Fatal(err)
					}
					if probe.frees != 0 || probe.released != 0 {
						t.Fatal("failed drain released ownership")
					}
					probe.syncErr = nil
				} else if err != nil {
					t.Fatal(err)
				}
				if err := result.Close(); err != nil {
					t.Fatal(err)
				}
				if err := result.Close(); err != nil {
					t.Fatal(err)
				}
				if probe.frees != want || probe.closed != want || probe.released != 1 || result.Stats().Frees != want {
					t.Fatal("indexed ownership did not retire exactly once")
				}
			})
		}
	}
}
