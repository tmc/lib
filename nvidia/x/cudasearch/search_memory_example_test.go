package cudasearch_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleClassifyBottleneck() {
	memory, compute, occupancy := 90.0, 20.0, 70.0
	state := cudasearch.ClassifyBottleneck(cudasearch.ProfileMetrics{MemoryThroughput: &memory, ComputeThroughput: &compute, WarpOccupancy: &occupancy})
	fmt.Println(state.StateName, state.PendingHardware)
	// Output: memory_bandwidth_limited true
}

func ExampleBottleneckMetrics() {
	fmt.Println(len(cudasearch.BottleneckMetrics()))
	// Output: 3
}

func ExampleCollectBottleneckProfile() {
	p, _, err := cudasearch.CollectBottleneckProfile(context.Background(), nil, cudasearch.BottleneckProfile{}, cuptibench.Workload{})
	fmt.Println(p.State.PendingHardware, errors.Is(err, cudatrial.ErrUnqualifiedOutcome))
	// Output: true true
}

func ExampleBottleneckProfile_HardwareObserved() {
	var p cudasearch.BottleneckProfile
	fmt.Println(p.HardwareObserved())
	// Output: false
}

func ExampleObserveCandidateTrajectory() {
	_, err := cudasearch.ObserveCandidateTrajectory(nil, cudatrial.TrajectoryReceipt{}, nil)
	fmt.Println(errors.Is(err, cudatrial.ErrUnqualifiedOutcome))
	// Output: true
}

func ExampleSearchMemory() {
	var catalog cudacache.StrategyKnowledgeBase
	catalog.Update(cudacache.Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Technique: "shared_memory_tiling", Confidence: 1, PredictedSpeedup: 1})
	memory := &cudasearch.SearchMemory{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Knowledge: &catalog, RuleTechniques: map[string]string{"tile": "shared_memory_tiling"}}
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "base"})
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	rules := []*egraph.Rewrite{egraph.MustRewrite("tile", "base", "optimized")}
	budget := cudasearch.DefaultBudget()
	budget.MaxStates = 2
	budget.MaxEvaluations = 2
	// MockEvaluator explores syntax only; it is not a native timing source.
	r, err := cudasearch.NewBeamSearch(cudasearch.BeamConfig{Budget: budget, Memory: memory}).Search(context.Background(), state, rules, &cudasearch.MockEvaluator{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Evaluations, r.FrozenMemory().Weights["tile"], r.FrozenMemory().State.PendingHardware)
	// Output: 2 1 true
}

func ExampleSearchResult_FrozenMemory() {
	result := new(cudasearch.SearchResult)
	fmt.Println(result.FrozenMemory() == nil)
	// Output: true
}
