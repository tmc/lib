package cudamega_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func ExamplePolicyFor() {
	chain, _ := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "load", Rank: 0, Scope: cudataskgraph.ScopeDevice},
		{ID: "scale", Deps: []string{"load"}, Rank: 1, Scope: cudataskgraph.ScopeDevice},
	})
	heads, _ := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "head0", Rank: 0, Scope: cudataskgraph.ScopeDevice},
		{ID: "head1", Rank: 0, Scope: cudataskgraph.ScopeDevice},
	})
	for _, g := range []cudataskgraph.Graph{chain, heads, {}} {
		p, err := cudamega.PolicyFor(g)
		if err != nil {
			fmt.Println("refused:", err)
			continue
		}
		fmt.Println(p)
	}
	// Output:
	// wave
	// counted
	// refused: graph has no tasks to choose a policy for
}

func ExampleAdmissionFromOccupancy() {
	a, err := cudamega.AdmissionFromOccupancy(cudamega.Occupancy{
		BlocksPerMultiprocessor: 2,
		Multiprocessors:         48,
		WorkerThreads:           128,
		SharedBytes:             1024,
	})
	fmt.Println(a.ResidentCTAs, err)

	_, err = cudamega.AdmissionFromOccupancy(cudamega.Occupancy{Multiprocessors: 48, WorkerThreads: 128, SharedBytes: 1024})
	fmt.Println(err)
	// Output:
	// 96 <nil>
	// occupancy blocks per multiprocessor is zero: unmeasured, or the worker does not fit
}
