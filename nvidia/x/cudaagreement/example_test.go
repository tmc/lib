package cudaagreement_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// The records below stand in for what two executors report after a run; on a
// device they come from Execute. The graph arm fell back on one step, so the
// gate fails even though the two arms left identical bytes behind.
func ExampleVerify() {
	out := []byte{0xde, 0xad, 0xbe, 0xef}
	launch := cudaagreement.Arm{
		Record: &cudaagreement.Record{
			Arm: "intchain/launch", Requested: cudaplan.BackendLaunch,
			Steps: []cudaagreement.StepRecord{
				{Step: "mul", Requested: cudaplan.BackendLaunch, Executed: cudaplan.BackendLaunch, Digest: "d1", Grid: 4, Block: 256},
				{Step: "add", Requested: cudaplan.BackendLaunch, Executed: cudaplan.BackendLaunch, Digest: "d2", Grid: 4, Block: 256},
			},
		},
		Output: out,
	}
	graph := cudaagreement.Arm{
		Record: &cudaagreement.Record{
			Arm: "intchain/graph", Requested: cudaplan.BackendGraph,
			Steps: []cudaagreement.StepRecord{
				{Step: "mul", Requested: cudaplan.BackendGraph, Executed: cudaplan.BackendGraph, Digest: "d1", Grid: 4, Block: 256},
				{Step: "add", Requested: cudaplan.BackendGraph, Executed: cudaplan.BackendLaunch, Declined: "graph refused the node", Digest: "d2", Grid: 4, Block: 256},
			},
		},
		Output: out,
	}
	arms := []cudaagreement.Arm{launch, graph}
	fmt.Println("outputs:", cudaagreement.Agree(arms))
	fmt.Println("gate:", cudaagreement.Verify(arms))
	// Output:
	// outputs: <nil>
	// gate: cudaagreement: arm intchain/graph did not execute on Graph: add ran on Launch (graph refused the node)
}

// A GEMM, a chain kernel that overlaps its tail, an idle gap, and a second
// chain kernel. Every nanosecond from the first start to the last end is in
// exactly one part.
func ExampleAccount() {
	a, err := cudaagreement.Account([]cudaagreement.Kernel{
		{Name: "gemm", Start: 0, End: 700},
		{Name: "rmsnorm", Start: 650, End: 800, Chain: true},
		{Name: "silu_mul", Start: 900, End: 1000, Chain: true},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a)
	f, err := a.Fractions()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("chain launches %.3f, chain span %.3f\n", f.ChainLaunches, f.ChainSpan)
	_, err = cudaagreement.SpanAccount{}.Fractions()
	fmt.Println(err)
	// Output:
	// launches chain=2 other=1 total=3; span_ns chain=200 other=650 overlap=50 idle=100 total=1000
	// chain launches 0.667, chain span 0.200
	// cudaagreement: region has no kernel launches; no fraction has a denominator
}
