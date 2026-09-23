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
