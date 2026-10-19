package cudadiag_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleAnalyzeTrainingJSONL() {
	// Constructed application logs and operator/activity bytes; no authentic
	// provider, native run, independent oracle or convergence is qualified.
	input, err := trainingSummaryFixture(nil)
	if err != nil {
		panic(err)
	}
	out, err := cudadiag.AnalyzeTrainingJSONL(context.Background(), input, nvtelemetry.WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	fmt.Println(len(out.Details.Phases), out.Recovery.Updates.Accepted, out.Quality)
	fmt.Println(out.Operators.KernelPopulation.Exact, out.Operators.MemberEdgePopulation.Associated, out.Edges[0].Outcome)
	// Output:
	// 8 2 accepted
	// 1 2 associated
}
