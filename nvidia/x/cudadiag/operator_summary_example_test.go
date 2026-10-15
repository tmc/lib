package cudadiag_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleAnalyzeOperatorJSONL() {
	// Constructed raw records and separately retained member inventory.
	// Counts below demonstrate host composition, not native/framework proof.
	activity, supplied, source := operatorSummaryInputs()
	var metadata bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&metadata, supplied, nvtelemetry.WorkloadLimits{}); err != nil {
		panic(err)
	}
	summary, err := cudadiag.AnalyzeOperatorJSONL(context.Background(), bytes.NewReader(activity), &metadata, source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	fmt.Println(summary.KernelPopulation.Total, summary.MemberEdgePopulation.Associated, summary.MemberEdgePopulation.Exact)
	// Output: 1 2 0
}
