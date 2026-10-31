package cudadiag_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleAnalyzeOperatorCPUJSON() {
	c, source := cpuSummaryInputs()
	var wire bytes.Buffer
	_ = nvtelemetry.WriteOperatorCPUContext(&wire, c, nvtelemetry.WorkloadLimits{})
	out, err := cudadiag.AnalyzeOperatorCPUJSON(context.Background(), &wire, source, nvtelemetry.WorkloadLimits{})
	fmt.Println(out.ObservedCPUAnnotations, out.NativeAnalysisStatus, out.NativePopulation == nil, err)
	// Output: 1 not_collected true <nil>
}

func ExampleOperatorCPUSummary() {
	c, source := cpuSummaryInputs()
	var wire bytes.Buffer
	_ = nvtelemetry.WriteOperatorCPUContext(&wire, c, nvtelemetry.WorkloadLimits{})
	out, _ := cudadiag.AnalyzeOperatorCPUJSON(context.Background(), &wire, source, nvtelemetry.WorkloadLimits{})
	fmt.Println(out.MembershipStatus, out.Annotations[0].Source.Locator)
	// Output: unavailable user_annotation/0
}

func ExampleOperatorCPUAnnotation() {
	a := cudadiag.OperatorCPUAnnotation{Source: nvtelemetry.OperatorReference{Locator: "user_annotation/0"}}
	fmt.Println(a.Source.Locator)
	// Output: user_annotation/0
}
