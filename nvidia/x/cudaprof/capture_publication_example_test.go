package cudaprof_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func examplePublication() (cudaprof.CapturePlan, cudaprof.CapturePublication) {
	// Synthetic data demonstrates transport validation, not hardware collection.
	schema, _ := cudakernel.NewSchema()
	artifact, _ := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_70\n.address_size 64\n.visible .entry kernel() { ret; }\n"), "kernel", schema, "sm_70")
	workload := cudacache.WorkloadIdentity{ProgramID: "example", ShapeDims: []int{64}, NumericalMode: "fp32"}
	id, _ := cudaprof.BindEvidence(artifact, workload, strings.Repeat("2", 64))
	receipt := exampleOwnerReceipt()
	receipt.Identity = id
	capture := &cuptirange.Capture{Image: []byte{1}, Ranges: 1, Quality: cuptirange.CollectionQuality{PassesStarted: 1, PassesStopped: 1, AllPassesSubmitted: true, Decoded: true}}
	result, _ := cudaprof.RangeCaptureProviderResult(receipt, capture, []byte{7}, []string{"metric"}, nil)
	plan := cudaprof.CapturePlan{Version: 1, Identity: id, Environment: receipt.Environment, Budget: cudaprof.CaptureBudget{MaxRecords: 10, MaxBytes: 1 << 20, MaxPasses: 2, MaxWallNanos: 1000}, Passes: []cudaprof.CapturePass{{ID: result.PassID, Replay: result.Replay, Providers: []cudaprof.CaptureProvider{"range"}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{"range": result.Config}}}}
	return plan, cudaprof.CapturePublication{Version: 1, Artifact: artifact, Workload: workload, CompilerKey: id.CompilerKey, Result: result}
}
func ExampleCapturePublication() {
	_, publication := examplePublication()
	fmt.Println(publication.Version, publication.Artifact.EntryPoint, publication.Result.Provider)
	// Output: 1 kernel range
}
func ExampleWriteCapturePublication() {
	plan, publication := examplePublication()
	var output bytes.Buffer
	err := cudaprof.WriteCapturePublication(&output, plan, publication)
	fmt.Println(output.Len() > 0, err)
	// Output: true <nil>
}
func ExampleReadCapturePublication() {
	plan, publication := examplePublication()
	var output bytes.Buffer
	_ = cudaprof.WriteCapturePublication(&output, plan, publication)
	got, err := cudaprof.ReadCapturePublication(&output, plan, plan.Budget.MaxBytes)
	fmt.Println(got.Result.Identity == plan.Identity, err)
	// Output: true <nil>
}
