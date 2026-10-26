package cudashadow_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabpfdev"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudashadow"
)

func ExampleBeginLaunchObservation() {
	artifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry kernel() { ret; }\n"), "kernel", cudakernel.Schema{}, "sm_80")
	if err != nil {
		panic(err)
	}
	source, err := cudabpfdev.NewArtifactSource(artifact, cudabpfdev.Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
	if err != nil {
		panic(err)
	}
	evaluator, err := cudashadow.NewEvaluator(4)
	if err != nil {
		panic(err)
	}
	// Supply the existing caller-owned loaded original's view here. This example
	// uses an unready owner and refuses observation without loading or launching.
	owner := new(cudabpfdev.LoadedInjection)
	_, err = cudashadow.BeginLaunchObservation(context.Background(), evaluator, source, owner.ObservationReader(), cudashadow.LaunchObservationOptions{
		Source: "example-host-epoch", Sequence: 1,
		Policy: cudashadow.PolicyInfo{Name: "artifact-ptx", Version: "1", Domain: cudashadow.DomainPTXInstrument, Seed: 23},
	})
	fmt.Println(err != nil, len(evaluator.Snapshots()))
	// Output: true 0
}

func ExamplePendingLaunchObservation_Complete() {
	var pending *cudashadow.PendingLaunchObservation
	fmt.Println(pending.Complete(context.Background()) != nil)
	// Output: true
}
