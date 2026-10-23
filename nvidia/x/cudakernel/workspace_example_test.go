package cudakernel_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func ExampleWorkspaceRequirements() {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true})
	if err != nil {
		fmt.Println(err)
		return
	}
	w := cudakernel.WorkspaceRequirements{Bytes: 4096, Alignment: 4096, Parameter: "scratch"}
	artifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("descriptor fixture only"), "kernel", schema, "sm_80", cudakernel.WithResources(cudakernel.ResourceRequirements{Workspace: w}))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(artifact.Resources().Workspace)
	// Output: {4096 4096 scratch}
}

func ExampleWorkspaceRequirements_Validate() {
	fmt.Println((cudakernel.WorkspaceRequirements{}).Validate())
	// Output: <nil>
}
