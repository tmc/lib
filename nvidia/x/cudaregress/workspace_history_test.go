package cudaregress

import (
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"testing"
)

func TestWorkspaceHistoryLinkRequiresMatchingCompiler(t *testing.T) {
	p, e, c, o := historyFixture(t)
	spec := o.Artifact.Spec()
	var err error
	spec.Schema, err = cudakernel.NewSchema(cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	spec.Resources.Workspace = cudakernel.WorkspaceRequirements{Bytes: 4096, Alignment: 4096, Parameter: "scratch"}
	o.Artifact, err = cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LinkToProfileHistory(p, e, c, o); err == nil {
		t.Fatal("zero expected workspace accepted positive artifact")
	}
	o.Compiler.Workspace = spec.Resources.Workspace
	r, err := LinkToProfileHistory(p, e, c, o)
	if err != nil || r.Compiler.Workspace != spec.Resources.Workspace || r.Artifact.Resources().Workspace != spec.Resources.Workspace {
		t.Fatalf("matching workspace link: %v", err)
	}
	o.Compiler.Workspace.Alignment = 512
	if _, err := LinkToProfileHistory(p, e, c, o); err == nil {
		t.Fatal("wrong expected alignment accepted")
	}
}
