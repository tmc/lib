package cudasearch

import (
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"strings"
	"testing"
)

func TestSequenceWorkspaceRefusedBeforeLoading(t *testing.T) {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("host-only sequence admission fixture"), "entry", schema, "sm_80", cudakernel.WithResources(cudakernel.ResourceRequirements{Workspace: cudakernel.WorkspaceRequirements{Bytes: 4096, Alignment: 4096, Parameter: "scratch"}}))
	if err != nil {
		t.Fatal(err)
	}
	got, err := sequenceArtifactHandles([]*cudakernel.Artifact{a}, 1)
	if got != nil || err == nil || !strings.Contains(err.Error(), "workspace is unsupported") {
		t.Fatalf("nonbinding sequence silently admitted workspace: %v %v", got, err)
	}
	spec := a.Spec()
	spec.Resources.Workspace = cudakernel.WorkspaceRequirements{}
	zero, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	got, err = sequenceArtifactHandles([]*cudakernel.Artifact{zero}, 1)
	if err != nil || len(got) != 1 || got[0] != *zero {
		t.Fatal("zero sequence handle behavior changed")
	}
}
