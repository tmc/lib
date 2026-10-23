package cudaprofilehistory

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func workspaceRecord(t testing.TB, workspaceBytes uint64) Record {
	t.Helper()
	r := fixture(t)
	spec := r.Artifact.Spec()
	var err error
	spec.Schema, err = cudakernel.NewSchema(cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	spec.Resources.Workspace = cudakernel.WorkspaceRequirements{Bytes: workspaceBytes, Alignment: 4096, Parameter: "scratch"}
	r.Artifact, err = cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	r.Compiler.Workspace = spec.Resources.Workspace
	r.CompilerKey = cudacache.CompilerCacheKey(r.CompilerInputKey, r.Compiler)
	r.Identity.CompilerKey = r.CompilerKey
	r.Facts = nil
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	return r
}

func TestWorkspaceHistoryObservationAndWire(t *testing.T) {
	a, b := workspaceRecord(t, 4096), workspaceRecord(t, 8192)
	if a.Artifact.Digest() != b.Artifact.Digest() {
		t.Fatal("raw digest became a metadata digest")
	}
	ka, err := ObservationKey(a)
	if err != nil {
		t.Fatal(err)
	}
	kb, err := ObservationKey(b)
	if err != nil || ka == kb {
		t.Fatal("workspace absent from observation identity")
	}
	s, err := NewStore(limits())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Append(a); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := s.WriteSnapshot(&out); err != nil {
		t.Fatal(err)
	}
	restored, err := ReadSnapshot(&out, limits())
	if err != nil {
		t.Fatal(err)
	}
	rows, err := restored.Records()
	if err != nil || len(rows) != 1 || rows[0].Artifact.Resources().Workspace != a.Artifact.Resources().Workspace || rows[0].Compiler.Workspace != a.Compiler.Workspace {
		t.Fatalf("workspace history snapshot: %v", err)
	}
	zero, _ := json.Marshal(fixture(t))
	if strings.Contains(string(zero), `"workspace"`) || strings.Contains(string(zero), `"Workspace"`) {
		t.Fatal("zero workspace changed legacy record JSON")
	}
	// Change only a declared artifact requirement: complete canonical record
	// identity sees the change even though code-byte identity remains equal.
	edited := a
	spec := a.Artifact.Spec()
	spec.Resources.Workspace.Bytes++
	edited.Artifact, err = cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	if err := edited.Validate(); err == nil {
		t.Fatal("direct history compiler/artifact mismatch accepted")
	}
	bad, _ := json.Marshal(edited)
	if _, err := DecodeRecord(bad, len(bad)); err == nil {
		t.Fatal("decoded history compiler/artifact mismatch accepted")
	}
}
