package cudakernel_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func workspaceArtifact(t testing.TB, w cudakernel.WorkspaceRequirements) *cudakernel.Artifact {
	t.Helper()
	s, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("workspace descriptor fixture"), "kernel", s, "sm_80", cudakernel.WithResources(cudakernel.ResourceRequirements{Workspace: w}))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestWorkspaceImmutableWire(t *testing.T) {
	w := cudakernel.WorkspaceRequirements{Bytes: 4096, Alignment: 4096, Parameter: "scratch"}
	a := workspaceArtifact(t, w)
	detached := a.Spec()
	detached.Resources.Workspace.Bytes = 1
	resource := a.Resources()
	resource.Workspace.Alignment = 1
	resource.Workspace.Parameter = "input"
	if a.Resources().Workspace != w {
		t.Fatal("workspace declaration aliased mutable construction/accessor data")
	}
	wire, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	for _, decode := range []func([]byte) (*cudakernel.Artifact, error){cudakernel.ParseArtifact, func(b []byte) (*cudakernel.Artifact, error) {
		var a cudakernel.Artifact
		err := json.Unmarshal(b, &a)
		return &a, err
	}} {
		b, err := decode(wire)
		if err != nil || b.Resources().Workspace != w || b.Digest() != a.Digest() {
			t.Fatalf("workspace wire: %v", err)
		}
	}
	spec := a.Spec()
	spec.Resources.Workspace.Bytes = 8192
	b, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	if a.Digest() != b.Digest() {
		t.Fatal("raw byte digest was redefined to include workspace metadata")
	}
	other, _ := json.Marshal(b)
	if bytes.Equal(wire, other) {
		t.Fatal("workspace metadata absent from descriptor wire")
	}
	var published *cudakernel.Artifact
	published, err = cudakernel.NewArtifact(cudakernel.KindPTX, a.Bytes(), "kernel", a.Schema(), "sm_80", func(s *cudakernel.ArtifactSpec) { s.Resources.Workspace = w; detached = *s })
	if err != nil {
		t.Fatal(err)
	}
	detached.Resources.Workspace.Bytes = 2
	if published.Resources().Workspace != w {
		t.Fatal("retained option data aliases immutable artifact")
	}
}

func TestWorkspaceZeroWireCompatibility(t *testing.T) {
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "entry", cudakernel.Schema{}, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte("fixture ptx"))
	want := fmt.Sprintf(`{"kind":1,"digest":"%s","raw":"Zml4dHVyZSBwdHg=","entry_point":"entry","schema":{"params":null,"total_param_bytes":0},"target":"sm_80","resources":{"static_shared_memory":0,"allocated_registers":0,"peak_live_gpr":0},"jit_options":{}}`, hex.EncodeToString(sum[:]))
	got, err := json.Marshal(a)
	if err != nil || string(got) != want {
		t.Fatalf("legacy zero wire changed: %s %v", got, err)
	}
	if r, _ := json.Marshal(cudakernel.ResourceRequirements{}); string(r) != `{"static_shared_memory":0,"allocated_registers":0,"peak_live_gpr":0}` {
		t.Fatalf("legacy resource bytes: %s", r)
	}
}

func TestWorkspaceRefusesInvalidDeclaration(t *testing.T) {
	base := workspaceArtifact(t, cudakernel.WorkspaceRequirements{Bytes: 64, Alignment: 16, Parameter: "scratch"})
	wire, _ := json.Marshal(base)
	for _, tc := range []struct {
		name   string
		change func(*cudakernel.ArtifactSpec)
	}{
		{"zero bytes with alignment", func(s *cudakernel.ArtifactSpec) { s.Resources.Workspace.Bytes = 0 }},
		{"missing alignment", func(s *cudakernel.ArtifactSpec) { s.Resources.Workspace.Alignment = 0 }},
		{"nonpower alignment", func(s *cudakernel.ArtifactSpec) { s.Resources.Workspace.Alignment = 24 }},
		{"absent slot", func(s *cudakernel.ArtifactSpec) { s.Resources.Workspace.Parameter = "absent" }},
		{"empty slot", func(s *cudakernel.ArtifactSpec) { s.Resources.Workspace.Parameter = "" }},
		{"nul slot", func(s *cudakernel.ArtifactSpec) { s.Resources.Workspace.Parameter = "scratch\x00" }},
		{"scalar slot", func(s *cudakernel.ArtifactSpec) {
			s.Schema.Params[1].Type = "u64"
			s.Schema.Params[1].IsPointer = false
		}},
		{"wrong pointer width", func(s *cudakernel.ArtifactSpec) { s.Schema.Params[1].ByteWidth = 4 }},
		{"duplicate slot", func(s *cudakernel.ArtifactSpec) { s.Schema.Params[2].Name = "scratch" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			spec := base.Spec()
			tc.change(&spec)
			if _, err := cudakernel.NewArtifactFromSpec(spec); err == nil {
				t.Fatal("invalid workspace accepted")
			}
			bad, _ := json.Marshal(struct {
				cudakernel.ArtifactSpec
				Digest string `json:"digest"`
			}{spec, base.Digest()})
			if _, err := cudakernel.ParseArtifact(bad); err == nil {
				t.Fatal("invalid wire accepted")
			}
			old := *base
			if err := json.Unmarshal(bad, base); err == nil || *base != old {
				t.Fatal("invalid decode replaced immutable handle")
			}
		})
	}
	if !strings.Contains(string(wire), `"workspace":{"bytes":64,"alignment":16,"parameter":"scratch"}`) {
		t.Fatal("positive workspace is not explicit")
	}
}
