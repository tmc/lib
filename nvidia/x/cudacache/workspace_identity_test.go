package cudacache_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func workspaceCacheArtifact(t testing.TB, w cudakernel.WorkspaceRequirements) *cudakernel.Artifact {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "scratch", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "other", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("workspace cache fixture"), "kernel", schema, "sm_80", cudakernel.WithResources(cudakernel.ResourceRequirements{Workspace: w}))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestWorkspaceCacheIdentityAndWire(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cache, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	adapter := cudacache.NewNVRTCAdapter(cache)
	id := cudacache.CompilerIdentity{NVRTCVersion: "fixture", DriverSONAME: "fixture-driver", TargetArch: "sm_80", OutputKind: nvrtcx.OutputPTX}
	// This golden was calculated from the current116 preimage's length-framed
	// compiler-key recipe. Zero workspace must not invalidate existing keys.
	if got := cudacache.CompilerCacheKey("workspace source", id); got != "07d4b90c50d40cdbe736bb42044e4af81b43b93d6e494acda966681b9741029d" {
		t.Fatalf("legacy zero key changed: %s", got)
	}
	type legacyIdentity struct {
		NVRTCVersion string
		DriverSONAME string
		TargetArch   string
		Flags        []string
		OutputKind   nvrtcx.OutputKind
		Provenance   cudacache.ToolProvenance
		JITOptions   cudakernel.JITOptions
	}
	want, _ := json.Marshal(legacyIdentity{id.NVRTCVersion, id.DriverSONAME, id.TargetArch, id.Flags, id.OutputKind, id.Provenance, id.JITOptions})
	got, _ := json.Marshal(id)
	if !bytes.Equal(got, want) {
		t.Fatalf("zero identity wire changed: %s != %s", got, want)
	}
	seen := make(map[string]bool)
	for _, w := range []cudakernel.WorkspaceRequirements{{}, {Bytes: 4096, Alignment: 4096, Parameter: "scratch"}, {Bytes: 8192, Alignment: 4096, Parameter: "scratch"}, {Bytes: 4096, Alignment: 512, Parameter: "scratch"}, {Bytes: 4096, Alignment: 4096, Parameter: "other"}} {
		identity := id
		identity.Workspace = w
		key := cudacache.CompilerCacheKey("workspace source", identity)
		if seen[key] {
			t.Fatal("workspace contract key collision")
		}
		seen[key] = true
		art := workspaceCacheArtifact(t, w)
		if err := adapter.Store(ctx, "workspace source", identity, art); err != nil {
			t.Fatal(err)
		}
		fresh, err := cudacache.New(cudacache.Config{RootDir: dir})
		if err != nil {
			t.Fatal(err)
		}
		loaded, hit, err := cudacache.NewNVRTCAdapter(fresh).Lookup(ctx, "workspace source", identity)
		if err != nil || !hit || loaded.Resources().Workspace != w || loaded.Digest() != art.Digest() {
			t.Fatalf("workspace cache roundtrip: %v %v", hit, err)
		}
		encoded, err := cudacache.MarshalArtifact(art)
		if err != nil {
			t.Fatal(err)
		}
		if w.Bytes == 0 {
			legacy, _ := json.Marshal(art)
			expected := append([]byte(`{"schema_version":1,`), legacy[1:]...)
			if !bytes.Equal(encoded, expected) {
				t.Fatalf("zero cache wire changed: %s", encoded)
			}
		}
	}
}

func TestWorkspaceCacheRefusesMismatch(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cache, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	adapter := cudacache.NewNVRTCAdapter(cache)
	w := cudakernel.WorkspaceRequirements{Bytes: 4096, Alignment: 4096, Parameter: "scratch"}
	id := cudacache.CompilerIdentity{TargetArch: "sm_80", OutputKind: nvrtcx.OutputPTX, Workspace: w}
	art := workspaceCacheArtifact(t, w)
	wrong := id
	wrong.Workspace.Bytes++
	if err := adapter.Store(ctx, "source", wrong, art); err == nil {
		t.Fatal("mismatched workspace published")
	}
	if err := adapter.Store(ctx, "source", id, art); err != nil {
		t.Fatal(err)
	}
	key := cudacache.CompilerCacheKey("source", id)
	path := filepath.Join(dir, "artifacts", key+"-"+art.Digest()+".artifact")
	wire, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var object map[string]any
	if err := json.Unmarshal(wire, &object); err != nil {
		t.Fatal(err)
	}
	object["resources"].(map[string]any)["workspace"].(map[string]any)["bytes"] = 8192
	edited, _ := json.Marshal(object)
	if err := os.WriteFile(path, edited, 0600); err != nil {
		t.Fatal(err)
	}
	fresh, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if _, hit, err := cudacache.NewNVRTCAdapter(fresh).Lookup(ctx, "source", id); hit || !errors.Is(err, cudacache.ErrCorruptArtifact) {
		t.Fatalf("same raw digest hid workspace tamper: %v %v", hit, err)
	}
	invalid := id
	invalid.Workspace.Alignment = 3
	if _, _, err := adapter.Lookup(ctx, "source", invalid); err == nil {
		t.Fatal("invalid expected alignment accepted")
	}
}
