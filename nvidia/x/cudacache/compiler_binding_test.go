package cudacache_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestVerifyToolRejectsUnverifiedClaimedDigest(t *testing.T) {
	f := newToolchainFixture(t)
	d := f.id.Provenance.Tools[0]
	if err := cudacache.VerifyTool(d); err != nil {
		t.Fatal("known constructor digest rejected:", err)
	}
	d.SHA256 = strings.Repeat("0", 64)
	if !errors.Is(cudacache.VerifyTool(d), cudacache.ErrToolIdentityChanged) {
		t.Fatal("stat identity alone verified a wrong claimed digest")
	}
}

func TestAdapterRejectsWrongCompilerArtifact(t *testing.T) {
	for _, tc := range []struct {
		name   string
		target string
		kind   cudakernel.Kind
		output nvrtcx.OutputKind
	}{
		{"target", "sm_90", cudakernel.KindPTX, nvrtcx.OutputPTX},
		{"output", "sm_121a", cudakernel.KindPTX, nvrtcx.OutputCUBIN},
		{"unknown kind", "sm_121a", cudakernel.Kind(255), nvrtcx.OutputPTX},
		{"unknown output", "sm_121a", cudakernel.KindPTX, nvrtcx.OutputKind(255)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newToolchainFixture(t)
			art := newTestArtifact(t, "k", tc.target)
			art.Kind = tc.kind
			f.id.OutputKind = tc.output
			if err := f.adapter.Store(context.Background(), "input", f.id, art); err == nil {
				t.Fatal("inconsistent artifact stored")
			}
			if _, err := os.Stat(f.manifestPath()); !os.IsNotExist(err) {
				t.Fatal("failed Store published manifest")
			}
		})
	}
}

func TestAdapterQuarantinesWrongArtifactMetadata(t *testing.T) {
	for _, field := range []string{"target", "kind"} {
		t.Run(field, func(t *testing.T) {
			f := newToolchainFixture(t)
			art := newTestArtifact(t, "k", "sm_121a")
			if err := f.adapter.Store(context.Background(), "input", f.id, art); err != nil {
				t.Fatal(err)
			}
			m := readManifest(t, f.manifestPath())
			path := filepath.Join(f.dir, "cache", "artifacts", m.ArtifactKey+".artifact")
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var stored cudakernel.Artifact
			if err := json.Unmarshal(b, &stored); err != nil {
				t.Fatal(err)
			}
			if field == "target" {
				stored.Target = "sm_90"
			} else {
				stored.Kind = cudakernel.KindLTOIR
			}
			b, err = json.Marshal(stored)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, b, 0600); err != nil {
				t.Fatal(err)
			}
			// A fresh reader exercises persisted data rather than a producer's
			// immutable in-memory artifact copy.
			c, err := cudacache.New(cudacache.Config{RootDir: filepath.Join(f.dir, "cache")})
			if err != nil {
				t.Fatal(err)
			}
			got, ok, err := cudacache.NewNVRTCAdapter(c).Lookup(context.Background(), "input", f.id)
			if got != nil || ok || !errors.Is(err, cudacache.ErrCorruptArtifact) {
				t.Fatalf("inconsistent lookup: %v %t %v", got, ok, err)
			}
			if _, err := os.Stat(f.manifestPath()); !os.IsNotExist(err) {
				t.Fatal("inconsistent manifest not quarantined")
			}
		})
	}
}
