package cudacache_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestManifestQuarantineSnapshot(t *testing.T) {
	for _, tt := range []struct {
		name       string
		republish  bool
		newPayload bool
	}{
		{"current manifest", false, false},
		{"new manifest same artifact", true, false},
		{"new manifest new artifact", true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			root := t.TempDir()
			id := cudacache.CompilerIdentity{NVRTCVersion: "13.0", TargetArch: "sm_121a", OutputKind: nvrtcx.OutputPTX}
			original := newTestArtifact(t, "k", id.TargetArch)
			producer, err := cudacache.New(cudacache.Config{RootDir: root})
			if err != nil {
				t.Fatal(err)
			}
			if err := cudacache.NewNVRTCAdapter(producer).Store(ctx, "input", id, original); err != nil {
				t.Fatal(err)
			}
			key := cudacache.CompilerCacheKey("input", id)
			path := filepath.Join(root, "manifests", key+".manifest")
			inspected, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			manifest := readManifest(t, path)
			// Leave an intact manifest naming a missing artifact. A fresh cache
			// must consult the remote source rather than the producer's memory.
			if err := os.Remove(filepath.Join(root, "artifacts", manifest.ArtifactKey+".artifact")); err != nil {
				t.Fatal(err)
			}
			replacement := original
			if tt.newPayload {
				raw := append(append([]byte(nil), original.Raw...), []byte("\n// replacement\n")...)
				replacement, err = cudakernel.NewArtifact(original.Kind, raw, original.EntryPoint, original.Schema, original.Target)
				if err != nil {
					t.Fatal(err)
				}
			}
			var adapter *cudacache.NVRTCAdapter
			var published []byte
			fetches := 0
			source := publicationSource{artifact: func() (io.ReadCloser, int64, error) {
				fetches++
				if tt.republish {
					// A real reentrant Store completes before the old lookup sees
					// its missing-artifact result. No callback may run under the
					// publication mutex needed by this Store.
					if err := adapter.Store(ctx, "input", id, replacement); err != nil {
						t.Fatal(err)
					}
					published, err = os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
				}
				return nil, 0, cudacache.ErrRemoteNotFound
			}}
			cache, err := cudacache.New(cudacache.Config{RootDir: root, Remote: source})
			if err != nil {
				t.Fatal(err)
			}
			adapter = cudacache.NewNVRTCAdapter(cache)
			got, ok, err := adapter.Lookup(ctx, "input", id)
			if got != nil || ok || !errors.Is(err, cudacache.ErrCorruptArtifact) || fetches != 1 {
				t.Fatalf("old lookup: artifact=%v hit=%t error=%v fetches=%d", got, ok, err, fetches)
			}
			if !tt.republish {
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Fatalf("current corrupt manifest retained: %v", err)
				}
				entries, err := os.ReadDir(filepath.Join(root, "quarantine"))
				if err != nil || len(entries) != 1 {
					t.Fatalf("quarantine entries=%v error=%v", entries, err)
				}
				retained, err := os.ReadFile(filepath.Join(root, "quarantine", entries[0].Name()))
				if err != nil || !bytes.Equal(retained, inspected) {
					t.Fatalf("quarantine did not retain inspected manifest: %v", err)
				}
				return
			}
			current, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(current, published) || bytes.Equal(current, inspected) {
				t.Fatalf("old lookup retired or changed completed replacement: %v", err)
			}
			fresh, err := cudacache.New(cudacache.Config{RootDir: root})
			if err != nil {
				t.Fatal(err)
			}
			for _, reader := range []*cudacache.NVRTCAdapter{adapter, cudacache.NewNVRTCAdapter(fresh)} {
				got, ok, err := reader.Lookup(ctx, "input", id)
				if err != nil || !ok || !reflect.DeepEqual(got, replacement) {
					t.Fatalf("completed replacement lookup: artifact=%v hit=%t error=%v", got, ok, err)
				}
			}
			if fetches != 1 {
				t.Fatalf("replacement required another remote fetch: %d", fetches)
			}
		})
	}
}

func TestManifestEvictionSnapshot(t *testing.T) {
	for _, newPayload := range []bool{false, true} {
		name := "same artifact"
		if newPayload {
			name = "new artifact"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			f := newToolchainFixture(t)
			original := newTestArtifact(t, "k", f.id.TargetArch)
			recorded := f.id.Provenance
			recorded.DriverVersion--
			// A host-only callback supplies an artifact with deliberately stale
			// provenance; no compiler or driver is invoked.
			if _, err := f.adapter.GetOrCompile(ctx, "input", f.id, func(context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
				return original, recorded, nil
			}); err != nil {
				t.Fatal(err)
			}
			inspected := readManifest(t, f.manifestPath())
			root := filepath.Join(f.dir, "cache")
			if err := os.Remove(filepath.Join(root, "artifacts", inspected.ArtifactKey+".artifact")); err != nil {
				t.Fatal(err)
			}
			replacement := original
			if newPayload {
				raw := append(append([]byte(nil), original.Raw...), []byte("\n// replacement\n")...)
				var err error
				replacement, err = cudakernel.NewArtifact(original.Kind, raw, original.EntryPoint, original.Schema, original.Target)
				if err != nil {
					t.Fatal(err)
				}
			}
			remoteData, err := cudacache.MarshalArtifact(original)
			if err != nil {
				t.Fatal(err)
			}
			var adapter *cudacache.NVRTCAdapter
			var published []byte
			fetches := 0
			source := publicationSource{artifact: func() (io.ReadCloser, int64, error) {
				fetches++
				if err := adapter.Store(ctx, "input", f.id, replacement); err != nil {
					t.Fatal(err)
				}
				published, err = os.ReadFile(f.manifestPath())
				if err != nil {
					t.Fatal(err)
				}
				return io.NopCloser(bytes.NewReader(remoteData)), int64(len(remoteData)), nil
			}}
			cache, err := cudacache.New(cudacache.Config{RootDir: root, Remote: source})
			if err != nil {
				t.Fatal(err)
			}
			adapter = cudacache.NewNVRTCAdapter(cache)
			if got, ok, err := adapter.Lookup(ctx, "input", f.id); got != nil || ok || err != nil || fetches != 1 {
				t.Fatalf("stale toolchain lookup: artifact=%v hit=%t error=%v fetches=%d", got, ok, err, fetches)
			}
			current, err := os.ReadFile(f.manifestPath())
			if err != nil || !bytes.Equal(current, published) {
				t.Fatalf("stale healing retired completed replacement: %v", err)
			}
			if current := readManifest(t, f.manifestPath()); current.Nonce == inspected.Nonce || current.Recorded.Fingerprint() != f.id.Provenance.Fingerprint() {
				t.Fatal("replacement did not publish a new manifest with current provenance")
			}
			if _, err := os.Stat(f.markerPath()); !os.IsNotExist(err) {
				t.Fatalf("stale lookup changed replacement marker state: %v", err)
			}
			fresh, err := cudacache.New(cudacache.Config{RootDir: root})
			if err != nil {
				t.Fatal(err)
			}
			for _, reader := range []*cudacache.NVRTCAdapter{adapter, cudacache.NewNVRTCAdapter(fresh)} {
				got, ok, err := reader.Lookup(ctx, "input", f.id)
				if err != nil || !ok || !reflect.DeepEqual(got, replacement) {
					t.Fatalf("completed replacement lookup: artifact=%v hit=%t error=%v", got, ok, err)
				}
			}
			if _, err := os.Stat(f.markerPath()); !os.IsNotExist(err) {
				t.Fatalf("current-provenance hit did not clear marker: %v", err)
			}
			if fetches != 1 {
				t.Fatalf("replacement required another remote fetch: %d", fetches)
			}
		})
	}
}
