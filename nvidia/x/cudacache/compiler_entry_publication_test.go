package cudacache

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type compilerEntrySource struct {
	fetch func() (io.ReadCloser, int64, error)
}

func (s compilerEntrySource) FetchArtifact(context.Context, string) (io.ReadCloser, int64, error) {
	return s.fetch()
}
func (compilerEntrySource) FetchReceipt(context.Context, string) ([]byte, error) {
	return nil, ErrRemoteNotFound
}
func (compilerEntrySource) PublishArtifact(context.Context, string, []byte) error { return nil }
func (compilerEntrySource) PublishReceipt(context.Context, string, []byte) error  { return nil }

type compilerEntryReader struct {
	io.Reader
	read, close func()
}

func (r *compilerEntryReader) Read(p []byte) (int, error) {
	if r.read != nil {
		f := r.read
		r.read = nil
		f()
	}
	return r.Reader.Read(p)
}
func (r *compilerEntryReader) Close() error {
	if r.close != nil {
		r.close()
	}
	return nil
}

func compilerEntryFixture(t *testing.T) (*Cache, CompilerIdentity, *cudakernel.Artifact) {
	t.Helper()
	dir := t.TempDir()
	tool := filepath.Join(dir, "tool")
	if err := os.WriteFile(tool, []byte("inert tool identity fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	d, err := DigestTool("fixture", tool)
	if err != nil {
		t.Fatal(err)
	}
	id := CompilerIdentity{NVRTCVersion: "fixture", TargetArch: "sm_121a", OutputKind: nvrtcx.OutputPTX,
		Provenance: ToolProvenance{Recipe: RecipeVersion, Tools: []ToolDigest{d}, DriverVersion: 13000}}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "data", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX,
		[]byte(".version 8.0\n.target sm_121a\n.address_size 64\n.entry k(.param .u64 ptr) { ret; }\n"), "k", schema, id.TargetArch)
	if err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{RootDir: filepath.Join(dir, "cache")})
	if err != nil {
		t.Fatal(err)
	}
	return c, id, art
}

func compilerEntryRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// All callbacks run in process and supply fixed artifacts. The tool file is
// hashed, never loaded or executed.
func TestCompilerEntryMarkerFreshness(t *testing.T) {
	for _, stage := range []string{"fetch", "read", "close"} {
		for _, old := range []string{"matching", "same mismatch", "different mismatch"} {
			for _, newPayload := range []bool{false, true} {
				name := stage + "/" + old + "/same artifact"
				if newPayload {
					name = stage + "/" + old + "/new artifact"
				}
				t.Run(name, func(t *testing.T) {
					ctx := context.Background()
					producer, id, original := compilerEntryFixture(t)
					first := NewNVRTCAdapter(producer)
					recorded := id.Provenance
					if old != "matching" {
						recorded.DriverVersion--
					}
					if _, err := first.GetOrCompile(ctx, "input", id, func(context.Context) (*cudakernel.Artifact, ToolProvenance, error) {
						return original, recorded, nil
					}); err != nil {
						t.Fatal(err)
					}
					key := CompilerCacheKey("input", id)
					inspected, err := checkManifest(key, compilerEntryRead(t, producer.manifestPath(key)))
					if err != nil {
						t.Fatal(err)
					}
					if err := os.Remove(producer.artifactPath(inspected.ArtifactKey)); err != nil {
						t.Fatal(err)
					}
					replacement := original
					if newPayload {
						replacement, err = cudakernel.NewArtifact(original.Kind, append(append([]byte(nil), original.Raw...), []byte("\n// replacement\n")...), original.EntryPoint, original.Schema, original.Target)
						if err != nil {
							t.Fatal(err)
						}
					}
					remote, err := json.Marshal(original)
					if err != nil {
						t.Fatal(err)
					}
					var adapter *NVRTCAdapter
					var published, marker []byte
					callbacks := 0
					replace := func() {
						callbacks++
						// Force a public miss, then a mismatch/rebuild cycle. Its
						// marker names the newly evicted nonce, not the old lookup.
						if err := os.Remove(producer.manifestPath(key)); err != nil {
							t.Fatal(err)
						}
						newRecorded := id.Provenance
						newRecorded.DriverVersion--
						if old == "different mismatch" {
							newRecorded.DriverVersion--
						}
						for range 2 {
							if _, err := adapter.GetOrCompile(ctx, "input", id, func(context.Context) (*cudakernel.Artifact, ToolProvenance, error) {
								return replacement, newRecorded, nil
							}); err != nil {
								t.Fatal(err)
							}
						}
						published = compilerEntryRead(t, producer.manifestPath(key))
						marker = compilerEntryRead(t, producer.markerPath(key))
					}
					source := compilerEntrySource{fetch: func() (io.ReadCloser, int64, error) {
						r := &compilerEntryReader{Reader: bytes.NewReader(remote)}
						switch stage {
						case "fetch":
							replace()
						case "read":
							r.read = replace
						case "close":
							r.close = replace
						}
						return r, int64(len(remote)), nil
					}}
					c, err := New(Config{RootDir: producer.rootDir, Remote: source})
					if err != nil {
						t.Fatal(err)
					}
					adapter = NewNVRTCAdapter(c)
					if got, hit, err := adapter.Lookup(ctx, "input", id); got != nil || hit || err != nil || callbacks != 1 {
						t.Fatalf("superseded lookup: artifact=%v hit=%t err=%v callbacks=%d", got, hit, err, callbacks)
					}
					if !bytes.Equal(compilerEntryRead(t, c.manifestPath(key)), published) || !bytes.Equal(compilerEntryRead(t, c.markerPath(key)), marker) {
						t.Fatal("superseded lookup changed replacement manifest or marker")
					}
					fresh, err := New(Config{RootDir: c.rootDir})
					if err != nil {
						t.Fatal(err)
					}
					for _, reader := range []*Cache{c, fresh} {
						m, err := checkManifest(key, published)
						if err != nil {
							t.Fatal(err)
						}
						got, err := reader.GetArtifact(ctx, m.ArtifactKey)
						if err != nil || !reflect.DeepEqual(got, replacement) {
							t.Fatalf("replacement artifact: %v", err)
						}
						if _, hit, err := NewNVRTCAdapter(reader).Lookup(ctx, "input", id); hit || !errors.Is(err, ErrUnhealableToolchainMismatch) {
							t.Fatalf("replacement mismatch: hit=%t err=%v", hit, err)
						}
					}
				})
			}
		}
	}
}

func compilerEntryWait(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("compiler entry operation did not finish")
	}
}

func TestCompilerEntryStoreRetirement(t *testing.T) {
	for _, retire := range []string{"quarantine", "mismatch"} {
		t.Run(retire, func(t *testing.T) {
			ctx := context.Background()
			c, id, art := compilerEntryFixture(t)
			a := NewNVRTCAdapter(c)
			recorded := id.Provenance
			recorded.DriverVersion--
			if err := a.store(ctx, "input", id, recorded, art); err != nil {
				t.Fatal(err)
			}
			key := CompilerCacheKey("input", id)
			inspected := compilerEntryRead(t, c.manifestPath(key))
			m, err := checkManifest(key, inspected)
			if err != nil {
				t.Fatal(err)
			}
			// Store reaches disk publication, then waits for the memory lock.
			// Retirement starts while Store owns publicationMu. Both must
			// complete without retiring the replacement, even for the same key.
			c.memMu.Lock()
			storeDone := make(chan struct{})
			var storeErr, retireErr error
			go func() { defer close(storeDone); storeErr = a.Store(ctx, "input", id, art) }()
			deadline := time.Now().Add(5 * time.Second)
			for c.publicationMu.TryLock() {
				c.publicationMu.Unlock()
				if time.Now().After(deadline) {
					c.memMu.Unlock()
					t.Fatal("Store did not reach publication")
				}
				runtime.Gosched()
			}
			retireDone := make(chan struct{})
			go func() {
				defer close(retireDone)
				if retire == "quarantine" {
					c.quarantineEntry(key, inspected, m.ArtifactKey)
				} else {
					_, retireErr = c.finishCompilerLookup(key, inspected, m, id.Provenance)
				}
			}()
			c.memMu.Unlock()
			compilerEntryWait(t, storeDone)
			compilerEntryWait(t, retireDone)
			if storeErr != nil || retireErr != nil {
				t.Fatalf("Store=%v retirement=%v", storeErr, retireErr)
			}
			fresh, err := New(Config{RootDir: c.rootDir})
			if err != nil {
				t.Fatal(err)
			}
			for _, reader := range []*Cache{c, fresh} {
				got, hit, err := NewNVRTCAdapter(reader).Lookup(ctx, "input", id)
				if err != nil || !hit || !reflect.DeepEqual(got, art) {
					t.Fatalf("completed Store: hit=%t error=%v", hit, err)
				}
			}
		})
	}
}
