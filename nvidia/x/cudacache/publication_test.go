package cudacache_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
)

// publicationSource only calls in-process functions; it does not use a transport.
type publicationSource struct {
	artifact func() (io.ReadCloser, int64, error)
	receipt  func() ([]byte, error)
}

func (s publicationSource) FetchArtifact(context.Context, string) (io.ReadCloser, int64, error) {
	return s.artifact()
}

func (s publicationSource) FetchReceipt(context.Context, string) ([]byte, error) {
	return s.receipt()
}

func (publicationSource) PublishArtifact(context.Context, string, []byte) error {
	return nil
}

func (publicationSource) PublishReceipt(context.Context, string, []byte) error {
	return nil
}

type publicationReader struct {
	io.Reader
	beforeRead func()
}

func (r *publicationReader) Read(p []byte) (int, error) {
	if r.beforeRead != nil {
		r.beforeRead()
		r.beforeRead = nil
	}
	return r.Reader.Read(p)
}

func publicationGate(t *testing.T) (<-chan struct{}, func(), func()) {
	t.Helper()
	entered := make(chan struct{})
	released := make(chan struct{})
	var once sync.Once
	release := func() { once.Do(func() { close(released) }) }
	t.Cleanup(release)
	block := func() {
		close(entered)
		<-released
	}
	return entered, block, release
}

func waitPublication(t *testing.T, done <-chan struct{}) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("publication callback or local operation did not finish")
	}
}

func TestLocalPublicationWins(t *testing.T) {
	ctx := context.Background()
	for _, stage := range []string{"artifact-fetch", "artifact-read"} {
		t.Run(stage, func(t *testing.T) {
			root := t.TempDir()
			art := newTestArtifact(t, "publication", "sm_121a")
			// Keep the immutable descriptor and binary identical. Distinct JSON
			// encodings reveal whether the older callback overwrites the local file.
			older, err := json.MarshalIndent(art, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			local, err := json.Marshal(art)
			if err != nil {
				t.Fatal(err)
			}
			entered, block, release := publicationGate(t)
			source := publicationSource{artifact: func() (io.ReadCloser, int64, error) {
				r := &publicationReader{Reader: bytes.NewReader(older)}
				if stage == "artifact-fetch" {
					block()
				} else {
					r.beforeRead = block
				}
				return io.NopCloser(r), int64(len(older)), nil
			}}
			c, err := cudacache.New(cudacache.Config{RootDir: root, Remote: source})
			if err != nil {
				t.Fatal(err)
			}
			key := art.Digest
			readDone := make(chan struct{})
			go func() {
				defer close(readDone)
				got, err := c.GetArtifact(ctx, key)
				if err != nil {
					t.Error(err)
				} else if !reflect.DeepEqual(got, art) {
					t.Error("callback returned a different artifact")
				}
			}()
			waitPublication(t, entered)
			putDone := make(chan struct{})
			go func() {
				defer close(putDone)
				if err := c.PutArtifact(ctx, key, art); err != nil {
					t.Error(err)
				}
			}()
			waitPublication(t, putDone)
			release()
			waitPublication(t, readDone)
			data, err := os.ReadFile(filepath.Join(root, "artifacts", key+".artifact"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, local) {
				t.Fatal("older callback replaced the completed local artifact publication")
			}
			fresh, err := cudacache.New(cudacache.Config{RootDir: root})
			if err != nil {
				t.Fatal(err)
			}
			for _, cache := range []*cudacache.Cache{c, fresh} {
				got, err := cache.GetArtifact(ctx, key)
				if err != nil || !reflect.DeepEqual(got, art) {
					t.Fatalf("GetArtifact after publication = %v, %v", got, err)
				}
			}
		})
	}

	t.Run("receipt-fetch", func(t *testing.T) {
		root := t.TempDir()
		older := &cudacache.SearchReceipt{
			PolicyID:    cudacache.SearchPolicyID{Strategy: "beam", ScoringEpoch: "epoch"},
			WorkloadKey: "workload",
			CandidateID: "candidate",
			ArtifactKey: "artifact",
			Epoch:       "epoch",
		}
		local := *older
		local.Complete = true
		local.Timings = map[string]int64{"wall_elapsed": 100}
		data, err := json.Marshal(older)
		if err != nil {
			t.Fatal(err)
		}
		entered, block, release := publicationGate(t)
		source := publicationSource{receipt: func() ([]byte, error) {
			block()
			return data, nil
		}}
		c, err := cudacache.New(cudacache.Config{RootDir: root, Remote: source})
		if err != nil {
			t.Fatal(err)
		}
		readDone := make(chan struct{})
		go func() {
			defer close(readDone)
			got, err := c.GetReceipt(ctx, local.PolicyID, local.WorkloadKey)
			if err != nil {
				t.Error(err)
			} else if !reflect.DeepEqual(got, &local) {
				t.Error("older callback returned the incomplete receipt")
			}
		}()
		waitPublication(t, entered)
		putDone := make(chan struct{})
		go func() {
			defer close(putDone)
			if err := c.PutReceipt(ctx, &local); err != nil {
				t.Error(err)
			}
		}()
		waitPublication(t, putDone)
		release()
		waitPublication(t, readDone)
		fresh, err := cudacache.New(cudacache.Config{RootDir: root})
		if err != nil {
			t.Fatal(err)
		}
		for _, cache := range []*cudacache.Cache{c, fresh} {
			got, err := cache.GetReceipt(ctx, local.PolicyID, local.WorkloadKey)
			if err != nil || !reflect.DeepEqual(got, &local) {
				t.Fatalf("GetReceipt after publication = %v, %v", got, err)
			}
		}
	})

	t.Run("receipt-writers", func(t *testing.T) {
		root := t.TempDir()
		c, err := cudacache.New(cudacache.Config{RootDir: root})
		if err != nil {
			t.Fatal(err)
		}
		policy := cudacache.SearchPolicyID{Strategy: "beam"}
		var wg sync.WaitGroup
		for i := 0; i < 32; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				r := &cudacache.SearchReceipt{
					PolicyID: policy, WorkloadKey: "workload",
					CandidateID: fmt.Sprintf("candidate-%d", i), ArtifactKey: "artifact",
				}
				if err := c.PutReceipt(ctx, r); err != nil {
					t.Error(err)
				}
			}(i)
		}
		wg.Wait()
		fresh, err := cudacache.New(cudacache.Config{RootDir: root})
		if err != nil {
			t.Fatal(err)
		}
		memory, err := c.GetReceipt(ctx, policy, "workload")
		if err != nil {
			t.Fatal(err)
		}
		disk, err := fresh.GetReceipt(ctx, policy, "workload")
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(memory, disk) {
			t.Fatal("completed writers left different disk and memory receipts")
		}
	})
}
