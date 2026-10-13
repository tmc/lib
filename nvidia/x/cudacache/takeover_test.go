package cudacache_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
)

func takeoverDir(t *testing.T) string {
	t.Helper()
	return t.TempDir()
}

func TestTakeoverKeyConfinement(t *testing.T) {
	dir := takeoverDir(t)
	c, err := cudacache.New(cudacache.Config{RootDir: filepath.Join(dir, "cache")})
	if err != nil {
		t.Fatal(err)
	}
	a := newTestArtifact(t, "fixture", "sm_90")
	for _, key := range []string{"../escape", "../../escape", "a/b", "a\\b", ".."} {
		if err := c.PutArtifact(context.Background(), key, a); err == nil {
			t.Errorf("accepted key %q", key)
		}
	}
}

func TestTakeoverArtifactIsolation(t *testing.T) {
	c, err := cudacache.New(cudacache.Config{RootDir: takeoverDir(t)})
	if err != nil {
		t.Fatal(err)
	}
	a := newTestArtifact(t, "fixture", "sm_90")
	want := string(a.Raw)
	if err := c.PutArtifact(context.Background(), "key", a); err != nil {
		t.Fatal(err)
	}
	a.Raw[0] = '!'
	b, err := c.GetArtifact(context.Background(), "key")
	if err != nil {
		t.Fatal(err)
	}
	if string(b.Raw) != want {
		t.Fatal("caller mutation changed stored artifact")
	}
	b.Raw[0] = '?'
	d, err := c.GetArtifact(context.Background(), "key")
	if err != nil {
		t.Fatal(err)
	}
	if string(d.Raw) != want {
		t.Fatal("returned artifact mutation changed cache")
	}
}

func TestTakeoverReceiptReplay(t *testing.T) {
	dir := takeoverDir(t)
	c, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	policy := cudacache.SearchPolicyID{Strategy: "beam"}
	r := cudacache.SearchReceipt{PolicyID: policy, WorkloadKey: "wrong", CandidateID: "candidate", ArtifactKey: "artifact"}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	key := cudacache.SearchReceiptKey(policy, "wanted")
	if err := os.WriteFile(filepath.Join(dir, "receipts", key+".receipt"), data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := c.GetReceipt(context.Background(), policy, "wanted"); err == nil {
		t.Fatal("accepted foreign-workload receipt under requested key")
	}
}

func TestTakeoverSignatureRequirement(t *testing.T) {
	reg := cudacache.NewMemoryRegistry()
	c, err := cudacache.New(cudacache.Config{RootDir: takeoverDir(t), Remote: reg, Trust: cudacache.TrustPolicy{RequireSigned: true}})
	if err != nil {
		return
	} // Unsupported verification may fail at construction.
	a := newTestArtifact(t, "fixture", "sm_90")
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := reg.PublishArtifact(context.Background(), "key", data); err != nil {
		t.Fatal(err)
	}
	if _, err := c.GetArtifact(context.Background(), "key"); err == nil {
		t.Fatal("unsigned artifact passed required signature policy")
	}
}

func TestTakeoverRemoteReceiptBoundedBeforeMaterialization(t *testing.T) {
	payload := bytes.Repeat([]byte("x"), 4096)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "4096")
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	c, err := cudacache.New(cudacache.Config{
		RootDir:                takeoverDir(t),
		MaxRemoteDownloadBytes: 128,
		Remote:                 cudacache.NewHTTPRemoteSource(server.URL, server.Client()),
	})
	if err != nil {
		t.Fatal(err)
	}
	policy := cudacache.SearchPolicyID{Strategy: "beam"}
	_, err = c.GetReceipt(context.Background(), policy, "workload")
	if !errors.Is(err, cudacache.ErrDownloadLimitExceeded) {
		t.Fatalf("receipt oversized response error = %v, want ErrDownloadLimitExceeded", err)
	}
}

func TestTakeoverTrustPolicyFailsClosed(t *testing.T) {
	_, err := cudacache.New(cudacache.Config{
		RootDir: takeoverDir(t), Remote: cudacache.NewMemoryRegistry(),
		Trust: cudacache.TrustPolicy{AllowedOrigins: []string{"https://trusted.example"}},
	})
	if !errors.Is(err, cudacache.ErrUntrustedSource) {
		t.Fatalf("configured unsupported trust policy error = %v", err)
	}
}

type canceledRemote struct{ started chan struct{} }

func (c canceledRemote) FetchArtifact(ctx context.Context, key string) (io.ReadCloser, int64, error) {
	close(c.started)
	<-ctx.Done()
	return nil, 0, ctx.Err()
}
func (canceledRemote) FetchReceipt(ctx context.Context, key string) ([]byte, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}
func (canceledRemote) PublishArtifact(context.Context, string, []byte) error { return nil }
func (canceledRemote) PublishReceipt(context.Context, string, []byte) error  { return nil }

func TestTakeoverRemoteCancellationPreserved(t *testing.T) {
	r := canceledRemote{started: make(chan struct{})}
	c, err := cudacache.New(cudacache.Config{RootDir: takeoverDir(t), Remote: r})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() { _, lookupErr := c.GetArtifact(ctx, "cancelled"); result <- lookupErr }()
	<-r.started
	cancel()
	err = <-result
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled lookup error = %v, want context.Canceled", err)
	}
}
