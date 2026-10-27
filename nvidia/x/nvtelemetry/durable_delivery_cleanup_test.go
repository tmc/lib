//go:build darwin || linux

package nvtelemetry_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type durableCleanupTransport func(*http.Request) (*http.Response, error)

func (f durableCleanupTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func newCleanupDelivery(t *testing.T) (*nvtelemetry.DurableDelivery, nvtelemetry.DurableDeliveryConfig) {
	t.Helper()
	c := nvtelemetry.DurableDeliveryConfig{
		Delivery: nvtelemetry.DeliveryConfig{Endpoint: "https://delivery.invalid/v1/metrics", Client: &http.Client{Transport: durableCleanupTransport(func(r *http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`)), Request: r}, nil
		})}, MaxQueuedBatches: 1, MaxQueuedBytes: 4096, MaxBatchBytes: 4096, MaxBatchDataPoints: 4, MaxResponseBytes: 512, MaxAttempts: 1, BatchTimeout: 5 * time.Second, RequestTimeout: 5 * time.Second, ShutdownTimeout: 5 * time.Second},
		SpoolDir: filepath.Join(t.TempDir(), "spool"), Owner: "durable-cleanup-control", MaxSpoolBatches: 2, MaxSpoolBytes: 8192,
	}
	d, err := nvtelemetry.NewDurableDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chmod(filepath.Join(c.SpoolDir, "lease"), 0700)
		if err := d.Close(context.Background()); err != nil {
			t.Errorf("cleanup Close: %v", err)
		}
	})
	payload := []byte(`{"resourceMetrics":[{"resource":{"attributes":[]},"scopeMetrics":[{"scope":{"name":"cleanup-control"},"metrics":[{"name":"control","unit":"1","gauge":{"dataPoints":[{"timeUnixNano":"10","asDouble":1}]}}]}]}]}`)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := d.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.SucceededBatches != 1 || s.PendingSpool != 0 {
		t.Fatalf("not drained: %+v", s)
	}
	return d, c
}

func TestDurableDeliveryCleanupFailureRetainsLease(t *testing.T) {
	d, c := newCleanupDelivery(t)
	lease := filepath.Join(c.SpoolDir, "lease")
	if err := os.Chmod(lease, 0500); err != nil {
		t.Fatal(err)
	}
	err := d.Close(context.Background())
	if !errors.Is(err, os.ErrPermission) {
		t.Fatalf("Close after denied lease cleanup = %v, want permission error", err)
	}
	if _, err := os.Stat(filepath.Join(lease, "lock")); err != nil {
		t.Fatalf("lease lock was lost: %v", err)
	}
	if err := nvtelemetry.RecoverDurableDelivery(c.SpoolDir, c.Owner); err == nil {
		t.Fatal("recovery admitted while failed cleanup owns lease")
	}
	if next, err := nvtelemetry.NewDurableDelivery(c); err == nil {
		_ = next.Close(context.Background())
		t.Fatal("second writer admitted before cleanup retry")
	}
	if err := os.Chmod(lease, 0700); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
	if _, err := os.Stat(lease); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lease remains after retry: %v", err)
	}
	next, err := nvtelemetry.NewDurableDelivery(c)
	if err != nil {
		t.Fatalf("new writer after cleanup: %v", err)
	}
	if err := next.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestDurableDeliveryCleanupConcurrentClose(t *testing.T) {
	d, c := newCleanupDelivery(t)
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- d.Close(context.Background()) }()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, err := os.Stat(filepath.Join(c.SpoolDir, "lease")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lease remains: %v", err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatalf("repeat Close: %v", err)
	}
}

func TestDurableDeliveryCleanupRetriesPartialRemoval(t *testing.T) {
	d, c := newCleanupDelivery(t)
	lease := filepath.Join(c.SpoolDir, "lease")
	extra := filepath.Join(lease, "unrelated")
	const contents = "do not remove this file"
	if err := os.WriteFile(extra, []byte(contents), 0600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Remove(extra) })
	if err := d.Close(context.Background()); err == nil || !strings.Contains(err.Error(), "remove lease directory") {
		t.Fatalf("Close with nonempty lease = %v", err)
	}
	if b, err := os.ReadFile(extra); err != nil || string(b) != contents {
		t.Fatalf("unrelated file changed: %q, %v", b, err)
	}
	if _, err := os.Stat(filepath.Join(lease, "lock")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lock removal did not precede directory failure: %v", err)
	}
	if err := os.Remove(extra); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatalf("retry partial cleanup: %v", err)
	}
	if _, err := os.Stat(lease); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lease remains: %v", err)
	}
}
