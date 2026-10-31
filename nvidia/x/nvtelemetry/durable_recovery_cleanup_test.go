//go:build darwin || linux

package nvtelemetry_test

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type recoveryCleanupTransport struct{}

func (recoveryCleanupTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("unexpected transport in recovery-only test")
}

func recoveryCleanupWriter(t *testing.T) (*nvtelemetry.DurableDelivery, nvtelemetry.DurableDeliveryConfig) {
	t.Helper()
	c := nvtelemetry.DurableDeliveryConfig{Delivery: nvtelemetry.DeliveryConfig{Endpoint: "https://recovery.invalid/v1/metrics", Client: &http.Client{Transport: recoveryCleanupTransport{}}, MaxQueuedBatches: 1, MaxQueuedBytes: 4096, MaxBatchBytes: 4096, MaxBatchDataPoints: 4, MaxResponseBytes: 512, MaxAttempts: 1, BatchTimeout: time.Second, RequestTimeout: time.Second, ShutdownTimeout: time.Second}, SpoolDir: filepath.Join(t.TempDir(), "spool"), Owner: "recovery-cleanup-control", MaxSpoolBatches: 2, MaxSpoolBytes: 8192}
	d, err := nvtelemetry.NewDurableDelivery(c)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := d.Close(context.Background()); err != nil {
			t.Errorf("writer cleanup: %v", err)
		}
	})
	return d, c
}

func recoveryCleanupStaleLease(t *testing.T, c nvtelemetry.DurableDeliveryConfig) string {
	t.Helper()
	lease := filepath.Join(c.SpoolDir, "lease")
	if err := os.Mkdir(lease, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(lease, "lock"), []byte("constructed abandoned task lock\n"), 0600); err != nil {
		t.Fatal(err)
	}
	return lease
}

func TestDurableRecoveryCleanupFailureReportsAndRetries(t *testing.T) {
	for _, wrapper := range []bool{false, true} {
		name := "manifest"
		if wrapper {
			name = "error-only"
		}
		t.Run(name, func(t *testing.T) {
			d, c := recoveryCleanupWriter(t)
			if err := d.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
			marker := filepath.Join(c.SpoolDir, "owner.json")
			before, err := os.ReadFile(marker)
			if err != nil {
				t.Fatal(err)
			}
			lease := recoveryCleanupStaleLease(t, c)
			t.Cleanup(func() { _ = os.Chmod(lease, 0700); _ = nvtelemetry.RecoverDurableDelivery(c.SpoolDir, c.Owner) })
			if err := os.Chmod(lease, 0500); err != nil {
				t.Fatal(err)
			}
			var m nvtelemetry.DurableDeliveryRecoveryManifest
			if wrapper {
				err = nvtelemetry.RecoverDurableDelivery(c.SpoolDir, c.Owner)
			} else {
				m, err = nvtelemetry.RecoverDurableDeliveryManifest(c.SpoolDir, c.Owner)
			}
			if !errors.Is(err, os.ErrPermission) {
				t.Fatalf("recovery with denied lock removal = %+v, %v; want permission error", m, err)
			}
			if m != (nvtelemetry.DurableDeliveryRecoveryManifest{}) {
				t.Fatalf("failed recovery returned success manifest: %+v", m)
			}
			if _, err := os.Stat(filepath.Join(lease, "lock")); err != nil {
				t.Fatalf("failed cleanup lost lock artifact: %v", err)
			}
			if err := os.Chmod(lease, 0700); err != nil {
				t.Fatal(err)
			}
			m, err = nvtelemetry.RecoverDurableDeliveryManifest(c.SpoolDir, c.Owner)
			if err != nil || !m.RecoveredLease {
				t.Fatalf("explicit recovery retry = %+v, %v", m, err)
			}
			if _, err := os.Stat(lease); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("lease remains: %v", err)
			}
			after, err := os.ReadFile(marker)
			if err != nil || string(after) != string(before) {
				t.Fatalf("owned marker changed: %q, %v", after, err)
			}
			next, err := nvtelemetry.NewDurableDelivery(c)
			if err != nil {
				t.Fatalf("writer after recovery: %v", err)
			}
			if err := next.Close(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDurableRecoveryCleanupPartialRemoval(t *testing.T) {
	d, c := recoveryCleanupWriter(t)
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	lease := recoveryCleanupStaleLease(t, c)
	extra := filepath.Join(lease, "unrelated")
	const contents = "keep this task-planted control"
	if err := os.WriteFile(extra, []byte(contents), 0600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Remove(extra); _ = nvtelemetry.RecoverDurableDelivery(c.SpoolDir, c.Owner) })
	m, err := nvtelemetry.RecoverDurableDeliveryManifest(c.SpoolDir, c.Owner)
	if err == nil || !strings.Contains(err.Error(), "remove recovery lease directory") {
		t.Fatalf("nonempty cleanup = %+v, %v", m, err)
	}
	if m != (nvtelemetry.DurableDeliveryRecoveryManifest{}) {
		t.Fatalf("failed cleanup returned success manifest: %+v", m)
	}
	if got, err := os.ReadFile(extra); err != nil || string(got) != contents {
		t.Fatalf("unrelated file changed: %q, %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(lease, "lock")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lock removal did not occur: %v", err)
	}
	if err := os.Remove(extra); err != nil {
		t.Fatal(err)
	}
	m, err = nvtelemetry.RecoverDurableDeliveryManifest(c.SpoolDir, c.Owner)
	if err != nil || !m.RecoveredLease {
		t.Fatalf("retry partial recovery = %+v, %v", m, err)
	}
}

func TestDurableRecoveryCleanupPreservesLiveWriter(t *testing.T) {
	d, c := recoveryCleanupWriter(t)
	lock := filepath.Join(c.SpoolDir, "lease", "lock")
	before, err := os.Stat(lock)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nvtelemetry.RecoverDurableDeliveryManifest(c.SpoolDir, c.Owner); err == nil {
		t.Fatal("recovery admitted live writer")
	}
	after, err := os.Stat(lock)
	if err != nil || !os.SameFile(before, after) {
		t.Fatalf("live writer lock changed: %v", err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	m, err := nvtelemetry.RecoverDurableDeliveryManifest(c.SpoolDir, c.Owner)
	if err != nil || m.RecoveredLease {
		t.Fatalf("recovery after clean writer = %+v, %v", m, err)
	}
}
