package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func durableDeliveryTestConfig(server *httptest.Server, spoolDir string) DurableDeliveryConfig {
	return DurableDeliveryConfig{
		Delivery: DeliveryConfig{
			Endpoint:           server.URL + "/v1/metrics",
			Client:             server.Client(),
			MaxQueuedBatches:   4,
			MaxQueuedBytes:     8192,
			MaxBatchBytes:      4096,
			MaxBatchDataPoints: 4,
			MaxResponseBytes:   512,
			MaxAttempts:        4,
			InitialBackoff:     time.Millisecond,
			MaxBackoff:         2 * time.Millisecond,
			BatchTimeout:       5 * time.Second,
			RequestTimeout:     2 * time.Second,
			ShutdownTimeout:    5 * time.Second,
		},
		SpoolDir:        spoolDir,
		Owner:           "task-delivery-test",
		MaxSpoolBatches: 8,
		MaxSpoolBytes:   32768,
	}
}

func TestDurableDeliveryLoopbackFailureAndRecovery(t *testing.T) {
	var attempts atomic.Int32
	var gotMu sync.Mutex
	var got [][]byte

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/metrics" || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("unexpected request header or path: %s", r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		gotMu.Lock()
		got = append(got, body)
		gotMu.Unlock()

		n := attempts.Add(1)
		if n <= 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, "temporarily unavailable")
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir := filepath.Join(t.TempDir(), "spool")
	cfg := durableDeliveryTestConfig(s, spoolDir)

	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte(deliveryFixture)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatal(err)
	}

	// Verify the batch was committed to disk before Close/drain
	stats := d.Stats()
	if stats.SpooledBatches != 1 || stats.SpooledBytes != uint64(len(payload)) {
		t.Fatalf("spool stats after enqueue: %+v", stats)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := d.Recover(ctx); err != nil {
		t.Fatalf("recover failed: %v", err)
	}

	if err := d.Close(context.Background()); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	stats = d.Stats()
	if stats.SucceededBatches != 1 || stats.RetryAttempts < 2 || stats.OutageResponses < 2 {
		t.Fatalf("expected retry and success, got stats=%+v", stats)
	}
	if stats.PendingSpool != 0 || stats.PendingSpoolBytes != 0 {
		t.Fatalf("pending spool not drained: %+v", stats)
	}

	// Spool file should have been deleted from disk upon success
	entries, err := os.ReadDir(spoolDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "spool-") {
			t.Fatalf("spool file %s was not deleted after successful delivery", e.Name())
		}
	}

	gotMu.Lock()
	defer gotMu.Unlock()
	if len(got) != 3 {
		t.Fatalf("expected 3 server attempts, got %d", len(got))
	}
	if !bytes.Equal(got[len(got)-1], payload) {
		t.Fatal("delivered payload does not match enqueued payload")
	}
}

func TestDurableDeliverySpoolOverflowAndDropCounters(t *testing.T) {
	// Server that always returns 503 to keep batches queued/retrying
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer s.Close()

	spoolDir := filepath.Join(t.TempDir(), "spool-overflow")
	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.MaxSpoolBatches = 2
	cfg.Delivery.InitialBackoff = 50 * time.Millisecond
	cfg.Delivery.MaxBackoff = 100 * time.Millisecond

	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		_ = d.Close(ctx)
	}()

	payload := []byte(deliveryFixture)

	// First two should succeed admission
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatalf("batch 1 enqueue: %v", err)
	}
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatalf("batch 2 enqueue: %v", err)
	}

	// Third enqueue must fail immediately with ErrDeliveryFull (spool budget exhausted)
	err = d.Enqueue(context.Background(), payload)
	if !errors.Is(err, ErrDeliveryFull) {
		t.Fatalf("expected ErrDeliveryFull, got %v", err)
	}

	stats := d.Stats()
	if stats.DroppedBatches != 1 {
		t.Fatalf("expected 1 dropped batch (producer loss), got %d", stats.DroppedBatches)
	}
	if stats.FailedBatches != 0 {
		t.Fatalf("expected 0 failed batches (delivery loss), got %d", stats.FailedBatches)
	}
	if stats.SpooledBatches != 2 {
		t.Fatalf("expected 2 spooled batches, got %d", stats.SpooledBatches)
	}

	// Verify disk has at most 2 spool files
	entries, err := os.ReadDir(spoolDir)
	if err != nil {
		t.Fatal(err)
	}
	spoolFileCount := 0
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "spool-") {
			spoolFileCount++
		}
	}
	if spoolFileCount > 2 {
		t.Fatalf("disk contains %d spool files, expected <= 2", spoolFileCount)
	}
}

func TestDurableDeliveryCrashRestartRecovery(t *testing.T) {
	spoolDir := filepath.Join(t.TempDir(), "spool-crash")

	// Dummy failing server for instance 1 to accumulate spooled batches
	dummyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))

	cfg1 := durableDeliveryTestConfig(dummyServer, spoolDir)
	cfg1.Delivery.InitialBackoff = 50 * time.Millisecond
	cfg1.Delivery.MaxBackoff = 100 * time.Millisecond

	d1, err := NewDurableDelivery(cfg1)
	if err != nil {
		t.Fatal(err)
	}

	batch1 := []byte(`{"resourceMetrics":[{"scopeMetrics":[{"metrics":[{"name":"nvtelemetry.power","unit":"W","gauge":{"dataPoints":[{"timeUnixNano":"1","asDouble":10}]}}]}]}]}`)
	batch2 := []byte(`{"resourceMetrics":[{"scopeMetrics":[{"metrics":[{"name":"nvtelemetry.power","unit":"W","gauge":{"dataPoints":[{"timeUnixNano":"2","asDouble":20}]}}]}]}]}`)

	if err := d1.Enqueue(context.Background(), batch1); err != nil {
		t.Fatalf("batch 1 enqueue: %v", err)
	}
	if err := d1.Enqueue(context.Background(), batch2); err != nil {
		t.Fatalf("batch 2 enqueue: %v", err)
	}

	// Close instance 1 with short timeout to simulate crash/interrupted shutdown while server is down
	closeCtx, closeCancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	closeErr := d1.Close(closeCtx)
	closeCancel()
	if !errors.Is(closeErr, ErrDeliveryIncomplete) {
		t.Fatalf("expected ErrDeliveryIncomplete from timed-out close, got %v", closeErr)
	}
	dummyServer.Close()
	<-d1.done

	// Verify spooled files are intact on disk
	entries, err := os.ReadDir(spoolDir)
	if err != nil {
		t.Fatal(err)
	}
	spoolFiles := 0
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "spool-") && strings.HasSuffix(e.Name(), ".json") {
			spoolFiles++
		}
	}
	if spoolFiles != 2 {
		t.Fatalf("expected 2 spool files on disk after shutdown, found %d", spoolFiles)
	}

	// Now start real server that accepts records
	var deliveredMu sync.Mutex
	var delivered [][]byte
	realServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		deliveredMu.Lock()
		delivered = append(delivered, b)
		deliveredMu.Unlock()
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer realServer.Close()

	// Create new delivery instance pointing to realServer with the same spoolDir
	cfg2 := durableDeliveryTestConfig(realServer, spoolDir)
	d2, err := NewDurableDelivery(cfg2)
	if err != nil {
		t.Fatalf("failed to create new delivery instance for recovery: %v", err)
	}

	stats2 := d2.Stats()
	if stats2.RecoveredBatches != 2 || stats2.PendingSpool != 2 {
		t.Fatalf("expected 2 recovered batches, got %+v", stats2)
	}

	// Recover and flush pending spooled records
	recoverCtx, recoverCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer recoverCancel()
	if err := d2.Recover(recoverCtx); err != nil {
		t.Fatalf("recovery flush failed: %v", err)
	}

	if err := d2.Close(context.Background()); err != nil {
		t.Fatalf("close d2 failed: %v", err)
	}

	stats2 = d2.Stats()
	if stats2.SucceededBatches != 2 || stats2.PendingSpool != 0 {
		t.Fatalf("expected 2 succeeded batches and 0 pending, got %+v", stats2)
	}

	deliveredMu.Lock()
	defer deliveredMu.Unlock()
	if len(delivered) != 2 {
		t.Fatalf("expected 2 delivered batches, got %d", len(delivered))
	}
	if !bytes.Equal(delivered[0], batch1) || !bytes.Equal(delivered[1], batch2) {
		t.Fatal("delivered batches did not preserve FIFO order")
	}

	// Verify all spool files were cleanly deleted
	entriesAfter, err := os.ReadDir(spoolDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entriesAfter {
		if strings.HasPrefix(e.Name(), "spool-") {
			t.Fatalf("spool file %s remaining after recovery", e.Name())
		}
	}
}

func TestDurableDeliveryUngracefulCrashAndRecoverHelper(t *testing.T) {
	spoolDir := filepath.Join(t.TempDir(), "spool-ungraceful")
	owner := "task-ungraceful"

	// Initialize directory by creating first instance
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.Owner = owner
	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Enqueue a batch
	payload := []byte(deliveryFixture)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatal(err)
	}

	// Wait for delivery
	_ = d.Recover(context.Background())
	_ = d.Close(context.Background())

	// Simulate ungraceful crash: writer died leaving lease directory and partial temporary file
	if err := os.Mkdir(filepath.Join(spoolDir, "lease"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(spoolDir, "pending-0000000000099.tmp"), []byte("partial crash data"), 0600); err != nil {
		t.Fatal(err)
	}

	// Attempting to open without recovery should fail because lease exists
	if _, err := NewDurableDelivery(cfg); err == nil {
		t.Fatal("expected lease error when lease exists")
	}

	// Foreign owner recovery must be rejected
	if err := RecoverDurableDelivery(spoolDir, "intruder-owner"); err == nil {
		t.Fatal("expected foreign owner recovery to be refused")
	}

	// Legitimate owner recovery must succeed
	if err := RecoverDurableDelivery(spoolDir, owner); err != nil {
		t.Fatalf("valid recovery failed: %v", err)
	}

	// Partial file should have been cleaned up
	if _, err := os.Stat(filepath.Join(spoolDir, "pending-0000000000099.tmp")); !os.IsNotExist(err) {
		t.Fatal("stale pending.tmp was not cleaned up during recovery")
	}

	// New instance can now open successfully
	d2, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatalf("failed to open after recovery: %v", err)
	}
	_ = d2.Close(context.Background())
}

func TestDurableDeliveryForeignFileAndSymlinkRefusal(t *testing.T) {
	spoolDir := filepath.Join(t.TempDir(), "spool-security")
	owner := "task-security"

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.Owner = owner

	// Initialize directory
	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	_ = d.Close(context.Background())

	// 1. Foreign file refusal
	foreignPath := filepath.Join(spoolDir, "foreign_script.sh")
	if err := os.WriteFile(foreignPath, []byte("echo bad"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := NewDurableDelivery(cfg); err == nil {
		t.Fatal("expected foreign file refusal")
	}
	if err := RecoverDurableDelivery(spoolDir, owner); err == nil {
		t.Fatal("expected foreign file refusal in RecoverDurableDelivery")
	}
	_ = os.Remove(foreignPath)

	// 2. Symlink refusal
	symlinkTarget := filepath.Join(t.TempDir(), "target")
	_ = os.WriteFile(symlinkTarget, []byte("target"), 0600)
	symlinkPath := filepath.Join(spoolDir, "spool-symlink")
	if err := os.Symlink(symlinkTarget, symlinkPath); err != nil {
		t.Fatal(err)
	}
	if _, err := NewDurableDelivery(cfg); err == nil {
		t.Fatal("expected symlink refusal")
	}
	if err := RecoverDurableDelivery(spoolDir, owner); err == nil {
		t.Fatal("expected symlink refusal in RecoverDurableDelivery")
	}
	_ = os.Remove(symlinkPath)

	// 3. Spool directory permission refusal
	if err := os.Chmod(spoolDir, 0777); err != nil {
		t.Fatal(err)
	}
	if _, err := NewDurableDelivery(cfg); err == nil {
		t.Fatal("expected non-private directory refusal")
	}
	_ = os.Chmod(spoolDir, 0700)

	// After cleanup, opening succeeds
	d2, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatalf("failed to open clean spool: %v", err)
	}
	_ = d2.Close(context.Background())
}

func TestDurableDeliveryAuthHeaderPrivacy(t *testing.T) {
	secretToken := "secret-token-xyz-12345-do-not-leak"
	secretEndpoint := "http://127.0.0.1:1/invalid/endpoint"

	spoolDir := filepath.Join(t.TempDir(), "spool-privacy")
	cfg := DurableDeliveryConfig{
		Delivery: DeliveryConfig{
			Endpoint:           secretEndpoint,
			Client:             &http.Client{Timeout: 100 * time.Millisecond},
			Headers:            http.Header{"Authorization": []string{"Bearer " + secretToken}},
			MaxQueuedBatches:   2,
			MaxQueuedBytes:     4096,
			MaxBatchBytes:      2048,
			MaxBatchDataPoints: 2,
			MaxResponseBytes:   256,
			MaxAttempts:        2,
			InitialBackoff:     time.Millisecond,
			MaxBackoff:         2 * time.Millisecond,
			BatchTimeout:       200 * time.Millisecond,
			RequestTimeout:     100 * time.Millisecond,
			ShutdownTimeout:    200 * time.Millisecond,
		},
		SpoolDir:        spoolDir,
		Owner:           "task-privacy",
		MaxSpoolBatches: 4,
		MaxSpoolBytes:   8192,
	}

	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte(deliveryFixture)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatal(err)
	}

	// Close will fail because secretEndpoint is invalid / unreachable
	closeCtx, closeCancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	closeErr := d.Close(closeCtx)
	closeCancel()

	if closeErr != nil {
		msg := closeErr.Error()
		if strings.Contains(msg, secretToken) {
			t.Fatalf("secret authorization token leaked in error message: %s", msg)
		}
	}

	stats := d.Stats()
	if stats.AuthFailures != 0 && stats.TransportFailures == 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestDurableDeliveryRace(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir := filepath.Join(t.TempDir(), "spool-race")
	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.MaxSpoolBatches = 64
	cfg.MaxSpoolBytes = 128 * 1024

	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte(deliveryFixture)
	var wg sync.WaitGroup

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				_ = d.Enqueue(context.Background(), payload)
				_ = d.Stats()
			}
		}()
	}

	wg.Wait()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := d.Recover(ctx); err != nil {
		t.Fatalf("recover race test: %v", err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatalf("close race test: %v", err)
	}
}

func TestDurableDeliveryLogs(t *testing.T) {
	var gotMu sync.Mutex
	var got [][]byte
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		gotMu.Lock()
		got = append(got, b)
		gotMu.Unlock()
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir := filepath.Join(t.TempDir(), "spool-logs")
	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.Delivery.Endpoint = s.URL + "/v1/logs"

	d, err := NewLogsDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}

	logsPayload := []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{"timeUnixNano":"100","body":{"stringValue":"gpu alert"}}]}]}]}`)
	if err := d.Enqueue(context.Background(), logsPayload); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := d.Recover(ctx); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(context.Background()); err != nil {
		t.Fatal(err)
	}

	stats := d.Stats()
	if stats.SucceededBatches != 1 {
		t.Fatalf("expected 1 succeeded log batch, got %+v", stats)
	}
	gotMu.Lock()
	defer gotMu.Unlock()
	if len(got) != 1 || !bytes.Equal(got[0], logsPayload) {
		t.Fatal("logs payload mismatch")
	}
}

func TestDurableDeliveryConcurrentCapacity(t *testing.T) {
	release := make(chan struct{})
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()
	defer close(release)

	payload := []byte(deliveryFixture)
	spoolDir := filepath.Join(t.TempDir(), "spool-concurrent-capacity")
	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.Delivery.MaxBatchBytes = len(payload)
	cfg.MaxSpoolBatches = 1
	cfg.MaxSpoolBytes = int64(len(payload))

	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = d.Close(ctx)
	}()

	start := make(chan struct{})
	var wg sync.WaitGroup
	errs := make(chan error, 64)

	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs <- d.Enqueue(context.Background(), payload)
		}()
	}

	close(start)
	wg.Wait()
	close(errs)

	admitted := 0
	for err := range errs {
		if err == nil {
			admitted++
		} else if !errors.Is(err, ErrDeliveryFull) {
			t.Errorf("unexpected enqueue error: %v", err)
		}
	}

	if admitted != 1 || d.Stats().PendingSpool != 1 || d.Stats().PendingSpoolBytes != int64(len(payload)) {
		t.Fatalf("unreserved concurrent admissions: admitted=%d stats=%+v", admitted, d.Stats())
	}
}

func TestDurableDeliveryLiveLeaseRefusal(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir := filepath.Join(t.TempDir(), "spool-live-lease")
	cfg := durableDeliveryTestConfig(s, spoolDir)

	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = d.Close(ctx)
	}()

	// Second writer without recovery must be rejected
	if other, err := NewDurableDelivery(cfg); err == nil {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = other.Close(ctx)
		t.Fatal("second writer admitted without recovery while first is running")
	}

	// Recovery while writer is live must fail
	if err := RecoverDurableDelivery(cfg.SpoolDir, cfg.Owner); err == nil {
		t.Fatal("recovery removed an actively held live writer lease")
	}
}

func TestDurableDeliveryRecoveryScanBound(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir := filepath.Join(t.TempDir(), "spool-scan-bound")
	if err := os.Mkdir(spoolDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(spoolDir, "owner.json"), []byte(`{"version":1,"owner":"scan-bound-owner"}`), 0600); err != nil {
		t.Fatal(err)
	}

	payload := []byte(deliveryFixture)
	for i := 1; i <= 4097; i++ {
		name := filepath.Join(spoolDir, fmt.Sprintf("spool-%013d.json", i))
		if err := os.WriteFile(name, payload, 0600); err != nil {
			t.Fatal(err)
		}
	}

	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.Owner = "scan-bound-owner"
	cfg.MaxSpoolBatches = 4096
	cfg.MaxSpoolBytes = int64(len(payload) * 4097)

	if d, err := NewDurableDelivery(cfg); err == nil {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = d.Close(ctx)
		t.Fatalf("expected NewDurableDelivery to fail when 4097 disk records exceed limit 4096")
	}
}

func TestDurableDeliveryDiskRemovalFailure(t *testing.T) {
	spoolDir := filepath.Join(t.TempDir(), "spool-removal-failure")
	diskErr := make(chan error, 1)

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		diskErr <- os.Chmod(spoolDir, 0500)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	cfg := durableDeliveryTestConfig(s, spoolDir)
	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = os.Chmod(spoolDir, 0700)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = d.Close(ctx)
	}()

	payload := []byte(deliveryFixture)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	recoverErr := d.Recover(ctx)

	if err := <-diskErr; err != nil {
		t.Fatal(err)
	}

	_, statErr := os.Stat(filepath.Join(spoolDir, "spool-0000000000001.json"))
	_ = os.Chmod(spoolDir, 0700)

	if recoverErr == nil || d.Stats().PendingSpool == 0 {
		t.Fatalf("disk removal failure discarded: recoverErr=%v retainedFile=%v stats=%+v", recoverErr, statErr == nil, d.Stats())
	}
}

func TestDurableDeliveryIDExhaustion(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{}`)
	}))
	defer s.Close()

	spoolDir := filepath.Join(t.TempDir(), "spool-id-exhaustion")
	if err := os.Mkdir(spoolDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(spoolDir, "owner.json"), []byte(`{"version":1,"owner":"id-owner"}`), 0600); err != nil {
		t.Fatal(err)
	}
	payload := []byte(deliveryFixture)
	if err := os.WriteFile(filepath.Join(spoolDir, "spool-9999999999999.json"), payload, 0600); err != nil {
		t.Fatal(err)
	}

	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.Owner = "id-owner"
	cfg.Delivery.MaxBatchBytes = len(payload)
	cfg.MaxSpoolBatches = 2
	cfg.MaxSpoolBytes = int64(2 * len(payload))

	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = d.Close(ctx)
	}()

	if err := d.Enqueue(context.Background(), payload); err == nil {
		_, statErr := os.Stat(filepath.Join(spoolDir, "spool-10000000000000.json"))
		t.Fatalf("expected ID exhaustion error, but 14-digit file was admitted (exists=%v)", statErr == nil)
	}
}

type heldTransport struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (h *heldTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	h.once.Do(func() { close(h.entered) })
	<-h.release
	return &http.Response{
		StatusCode: 401,
		Body:       io.NopCloser(strings.NewReader(`{}`)),
		Header:     make(http.Header),
		Request:    r,
	}, nil
}

func TestDurableDeliveryLeaseRetainedUntilWorkerQuiesces(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer s.Close()
	h := &heldTransport{entered: make(chan struct{}), release: make(chan struct{})}
	defer close(h.release)
	spoolDir := filepath.Join(t.TempDir(), "spool-lease-retention")
	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.Delivery.Client = &http.Client{Transport: h}
	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	payload := []byte(deliveryFixture)
	if err := d.Enqueue(context.Background(), payload); err != nil {
		t.Fatal(err)
	}
	select {
	case <-h.entered:
	case <-time.After(time.Second):
		t.Fatal("worker not entered")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := d.Close(ctx); !errors.Is(err, ErrDeliveryIncomplete) {
		t.Fatalf("close=%v", err)
	}
	// The previous worker is demonstrably still inside its transport.
	if next, err := NewDurableDelivery(cfg); err == nil {
		ctx2, cancel2 := context.WithCancel(context.Background())
		cancel2()
		_ = next.Close(ctx2)
		t.Fatal("second writer admitted before first worker quiesced")
	}
}

func TestDurableDeliveryLiveRecoveryDoesNotDeletePending(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer s.Close()
	spoolDir := filepath.Join(t.TempDir(), "spool-live-recovery")
	cfg := durableDeliveryTestConfig(s, spoolDir)
	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_ = d.Close(ctx)
	}()
	pending := filepath.Join(cfg.SpoolDir, "pending-0000000000001.tmp")
	payload := []byte(deliveryFixture)
	if err := os.WriteFile(pending, payload, 0600); err != nil {
		t.Fatal(err)
	}
	if err := RecoverDurableDelivery(cfg.SpoolDir, cfg.Owner); err == nil {
		t.Fatal("live recovery admitted")
	}
	if _, err := os.Stat(pending); err != nil {
		t.Fatalf("refused live recovery mutated pending record: %v", err)
	}
}

func TestDurableDeliveryCloseQuiescesActiveAdmissions(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Millisecond)
	}))
	defer s.Close()
	spoolDir := filepath.Join(t.TempDir(), "spool-admission-quiesce")
	cfg := durableDeliveryTestConfig(s, spoolDir)
	cfg.MaxSpoolBatches = 100
	d, err := NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}

	payload := []byte(deliveryFixture)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = d.Enqueue(context.Background(), payload)
		}()
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	closeErr := d.Close(ctx)
	wg.Wait()

	stats := d.Stats()
	if closeErr == nil && stats.PendingSpool != 0 {
		t.Fatalf("close returned nil with pending spool records: %+v", stats)
	}
}
