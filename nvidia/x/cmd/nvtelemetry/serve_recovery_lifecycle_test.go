package main

import (
	"bytes"
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

func recoveryLifecycle(t *testing.T) (*serveLifecycle, string, *ParsedConfig) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "retained")
	ret, err := nvtelemetry.OpenRetention(dir, nvtelemetry.RetentionConfig{Owner: "synthetic-test", MaxFiles: 8, MaxBytes: 8 << 20, MaxStreamBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ret.Close(); err != nil {
			t.Error(err)
		}
	})
	cfg, err := validateAndBuildParsedConfig(ServeConfig{Listen: "127.0.0.1:0", Interval: "1ms", MaxAge: "1s", Duration: "0s", Budget: 1000, ShutdownTimeout: "1s", Mock: true})
	if err != nil {
		t.Fatal(err)
	}
	return &serveLifecycle{retention: ret, maxStreamBytes: 1 << 20}, dir, cfg
}

func TestServeRecoveryHealthyGeneration(t *testing.T) {
	l, dir, cfg := recoveryLifecycle(t)
	sampler, cleanup, err := l.sampler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	for range 3 {
		if _, err := sampler.Sample(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	cleanup()
	cleanup() // generation cleanup is idempotent
	if err := l.result(); err != nil {
		t.Fatal(err)
	}
	if got := l.retention.Status().Stores; got != 1 {
		t.Fatalf("stores=%d", got)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	found := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		manifest, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 1<<20)
		if err != nil {
			continue
		}
		if manifest.TotalSamples != 3 || !manifest.IsComplete || manifest.LostRecords != 0 {
			t.Fatalf("manifest=%+v", manifest)
		}
		found++
	}
	if found != 1 {
		t.Fatalf("strict complete streams=%d", found)
	}
}

func TestServeRecoveryPublicReloadSealsEachGeneration(t *testing.T) {
	l, _, cfg := recoveryLifecycle(t)
	agent, err := NewAgent(cfg, "", io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	agent.ctx, agent.cancel = context.WithCancel(t.Context())
	var mu sync.Mutex
	var current *serveLifecycleSampler
	agent.SetSamplerFactory(func(c *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		sampler, cleanup, err := l.sampler(c)
		if err == nil {
			mu.Lock()
			current = sampler.(*serveLifecycleSampler)
			mu.Unlock()
		}
		return sampler, cleanup, err
	})
	defer func() {
		if err := agent.Shutdown(context.Background()); err != nil {
			t.Error(err)
		}
	}()
	waitSamples := func() {
		t.Helper()
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			mu.Lock()
			s := current
			mu.Unlock()
			if s != nil {
				s.mu.Lock()
				n := s.sampleCount
				s.mu.Unlock()
				if n >= 2 {
					return
				}
			}
			time.Sleep(time.Millisecond)
		}
		t.Fatal("synthetic generation did not acquire two samples")
	}
	if err := agent.ReloadConfig(cfg); err != nil {
		t.Fatal(err)
	}
	waitSamples()
	next := *cfg
	next.Interval = 2 * time.Millisecond
	if err := agent.ReloadConfig(&next); err != nil {
		t.Fatal(err)
	}
	if stores := l.retention.Status().Stores; stores != 1 {
		t.Fatalf("old generation not stored before replacement: %d", stores)
	}
	waitSamples()
	if err := agent.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := l.result(); err != nil {
		t.Fatal(err)
	}
	if stores := l.retention.Status().Stores; stores != 2 {
		t.Fatalf("stored=%d", stores)
	}
}

type recoveryTransport func(*http.Request) (*http.Response, error)

func (f recoveryTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestServeRecoveryExportFailureSurvivesDrainedQueue(t *testing.T) {
	cfg := nvtelemetry.DurableDeliveryConfig{SpoolDir: filepath.Join(t.TempDir(), "spool"), Owner: "synthetic-test", MaxSpoolBatches: 2, MaxSpoolBytes: 1 << 20, Delivery: nvtelemetry.DeliveryConfig{Endpoint: "http://synthetic.invalid/v1/metrics", Client: &http.Client{Transport: recoveryTransport(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("{}"))}, nil
	})}, MaxQueuedBatches: 2, MaxQueuedBytes: 1 << 20, MaxBatchBytes: 1 << 20, MaxBatchDataPoints: 100, MaxResponseBytes: 4096, MaxAttempts: 1, InitialBackoff: time.Millisecond, MaxBackoff: time.Millisecond, BatchTimeout: time.Second, RequestTimeout: time.Second, ShutdownTimeout: time.Second}}
	d, err := nvtelemetry.NewDurableDelivery(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close(context.Background())
	l := &serveLifecycle{metrics: d}
	base := nvtelemetry.NewFakeSampler()
	base.Enqueue([]nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, Value: 123, Evidence: nvtelemetry.EvidenceMeasured, Backend: "mock", DeviceUUID: "GPU-synthetic", TimestampUnix: time.Now().UnixNano()}})
	base.Enqueue([]nvtelemetry.Reading{{Name: nvtelemetry.MetricTotalEnergyMJ, Unit: nvtelemetry.UnitMillijoules, Value: 1, Evidence: nvtelemetry.EvidenceMeasured, Backend: "mock", DeviceUUID: "GPU-synthetic", TimestampUnix: time.Now().UnixNano()}})
	s := &serveLifecycleSampler{base: base, durableMetrics: d, failure: l.fail}
	if _, err := s.Sample(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Sample(t.Context()); err == nil {
		t.Fatal("counter without epoch accepted")
	}
	if err := d.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if d.Stats().SucceededBatches != 1 {
		t.Fatalf("successful controlled sink deliveries=%+v", d.Stats())
	}
	if d.Stats().PendingSpool != 0 {
		t.Fatal("expected drained queue")
	}
	if l.result() == nil {
		t.Fatal("encoding failure erased by empty queue")
	}
	// A later admission error is independently observable even after the real
	// worker drained and closed successfully. No socket or listener is used.
	closed := &serveLifecycle{}
	s.failure = closed.fail
	base.Enqueue([]nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, Value: 5, Evidence: nvtelemetry.EvidenceMeasured, Backend: "mock", DeviceUUID: "GPU-synthetic", TimestampUnix: time.Now().UnixNano()}})
	if _, err := s.Sample(t.Context()); !errors.Is(err, nvtelemetry.ErrDeliveryClosed) {
		t.Fatalf("admission error=%v", err)
	}
	if !errors.Is(closed.result(), nvtelemetry.ErrDeliveryClosed) {
		t.Fatalf("retained admission error=%v", closed.result())
	}

}

func TestServeRecoveryRetainedBoundAndFailure(t *testing.T) {
	l, _, cfg := recoveryLifecycle(t)
	cfg.Budget = 2
	sampler, cleanup, err := l.sampler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	if _, _, err := l.sampler(cfg); err == nil {
		t.Fatal("overlapping retained generation admitted")
	}
	for range 2 {
		if _, err := sampler.Sample(t.Context()); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := sampler.Sample(t.Context()); !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("bound error=%v", err)
	}
	cleanup()
	if l.result() == nil {
		t.Fatal("overflow reported complete")
	}
	if l.retention.Status().Stores != 0 {
		t.Fatal("incomplete generation published")
	}
}

func TestServeRecoveryStoreFailureRetainsGeneration(t *testing.T) {
	l, _, cfg := recoveryLifecycle(t)
	sampler, cleanup, err := l.sampler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sampler.Sample(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := l.retention.Close(); err != nil {
		t.Fatal(err)
	}
	cleanup()
	if !l.drained() || l.result() == nil {
		t.Fatal("failed store confused drained source with failed persistence")
	}
	if l.retained != sampler || l.retained.buffer.Len() == 0 {
		t.Fatal("failed generation bytes discarded")
	}
	if _, _, err := l.sampler(cfg); err == nil {
		t.Fatal("replacement overwrote failed generation")
	}
}

func TestServeRecoveryCompletedRefusesFailures(t *testing.T) {
	for _, m := range []ServeRecoveryManifest{
		{SchemaVersion: ServeRecoverySchemaVersion, Owner: "synthetic", Completed: true, LifecycleFailures: 1},
		{SchemaVersion: ServeRecoverySchemaVersion, Owner: "synthetic", Completed: true, MetricsFinal: &nvtelemetry.DurableDeliveryStats{DeliveryStats: nvtelemetry.DeliveryStats{DroppedBatches: 1}}},
		{SchemaVersion: ServeRecoverySchemaVersion, Owner: "synthetic", Completed: true, LogsFinal: &nvtelemetry.DurableDeliveryStats{PendingSpool: 1}},
	} {
		if err := WriteServeRecoveryManifest(io.Discard, m); err == nil {
			t.Fatal("invalid complete receipt admitted")
		}
		m.Completed = false
		var buf bytes.Buffer
		if err := WriteServeRecoveryManifest(&buf, m); err != nil {
			t.Fatal(err)
		}
		got, err := ReadServeRecoveryManifest(&buf)
		if err != nil || got.Completed {
			t.Fatalf("failure receipt=%+v %v", got, err)
		}
	}
}

func TestServeRecoveryRejectsUnboundedGeneration(t *testing.T) {
	l, _, cfg := recoveryLifecycle(t)
	for _, budget := range []int{-1, 0, 1<<20 + 1} {
		copy := *cfg
		copy.Budget = budget
		if _, _, err := l.sampler(&copy); err == nil {
			t.Fatalf("unbounded budget admitted: %d", budget)
		}
	}
	if !l.drained() || l.retention.Status().Stores != 0 {
		t.Fatal("invalid generation acquired retention ownership")
	}
}
