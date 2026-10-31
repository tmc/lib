package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type heldReloadTransport struct {
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (t *heldReloadTransport) RoundTrip(*http.Request) (*http.Response, error) {
	close(t.entered)
	<-t.release
	return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`))}, nil
}
func (t *heldReloadTransport) unblock() { t.once.Do(func() { close(t.release) }) }

const reloadPayload = `{"resourceMetrics":[{"scopeMetrics":[{"metrics":[{"name":"power","unit":"W","gauge":{"dataPoints":[{"asDouble":46}]}}]}]}]}`

func heldDeliveryAgent(t *testing.T) (*Agent, *ParsedConfig, *nvtelemetry.Delivery, *heldReloadTransport, <-chan struct{}) {
	t.Helper()
	tr := &heldReloadTransport{entered: make(chan struct{}), release: make(chan struct{})}
	d, err := nvtelemetry.NewDelivery(nvtelemetry.DeliveryConfig{Endpoint: "http://offline.invalid/v1/metrics", Client: &http.Client{Transport: tr}, MaxQueuedBatches: 1, MaxQueuedBytes: 4096, MaxBatchBytes: 2048, MaxBatchDataPoints: 10, MaxResponseBytes: 1024, MaxAttempts: 1, BatchTimeout: time.Second, RequestTimeout: time.Second, ShutdownTimeout: 30 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		tr.unblock()
		select {
		case <-d.Done():
		case <-time.After(3 * time.Second):
			t.Error("delivery worker did not exit")
		}
	})
	if err := d.Enqueue(t.Context(), []byte(reloadPayload)); err != nil {
		t.Fatal(err)
	}
	select {
	case <-tr.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("transport did not start")
	}
	cfg := &ParsedConfig{ShutdownTimeout: 30 * time.Millisecond, MaxAge: time.Second, Interval: time.Hour, Budget: 10, Mock: true}
	a, err := NewAgent(cfg, "", io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	a.ctx, a.cancel = context.WithCancel(context.Background())
	done := make(chan struct{})
	close(done)
	cleaned := make(chan struct{})
	sampler := nvtelemetry.NewFakeSampler()
	collector := nvtelemetry.NewCollector("reload-test", sampler, nvtelemetry.RealClock{}, nvtelemetry.CollectorConfig{}, nil)
	old := &runningTarget{ctx: context.Background(), cancel: func() {}, collector: collector, collectorDone: done, metricsDelivery: d, metricsCleanup: func() { close(cleaned) }}
	old.stopping.Store(true)
	a.target = old
	t.Cleanup(func() { _ = a.Shutdown(context.Background()) })
	return a, cfg, d, tr, cleaned
}

func waitClosed(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatal("channel did not close")
	}
}
func assertOpen(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
		t.Fatal("channel closed prematurely")
	default:
	}
}

func TestReloadDeliveryQuiescence(t *testing.T) {
	a, cfg, d, tr, cleaned := heldDeliveryAgent(t)
	old := a.target
	err := a.ReloadConfig(cfg)
	if !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
		t.Fatalf("first reload error = %v", err)
	}
	if a.target != old {
		t.Fatal("old target dropped before delivery worker exited")
	}
	assertOpen(t, d.Done())
	assertOpen(t, cleaned)
	tr.unblock()
	waitClosed(t, d.Done())
	err = a.ReloadConfig(cfg)
	if !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
		t.Fatalf("acknowledgment error = %v", err)
	}
	waitClosed(t, cleaned)
	if a.target != nil {
		t.Fatal("failed but quiescent target was not retired")
	}
	started := make(chan struct{})
	var startedOnce sync.Once
	a.SetSamplerFactory(func(*ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		fake := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock", Model: "synthetic", Status: nvtelemetry.DeviceStatusOk})
		fake.SetFallback(func(context.Context) ([]nvtelemetry.Reading, error) {
			startedOnce.Do(func() { close(started) })
			return []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU-mock", Evidence: nvtelemetry.EvidenceDerived, Backend: "mock", Value: 30000, TimestampUnix: time.Now().UnixNano()}}, nil
		})
		return fake, nil, nil
	})
	if err := a.ReloadConfig(cfg); err != nil {
		t.Fatalf("subsequent reload: %v", err)
	}
	waitClosed(t, started)
	if a.target == nil {
		t.Fatal("replacement target not started")
	}
	if err := a.Shutdown(context.Background()); err != nil {
		t.Fatalf("shutdown replacement: %v", err)
	}
}

func TestShutdownRetainsDeliveryUntilExit(t *testing.T) {
	a, _, d, tr, cleaned := heldDeliveryAgent(t)
	if err := a.Shutdown(context.Background()); !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
		t.Fatalf("shutdown error = %v", err)
	}
	assertOpen(t, d.Done())
	assertOpen(t, cleaned)
	tr.unblock()
	waitClosed(t, d.Done())
	waitClosed(t, cleaned)
	if err := a.Shutdown(context.Background()); !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
		t.Fatalf("repeated shutdown lost failure: %v", err)
	}
}

func TestConcurrentReloadShutdownRetainsDelivery(t *testing.T) {
	a, cfg, d, tr, cleaned := heldDeliveryAgent(t)
	reloadDone := make(chan error, 1)
	go func() { reloadDone <- a.ReloadConfig(cfg) }()
	deadline := time.After(3 * time.Second)
	for {
		err := d.Enqueue(context.Background(), []byte(reloadPayload))
		if errors.Is(err, nvtelemetry.ErrDeliveryClosed) {
			break
		}
		if !errors.Is(err, nvtelemetry.ErrDeliveryFull) {
			t.Fatalf("waiting for reload admission close: %v", err)
		}
		select {
		case <-deadline:
			t.Fatal("reload did not close admission")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	shutdownDone := make(chan error, 1)
	go func() { shutdownDone <- a.Shutdown(context.Background()) }()
	select {
	case err := <-reloadDone:
		if !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
			t.Fatalf("reload: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("reload blocked")
	}
	select {
	case err := <-shutdownDone:
		if !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
			t.Fatalf("shutdown: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("shutdown blocked")
	}
	assertOpen(t, d.Done())
	assertOpen(t, cleaned)
	tr.unblock()
	waitClosed(t, d.Done())
	waitClosed(t, cleaned)
}

func TestReloadCancellationRetainsDelivery(t *testing.T) {
	a, _, d, tr, cleaned := heldDeliveryAgent(t)
	before, cancelBefore := context.WithCancel(context.Background())
	cancelBefore()
	if err := a.Reload(before); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled reload: %v", err)
	}
	assertOpen(t, d.Done())
	assertOpen(t, cleaned)
	ctx, cancel := context.WithCancel(context.Background())
	reloadDone := make(chan error, 1)
	go func() { reloadDone <- a.Reload(ctx) }()
	deadline := time.After(3 * time.Second)
	for {
		err := d.Enqueue(context.Background(), []byte(reloadPayload))
		if errors.Is(err, nvtelemetry.ErrDeliveryClosed) {
			break
		}
		if !errors.Is(err, nvtelemetry.ErrDeliveryFull) {
			t.Fatalf("waiting for admission close: %v", err)
		}
		select {
		case <-deadline:
			t.Fatal("reload did not close admission")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	select {
	case err := <-reloadDone:
		if !errors.Is(err, context.Canceled) || !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
			t.Fatalf("canceled reload: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("canceled reload blocked")
	}
	assertOpen(t, d.Done())
	assertOpen(t, cleaned)
	tr.unblock()
	waitClosed(t, d.Done())
}
