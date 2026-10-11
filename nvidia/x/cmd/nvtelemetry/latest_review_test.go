package main

import (
	"context"
	"errors"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestLatestRollbackFailureTerminates(t *testing.T) {
	cfg := independentCfg()
	var acquisitions atomic.Int32
	rollback := errors.New("rollback unavailable")
	errReplacement := errors.New("replacement unavailable")
	factory := func(*ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		n := acquisitions.Add(1)
		if n == 1 {
			return nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock"}), nil, nil
		}
		if n == 2 {
			return nil, nil, errReplacement
		}
		return nil, nil, rollback
	}
	a, cancel, done := independentStart(t, cfg, factory)
	defer cancel()

	// Establish old-collector drain completion before testing replacement failure.
	a.targetMu.Lock()
	old := a.target
	a.targetMu.Unlock()
	old.stopping.Store(true)
	old.cancel()
	go func() { _, _ = old.collector.Stop() }()
	select {
	case <-old.collectorDone:
	case <-time.After(time.Second):
		t.Fatal("old collector did not drain for replacement fixture")
	}

	next := *cfg
	next.Interval = 2 * time.Minute
	reloadErr := a.ReloadConfig(&next)
	if reloadErr == nil || !errors.Is(reloadErr, errReplacement) {
		t.Fatalf("replacement failure branch not reached: %v", reloadErr)
	}
	select {
	case e := <-done:
		if !errors.Is(e, rollback) || !strings.Contains(e.Error(), "replacement unavailable") {
			t.Fatalf("terminal failures lost: %v", e)
		}
	case <-time.After(time.Second):
		t.Fatal("failed rollback did not terminate")
	}
}

func TestLatestOutageOpenMetrics(t *testing.T) {
	a, e := NewAgent(independentCfg(), "", nil, nil)
	if e != nil {
		t.Fatal(e)
	}
	request := httptest.NewRequest("GET", "/metrics", nil)
	request.Header.Set("Accept", "application/openmetrics-text; version=1.0.0")
	r := httptest.NewRecorder()
	a.handleMetrics(r, request)
	if r.Code != 200 || !strings.HasPrefix(r.Header().Get("Content-Type"), "application/openmetrics-text") || !strings.HasSuffix(r.Body.String(), "# EOF\n") {
		t.Errorf("outage does not preserve negotiated OpenMetrics: code=%d type=%q EOF=%v", r.Code, r.Header().Get("Content-Type"), strings.HasSuffix(r.Body.String(), "# EOF\n"))
	}
}

func TestLatestTimedOutTransitionCannotReportHealthyNoop(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var acquisitions atomic.Int32
	factory := func(*ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		n := acquisitions.Add(1)
		f := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock"})
		if n == 1 {
			f.SetFallback(func(context.Context) ([]nvtelemetry.Reading, error) { close(entered); <-release; return nil, nil })
		}
		return f, nil, nil
	}
	cfg := independentCfg()
	a, cancel, done := independentStart(t, cfg, factory)
	defer func() { cancel(); a.Shutdown(context.Background()) }()
	<-entered
	next := *cfg
	next.Interval = 2 * time.Minute
	if e := a.ReloadConfig(&next); e == nil {
		t.Fatal("blocked transition succeeded")
	}
	close(release)
	a.targetMu.Lock()
	oldDone := a.target.collectorDone
	a.targetMu.Unlock()
	select {
	case <-oldDone:
	case <-time.After(time.Second):
		t.Fatal("old source did not return")
	}
	e := a.ReloadConfig(cfg)
	select {
	case <-done:
		return
	case <-time.After(20 * time.Millisecond):
	}
	if e == nil && acquisitions.Load() == 1 {
		t.Error("successful unchanged-config retry reused canceled finished collector without reacquisition")
	}
}
