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

func TestSuccessorFailedAcquisitionKeepsServiceOrTerminates(t *testing.T) {
	cfg := independentCfg()
	var acquisitions atomic.Int32
	errReplacementUnavailable := errors.New("replacement unavailable")
	factory := func(c *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		acquisitions.Add(1)
		if c.Interval != cfg.Interval {
			return nil, nil, errReplacementUnavailable
		}
		return nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock"}), nil, nil
	}
	a, cancel, done := independentStart(t, cfg, factory)
	defer func() { cancel(); a.Shutdown(context.Background()) }()

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
	if reloadErr == nil || !errors.Is(reloadErr, errReplacementUnavailable) {
		t.Fatalf("replacement failure branch not reached: %v", reloadErr)
	}
	select {
	case <-done:
		return
	case <-time.After(50 * time.Millisecond):
	}
	if acquisitions.Load() < 3 {
		t.Errorf("reload stopped old target, failed replacement and left daemon running without restored source: acquisitions=%d reloadErr=%v", acquisitions.Load(), reloadErr)
	}
}

func TestSuccessorOutageSelfMetricsAreScrapable(t *testing.T) {
	a, e := NewAgent(independentCfg(), "", nil, nil)
	if e != nil {
		t.Fatal(e)
	}
	r := httptest.NewRecorder()
	a.handleMetrics(r, httptest.NewRequest("GET", "/metrics", nil))
	if r.Code != 200 || strings.HasPrefix(r.Body.String(), "telemetry unavailable") {
		t.Errorf("selfmetrics only embedded in non-success error response: status=%d body=%q", r.Code, r.Body.String())
	}
}
