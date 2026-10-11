package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func newTestAgent(t *testing.T, customSampler nvtelemetry.Sampler) (*Agent, *endpointLine, func()) {
	t.Helper()
	cfg := &ParsedConfig{
		Listen:          "127.0.0.1:0",
		Interval:        10 * time.Millisecond,
		MaxAge:          500 * time.Millisecond,
		Budget:          1000,
		ShutdownTimeout: 2 * time.Second,
		Mock:            customSampler == nil,
		IsLoopback:      true,
	}

	line := &endpointLine{addr: make(chan string, 1)}
	agent, err := NewAgent(cfg, "", line, io.Discard)
	if err != nil {
		t.Fatalf("NewAgent: %v", err)
	}

	if customSampler != nil {
		agent.SetSamplerFactory(func(_ *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
			return customSampler, nil, nil
		})
	}

	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() {
		done <- agent.Start(ctx)
	}()

	select {
	case <-line.addr:
	case err := <-done:
		t.Fatalf("agent startup failed: %v", err)
	case <-time.After(3 * time.Second):
		t.Fatal("agent startup timeout")
	}

	cleanup := func() {
		cancel()
		_ = agent.Shutdown(context.Background())
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatal("agent shutdown timeout")
		}
	}
	return agent, line, cleanup
}

func httpGet(t *testing.T, url, accept string) (int, string) {
	t.Helper()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		t.Fatal(err)
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, string(raw)
}

func TestHealthzAndReadyzEndpoints(t *testing.T) {
	agent, line, cleanup := newTestAgent(t, nil)
	defer cleanup()

	addr := agent.Addr()
	healthzURL := fmt.Sprintf("http://%s/healthz", addr)
	readyzURL := fmt.Sprintf("http://%s/readyz", addr)

	// Liveness: /healthz must return 200 OK immediately
	code, body := httpGet(t, healthzURL, "")
	if code != http.StatusOK || body != "ok\n" {
		t.Fatalf("/healthz = (%d, %q), want (200, %q)", code, body, "ok\n")
	}

	// Readiness: /readyz returns 200 once snapshot is published
	deadline := time.Now().Add(2 * time.Second)
	ready := false
	for time.Now().Before(deadline) {
		code, body = httpGet(t, readyzURL, "")
		if code == http.StatusOK && body == "ready\n" {
			ready = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !ready {
		t.Fatalf("/readyz did not become ready before timeout: last (%d, %q)", code, body)
	}

	_ = line
}

func TestAgentDynamicReloadValid(t *testing.T) {
	agent, _, cleanup := newTestAgent(t, nil)
	defer cleanup()

	addr := agent.Addr()
	metricsURL := fmt.Sprintf("http://%s/metrics", addr)

	// Scrape to ensure initial collection has occurred
	deadline := time.Now().Add(5 * time.Second)
	var initialReady bool
	for time.Now().Before(deadline) {
		code, body := httpGet(t, metricsURL, "")
		if code == http.StatusOK && strings.Contains(body, "GPU-mock") {
			initialReady = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !initialReady {
		t.Fatal("timed out waiting for initial telemetry sample to become available")
	}

	// Reload with valid updated configuration
	newCfg := &ParsedConfig{
		Listen:          agent.cfg.Listen,
		Interval:        5 * time.Millisecond,
		MaxAge:          1 * time.Second,
		Budget:          500,
		ShutdownTimeout: 2 * time.Second,
		Mock:            true,
		IsLoopback:      true,
	}

	if err := agent.ReloadConfig(newCfg); err != nil {
		t.Fatalf("ReloadConfig valid failed: %v", err)
	}

	// Verify reload counter incremented in metrics exposition
	deadline = time.Now().Add(2 * time.Second)
	var body string
	for time.Now().Before(deadline) {
		code, b := httpGet(t, metricsURL, "")
		if code == http.StatusOK {
			body = b
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if body == "" {
		t.Fatalf("scrape after reload failed to become ready")
	}
	if !strings.Contains(body, `nvtelemetry_agent_reloads_total{result="success"} 1`) {
		t.Fatalf("expected reload success counter in exposition, got:\n%s", body)
	}
}

func TestAgentDynamicReloadInvalidPreservesState(t *testing.T) {
	agent, _, cleanup := newTestAgent(t, nil)
	defer cleanup()

	addr := agent.Addr()
	metricsURL := fmt.Sprintf("http://%s/metrics", addr)

	// Wait until ready and initial sample is available
	deadline := time.Now().Add(5 * time.Second)
	var ready bool
	for time.Now().Before(deadline) {
		code, body := httpGet(t, metricsURL, "")
		if code == http.StatusOK && strings.Contains(body, "GPU-mock") {
			ready = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !ready {
		t.Fatal("timed out waiting for initial telemetry sample to become available")
	}

	// Attempt reload with invalid config (illegal address change)
	invalidCfg := &ParsedConfig{
		Listen:          "127.0.0.1:9999", // Cannot change listen address
		Interval:        10 * time.Millisecond,
		MaxAge:          500 * time.Millisecond,
		Budget:          1000,
		ShutdownTimeout: 2 * time.Second,
		Mock:            true,
		IsLoopback:      true,
	}

	if err := agent.ReloadConfig(invalidCfg); err == nil {
		t.Fatal("expected ReloadConfig to fail for changed listen address")
	}

	// Verify running state is completely preserved and serving
	code, body := httpGet(t, metricsURL, "")
	if code != http.StatusOK {
		t.Fatalf("scrape after failed reload failed: code=%d", code)
	}
	if !strings.Contains(body, `nvtelemetry_agent_reloads_total{result="failure"} 1`) {
		t.Fatalf("expected reload failure counter in exposition, got:\n%s", body)
	}
	if !strings.Contains(body, "GPU-mock") {
		t.Fatalf("expected existing telemetry data preserved, got:\n%s", body)
	}
}

func TestAgentDynamicReloadTargetTransition(t *testing.T) {
	var targetADrained atomic.Bool
	var targetBStarted atomic.Bool

	metaA := nvtelemetry.DeviceMetadata{UUID: "GPU-target-A", Model: "testA", Status: nvtelemetry.DeviceStatusOk}
	samplerA := nvtelemetry.NewFakeSampler(metaA)
	samplerA.SetFallback(func(ctx context.Context) ([]nvtelemetry.Reading, error) {
		if ctx.Err() != nil {
			targetADrained.Store(true)
			return nil, ctx.Err()
		}
		return []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU-target-A", Evidence: nvtelemetry.EvidenceDerived, Backend: "mock", Value: 25000, TimestampUnix: time.Now().UnixNano()}}, nil
	})

	metaB := nvtelemetry.DeviceMetadata{UUID: "GPU-target-B", Model: "testB", Status: nvtelemetry.DeviceStatusOk}
	samplerB := nvtelemetry.NewFakeSampler(metaB)
	samplerB.SetFallback(func(ctx context.Context) ([]nvtelemetry.Reading, error) {
		targetBStarted.Store(true)
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU-target-B", Evidence: nvtelemetry.EvidenceDerived, Backend: "mock", Value: 35000, TimestampUnix: time.Now().UnixNano()}}, nil
	})

	agent, _, cleanup := newTestAgent(t, samplerA)
	defer cleanup()

	addr := agent.Addr()
	metricsURL := fmt.Sprintf("http://%s/metrics", addr)

	// Wait for target A readings
	deadline := time.Now().Add(5 * time.Second)
	var readyA bool
	for time.Now().Before(deadline) {
		code, body := httpGet(t, metricsURL, "")
		if code == http.StatusOK && strings.Contains(body, "GPU-target-A") {
			readyA = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !readyA {
		t.Fatal("timed out waiting for target A readings to become available")
	}

	// Switch target to Target B
	agent.SetSamplerFactory(func(c *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		return samplerB, nil, nil
	})
	newCfg := &ParsedConfig{
		Listen:          agent.cfg.Listen,
		Interval:        5 * time.Millisecond,
		MaxAge:          1 * time.Second,
		Budget:          2000,
		ShutdownTimeout: 2 * time.Second,
		Mock:            true,
		IsLoopback:      true,
	}

	if err := agent.ReloadConfig(newCfg); err != nil {
		t.Fatalf("reload to target B failed: %v", err)
	}

	// Verify Target B is active and Target A was drained
	deadline = time.Now().Add(2 * time.Second)
	targetBSeen := false
	for time.Now().Before(deadline) {
		code, body := httpGet(t, metricsURL, "")
		if code == http.StatusOK && strings.Contains(body, "GPU-target-B") {
			targetBSeen = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !targetBSeen {
		t.Fatal("target B was not observed after transition")
	}
}

type blockedSampler struct {
	blockCh chan struct{}
	entered chan struct{}
}

func (b *blockedSampler) Devices() []nvtelemetry.DeviceMetadata {
	return []nvtelemetry.DeviceMetadata{{UUID: "GPU-blocked", Model: "blocked", Status: nvtelemetry.DeviceStatusOk}}
}

func (b *blockedSampler) Sample(ctx context.Context) ([]nvtelemetry.Reading, error) {
	if b.entered != nil {
		select {
		case b.entered <- struct{}{}:
		default:
		}
	}
	// Block until unblocked
	select {
	case <-b.blockCh:
		return nil, errors.New("unblocked")
	case <-time.After(10 * time.Second):
		return nil, errors.New("timeout")
	}
}

func TestAgentBlockedSourceRefusesRelease(t *testing.T) {
	blockCh := make(chan struct{})
	defer close(blockCh)
	entered := make(chan struct{}, 1)

	bs := &blockedSampler{blockCh: blockCh, entered: entered}
	// Start with blocked sampler as initial target
	agent, _, cleanup := newTestAgent(t, bs)
	defer cleanup()

	agent.cfg.ShutdownTimeout = 50 * time.Millisecond

	// Wait for sampler to enter Sample before triggering reload,
	// proving the blocked-source precondition is established.
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for sampler to enter Sample")
	}

	// Now attempt reload to target B while target A (bs) is blocked
	metaB := nvtelemetry.DeviceMetadata{UUID: "GPU-target-B", Model: "testB", Status: nvtelemetry.DeviceStatusOk}
	samplerB := nvtelemetry.NewFakeSampler(metaB)
	agent.SetSamplerFactory(func(c *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		return samplerB, nil, nil
	})

	newCfg := &ParsedConfig{
		Listen:          agent.cfg.Listen,
		Interval:        5 * time.Millisecond,
		MaxAge:          1 * time.Second,
		Budget:          5000,
		ShutdownTimeout: 50 * time.Millisecond,
		Mock:            true,
		IsLoopback:      true,
	}

	err := agent.ReloadConfig(newCfg)
	if err == nil {
		t.Fatal("expected reload to fail when active target is blocked")
	}
	if !strings.Contains(err.Error(), "blocked source retained") {
		t.Fatalf("expected 'blocked source retained' error, got: %v", err)
	}

	// Verify blocked source retained ownership
	agent.targetMu.Lock()
	activeTgt := agent.target
	agent.targetMu.Unlock()

	if activeTgt == nil || activeTgt.sampler != nvtelemetry.Sampler(bs) {
		t.Fatal("blocked source did not retain ownership")
	}
}

func TestAgentSelfMetricsDistinction(t *testing.T) {
	meta := nvtelemetry.DeviceMetadata{UUID: "GPU-mock", Model: "synthetic", Status: nvtelemetry.DeviceStatusOk}
	sampler := nvtelemetry.NewFakeSampler(meta)

	var shouldFailCollection atomic.Bool
	sampler.SetFallback(func(ctx context.Context) ([]nvtelemetry.Reading, error) {
		evidence := nvtelemetry.EvidenceDerived
		val := 40000.0
		reason := ""
		if shouldFailCollection.Load() {
			evidence = nvtelemetry.EvidenceUnavailable
			val = 0
			reason = "sensor_read_error"
		}
		return []nvtelemetry.Reading{{
			Name:              nvtelemetry.MetricPowerUsageMW,
			Unit:              nvtelemetry.UnitMilliwatts,
			DeviceUUID:        "GPU-mock",
			Evidence:          evidence,
			UnavailableReason: reason,
			Backend:           "mock",
			Value:             val,
			TimestampUnix:     time.Now().UnixNano(),
		}}, nil
	})

	agent, _, cleanup := newTestAgent(t, sampler)
	defer cleanup()

	addr := agent.Addr()
	metricsURL := fmt.Sprintf("http://%s/metrics", addr)

	// Wait until at least 1 successful collection
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		code, _ := httpGet(t, metricsURL, "")
		if code == http.StatusOK {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Simulate a collection failure
	shouldFailCollection.Store(true)
	time.Sleep(50 * time.Millisecond) // Allow collector tick to run

	// Scrape OpenMetrics
	code, body := httpGet(t, metricsURL, "application/openmetrics-text;version=1.0.0")
	if code != http.StatusOK {
		t.Fatalf("OpenMetrics scrape failed: code=%d", code)
	}

	// Must end with # EOF
	if !strings.HasSuffix(body, "# EOF\n") {
		t.Fatalf("OpenMetrics scrape missing # EOF suffix:\n%s", body)
	}

	// Check self-metrics distinguishing collection failures and delivery failures
	if !strings.Contains(body, "nvtelemetry_agent_collection_failures_total") {
		t.Fatalf("missing collection failures self-metric in exposition:\n%s", body)
	}
	if !strings.Contains(body, "nvtelemetry_agent_delivery_failures_total") {
		t.Fatalf("missing delivery failures self-metric in exposition:\n%s", body)
	}
	if !strings.Contains(body, "nvtelemetry_agent_up 1") {
		t.Fatalf("missing agent up self-metric in exposition:\n%s", body)
	}
}

func TestAgentGracefulShutdownDrain(t *testing.T) {
	agent, _, cleanup := newTestAgent(t, nil)
	defer cleanup()

	// Wait for a sample to be produced
	time.Sleep(30 * time.Millisecond)

	// Liveness: while running, /healthz is 200
	code, body := httpGet(t, fmt.Sprintf("http://%s/healthz", agent.Addr()), "")
	if code != http.StatusOK || body != "ok\n" {
		t.Fatalf("healthz before shutdown = (%d, %q), want (200, ok)", code, body)
	}

	// Signal shutting down state and verify /healthz reports 503
	agent.shuttingDown.Store(true)
	code, body = httpGet(t, fmt.Sprintf("http://%s/healthz", agent.Addr()), "")
	if code != http.StatusServiceUnavailable || !strings.Contains(body, "shutting down") {
		t.Fatalf("healthz during shutdown = (%d, %q), want (503, shutting down)", code, body)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	err := agent.Shutdown(ctx)
	if err != nil {
		t.Fatalf("graceful shutdown failed: %v", err)
	}
}

func TestSIGHUPReload(t *testing.T) {
	agent, _, cleanup := newTestAgent(t, nil)
	defer cleanup()

	// Sending SIGHUP to the current process should not crash and should trigger reload handler
	_ = syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	time.Sleep(50 * time.Millisecond)

	code, _ := httpGet(t, fmt.Sprintf("http://%s/healthz", agent.Addr()), "")
	if code != http.StatusOK {
		t.Fatalf("healthz after SIGHUP = %d, want 200", code)
	}
}
