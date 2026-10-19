package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestOwnedDeliveryConfigNormalizedBeforeValidation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	raw := ServeConfig{Mock: true, OTLPMetricsConfig: "metrics-does-not-exist.json", OTLPLogsConfig: "logs-does-not-exist.json"}
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	owner := &serveDeliveryOwnership{metrics: filepath.Join(dir, raw.OTLPMetricsConfig), logs: filepath.Join(dir, raw.OTLPLogsConfig)}
	cfg, err := parseConfigFile(path, owner.normalizeRaw)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.OTLPMetricsConfig != "" || cfg.OTLPLogsConfig != "" {
		t.Fatal("owned fields not normalized")
	}
	// The public legacy parser still validates endpoint files; normalization is
	// private and used only when serve already owns their durable exporters.
	if _, err := ParseConfigFile(path); err == nil || !strings.Contains(err.Error(), "open endpoint configuration file failed") {
		t.Fatalf("legacy endpoint validation=%v", err)
	}
	sentinel := errors.New("normalization refusal")
	if _, err := parseConfigBytesWith(data, dir, func(r ServeConfig) (ServeConfig, error) {
		if r.OTLPMetricsConfig != owner.metrics {
			t.Fatal("normalizer ran before relative path resolution")
		}
		return r, sentinel
	}); !errors.Is(err, sentinel) {
		t.Fatalf("normalizer error identity=%v", err)
	}
	calls := 0
	if _, err := parseConfigBytesWith([]byte(`{"mock":true,"mock":false}`), dir, func(r ServeConfig) (ServeConfig, error) { calls++; return r, nil }); err == nil || calls != 0 {
		t.Fatal("normalizer bypassed strict parser")
	}
}

func TestOwnedDeliveryPublicFileReloadPreservesOwnership(t *testing.T) {
	l, dir, cfg := recoveryLifecycle(t)
	owner := &serveDeliveryOwnership{metrics: filepath.Join(dir, "metrics-must-not-open.json"), logs: filepath.Join(dir, "logs-must-not-open.json")}
	cfg.OTLPMetricsConfig, cfg.OTLPLogsConfig = owner.metrics, owner.logs
	path := filepath.Join(t.TempDir(), "config.json")
	writeConfig := func(interval, metrics string) {
		t.Helper()
		raw := ServeConfig{Mock: true, Listen: cfg.Listen, Interval: interval, MaxAge: "1s", ShutdownTimeout: "1s", Budget: cfg.Budget, OTLPMetricsConfig: metrics, OTLPLogsConfig: owner.logs}
		data, err := json.Marshal(raw)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	agent, err := NewAgent(cfg, path, io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	agent.ownedDelivery = owner
	agent.ctx, agent.cancel = context.WithCancel(t.Context())
	acquisitions := 0
	agent.SetSamplerFactory(func(c *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		acquisitions++
		if c.OTLPMetricsConfig != "" || c.OTLPLogsConfig != "" {
			t.Error("factory received duplicate exporter configuration")
		}
		return l.sampler(c)
	})
	defer func() {
		if err := agent.Shutdown(context.Background()); err != nil {
			t.Error(err)
		}
	}()
	waitSamples := func() {
		t.Helper()
		deadline := time.Now().Add(time.Second)
		for time.Now().Before(deadline) {
			agent.targetMu.Lock()
			target := agent.target
			agent.targetMu.Unlock()
			if target != nil {
				s := target.sampler.(*serveLifecycleSampler)
				s.mu.Lock()
				n := s.sampleCount
				s.mu.Unlock()
				if n >= 2 {
					return
				}
			}
			time.Sleep(time.Millisecond)
		}
		t.Fatal("generation did not produce readings")
	}
	writeConfig("1ms", owner.metrics)
	if err := agent.Reload(t.Context()); err != nil {
		t.Fatal(err)
	}
	waitSamples()
	writeConfig("2ms", owner.metrics)
	if err := agent.Reload(t.Context()); err != nil {
		t.Fatal(err)
	}
	waitSamples()
	if acquisitions != 2 || l.retention.Status().Stores != 1 {
		t.Fatalf("acquisitions=%d stores=%d", acquisitions, l.retention.Status().Stores)
	}
	agent.targetMu.Lock()
	target := agent.target
	agent.targetMu.Unlock()
	if target.metricsDelivery != nil || target.logsDelivery != nil {
		t.Fatal("ordinary exporter acquired for durable-owned endpoint")
	}
	writeConfig("3ms", owner.metrics+"-changed")
	if err := agent.Reload(t.Context()); err == nil || !strings.Contains(err.Error(), "requires restart") {
		t.Fatalf("endpoint change=%v", err)
	}
	if acquisitions != 2 || target.ctx.Err() != nil {
		t.Fatal("refusal acquired source or drained existing generation")
	}
	// In-memory configuration follows the same ownership rule without mutating
	// the caller. Removing the owned endpoint is also an explicit change.
	same := *cfg
	same.Interval = 3 * time.Millisecond
	if err := agent.ReloadConfig(&same); err != nil {
		t.Fatal(err)
	}
	waitSamples()
	if same.OTLPMetricsConfig != owner.metrics || same.OTLPLogsConfig != owner.logs {
		t.Fatal("caller endpoint fields mutated")
	}
	same.OTLPMetricsConfig = "caller-mutated-after-reload"
	if agent.cfg.OTLPMetricsConfig != owner.metrics {
		t.Fatal("Agent retained mutable caller config")
	}
	changed := *cfg
	changed.OTLPMetricsConfig = ""
	if err := agent.ReloadConfig(&changed); err == nil {
		t.Fatal("disabled owned endpoint admitted")
	}
	if err := agent.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := l.result(); err != nil {
		t.Fatal(err)
	}
	if stores := l.retention.Status().Stores; stores != 3 {
		t.Fatalf("retained generations=%d", stores)
	}
}

func TestOwnedDeliveryTargetCloneAndDefault(t *testing.T) {
	cfg := independentCfg()
	cfg.OTLPMetricsConfig = "metrics"
	cfg.DeviceUUIDs = []string{"GPU-synthetic"}
	agent, err := NewAgent(cfg, "", io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := agent.targetConfig(cfg)
	if err != nil || legacy != cfg {
		t.Fatalf("default behavior changed: %v", err)
	}
	agent.ownedDelivery = &serveDeliveryOwnership{metrics: "metrics"}
	normalized, err := agent.targetConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	normalized.DeviceUUIDs[0] = "different"
	if cfg.DeviceUUIDs[0] != "GPU-synthetic" || cfg.OTLPMetricsConfig != "metrics" || normalized.OTLPMetricsConfig != "" {
		t.Fatal("owned normalization mutated caller")
	}
}
