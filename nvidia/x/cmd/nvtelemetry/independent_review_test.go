package main

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestIndependentStrictConfig(t *testing.T) {
	for _, s := range []string{`null`, `{"mock":null}`, `{"budget":null}`, `{"mock":true,"interval":null}`} {
		if _, e := ParseConfigJSON([]byte(s)); e == nil {
			t.Errorf("accepted null configuration %s", s)
		}
	}
	for _, s := range []string{`{"mock":true,"mock":false}`, `{"mock":true} {}`, `{"mock":true,"unknown":1}`} {
		if _, e := ParseConfigJSON([]byte(s)); e == nil {
			t.Errorf("accepted contradiction %s", s)
		}
	}
	if _, e := ParseConfigJSON([]byte(`{"mock":true}` + strings.Repeat(" ", 1<<20))); e == nil {
		t.Error("public byte parser bypasses 1MiB bound")
	}
}
func independentStart(t *testing.T, cfg *ParsedConfig, fn func(*ParsedConfig) (nvtelemetry.Sampler, func(), error)) (*Agent, context.CancelFunc, chan error) {
	t.Helper()
	line := &endpointLine{addr: make(chan string, 1)}
	a, e := NewAgent(cfg, "", line, io.Discard)
	if e != nil {
		t.Fatal(e)
	}
	if fn != nil {
		a.SetSamplerFactory(fn)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- a.Start(ctx) }()
	select {
	case <-line.addr:
	case e := <-done:
		t.Fatal(e)
	case <-time.After(time.Second):
		t.Fatal("startup timeout")
	}
	return a, cancel, done
}
func independentCfg() *ParsedConfig {
	return &ParsedConfig{Listen: "127.0.0.1:0", Interval: time.Minute, MaxAge: time.Second, Budget: 100, ShutdownTimeout: 30 * time.Millisecond, Mock: true, IsLoopback: true}
}
func TestIndependentReloadMaxAge(t *testing.T) {
	cfg := independentCfg()
	a, cancel, done := independentStart(t, cfg, nil)
	defer func() { cancel(); a.Shutdown(context.Background()); <-done }()
	next := *cfg
	next.MaxAge = time.Millisecond
	if e := a.ReloadConfig(&next); e != nil {
		t.Fatal(e)
	}
	if e := a.metricsHandler.Update([]nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, DeviceUUID: "GPU-mock", Evidence: nvtelemetry.EvidenceDerived, Value: 1}}); e != nil {
		t.Fatal(e)
	}
	time.Sleep(10 * time.Millisecond)
	r := httptest.NewRecorder()
	a.handleReadyz(r, httptest.NewRequest("GET", "/readyz", nil))
	if r.Code != 503 {
		t.Errorf("reloaded 1ms max_age ignored: ready=%d", r.Code)
	}
}
func TestIndependentBlockedOwnership(t *testing.T) {
	for _, mode := range []string{"reload", "shutdown"} {
		t.Run(mode, func(t *testing.T) {
			entered := make(chan struct{})
			release := make(chan struct{})
			var acquired, cleaned atomic.Int32
			factory := func(*ParsedConfig) (nvtelemetry.Sampler, func(), error) {
				n := acquired.Add(1)
				f := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{UUID: "GPU-mock"})
				if n == 1 {
					f.SetFallback(func(context.Context) ([]nvtelemetry.Reading, error) { close(entered); <-release; return nil, nil })
				}
				return f, func() { cleaned.Add(1) }, nil
			}
			cfg := independentCfg()
			a, cancel, done := independentStart(t, cfg, factory)
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("sample not entered")
			}
			if mode == "reload" {
				next := *cfg
				next.Interval = 2 * time.Minute
				if e := a.ReloadConfig(&next); e == nil {
					t.Error("blocked reload succeeded")
				}
				if acquired.Load() != 1 {
					t.Errorf("replacement acquired before old source drained: %d", acquired.Load())
				}
			} else {
				if e := a.Shutdown(context.Background()); e == nil {
					t.Error("blocked shutdown succeeded")
				}
				if cleaned.Load() != 0 {
					t.Errorf("cleaned active sampler before blocked source returned: %d", cleaned.Load())
				}
			}
			close(release)
			cancel()
			a.Shutdown(context.Background())
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("shutdown failed to finish")
			}
		})
	}
}

func TestIndependentValidationDoesNotStartWorkers(t *testing.T) {
	count := func() int {
		b := make([]byte, 1<<20)
		n := runtime.Stack(b, true)
		return strings.Count(string(b[:n]), "(*Delivery).run(")
	}
	before := count()
	p := filepath.Join(t.TempDir(), "endpoint.json")
	if e := os.WriteFile(p, []byte(`{"url":"http://127.0.0.1:1/v1/metrics","allow_http":true,"timeout_nanos":1000000,"max_bytes":4096}`), 0600); e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 3; i++ {
		if _, e := ParseConfigJSON([]byte(fmt.Sprintf(`{"mock":true,"otlp_metrics_config":%q}`, p))); e != nil {
			t.Fatal(e)
		}
	}
	time.Sleep(10 * time.Millisecond)
	if after := count(); after != before {
		t.Errorf("config validation leaked delivery workers: before=%d after=%d", before, after)
	}
}

func TestIndependentCanceledReload(t *testing.T) {
	a, cancel, done := independentStart(t, independentCfg(), nil)
	defer func() { cancel(); a.Shutdown(context.Background()); <-done }()
	ctx, stop := context.WithCancel(context.Background())
	stop()
	if e := a.Reload(ctx); e == nil {
		t.Error("reload ignored canceled caller context")
	}
}
func TestIndependentOutageSelfMetrics(t *testing.T) {
	a, e := NewAgent(independentCfg(), "", nil, nil)
	if e != nil {
		t.Fatal(e)
	}
	r := httptest.NewRecorder()
	a.handleMetrics(r, httptest.NewRequest("GET", "/metrics", nil))
	if !strings.Contains(r.Body.String(), "nvtelemetry_agent_collection_attempts_total") {
		t.Errorf("outage suppresses agent self-metrics: code=%d body=%q", r.Code, r.Body.String())
	}
}
