package main

import (
	"context"
	"io"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// ReloadConfig holds mu while draining the old collector. Its real sample hook
// must be able to finish publishing under that lock, before collectorDone closes.
func TestAgentReloadSampleHookIndependentOfConfigLock(t *testing.T) {
	_, _, cfg := recoveryLifecycle(t)
	cfg.Interval = time.Hour
	agent, err := NewAgent(cfg, "", io.Discard, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	agent.ctx, agent.cancel = context.WithCancel(t.Context())
	entered := make(chan struct{})
	release := make(chan struct{})
	first := true
	agent.SetSamplerFactory(func(c *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
		base, cleanup, err := buildBaseServeSampler(c)
		if !first || err != nil {
			return base, cleanup, err
		}
		first = false
		sampler := nvtelemetry.NewFakeSampler(base.Devices()...)
		sampler.SetFallback(func(ctx context.Context) ([]nvtelemetry.Reading, error) {
			close(entered)
			select {
			case <-release:
				return base.Sample(ctx)
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		})
		return sampler, cleanup, nil
	})
	defer func() {
		if err := agent.Shutdown(context.Background()); err != nil {
			t.Error(err)
		}
	}()
	if err := agent.ReloadConfig(cfg); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("initial source did not start")
	}
	// Explicitly hold precisely the lock retained by public ReloadConfig while
	// releasing a real acquisition. collectionSuccesses is incremented by the
	// actual installed hook only after getMetricsHandler and Handler.Update.
	agent.mu.Lock()
	close(release)
	deadline := time.Now().Add(time.Second)
	for agent.metrics.collectionSuccesses.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	published := agent.metrics.collectionSuccesses.Load()
	agent.mu.Unlock()
	if published != 1 {
		t.Fatal("sample hook could not finish while reload configuration lock was held")
	}
	next := *cfg
	next.Interval = 2 * time.Millisecond
	if err := agent.ReloadConfig(&next); err != nil {
		t.Fatal(err)
	}
	// Establish that the replacement actually started before shutting it down;
	// stopping an as-yet-unstarted Collector exercises a different contract.
	deadline = time.Now().Add(time.Second)
	for agent.metrics.collectionSuccesses.Load() < 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if agent.metrics.collectionSuccesses.Load() < 2 {
		t.Fatal("replacement source did not publish")
	}
	if err := agent.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
}
