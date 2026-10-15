package nvtelemetry

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func awaitSampleResult(t *testing.T, result <-chan error) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(2 * time.Second):
		t.Fatal("collection did not return; admission or callback deadlocked")
		return nil
	}
}

func TestCollectorSampleAdmission(t *testing.T) {
	for _, first := range []string{"Step", "Run"} {
		t.Run(first+" then Step", func(t *testing.T) {
			entered := make(chan struct{}, 2)
			release := make(chan struct{})
			var active, maxActive atomic.Int32
			sampler := &phaseContractSampler{sample: func(ctx context.Context) ([]Reading, error) {
				n := active.Add(1)
				defer active.Add(-1)
				for old := maxActive.Load(); n > old; old = maxActive.Load() {
					if maxActive.CompareAndSwap(old, n) {
						break
					}
				}
				entered <- struct{}{}
				select {
				case <-release:
					return nil, nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			}}
			c := NewCollector("sample-admission", sampler, NewFakeClock(1, 1), CollectorConfig{IntervalNanos: int64(time.Hour)}, nil)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			firstDone := make(chan error, 1)
			go func() {
				if first == "Run" {
					_, err := c.Run(ctx)
					firstDone <- err
				} else {
					firstDone <- c.Step(ctx)
				}
			}()
			select {
			case <-entered:
			case <-time.After(2 * time.Second):
				t.Fatal("initial sampler did not enter")
			}
			secondDone := make(chan error, 1)
			go func() { secondDone <- c.Step(t.Context()) }()
			select {
			case err := <-secondDone:
				if err == nil || !strings.Contains(err.Error(), "in progress") {
					t.Fatalf("overlap did not refuse: %v", err)
				}
			case <-entered:
				close(release)
				cancel()
				_ = awaitSampleResult(t, firstDone)
				_ = awaitSampleResult(t, secondDone)
				t.Fatal("actual Sampler.Sample overlapped")
			case <-time.After(2 * time.Second):
				t.Fatal("second Step queued behind blocked provider")
			}
			cancel()
			if err := awaitSampleResult(t, firstDone); !errors.Is(err, context.Canceled) {
				t.Fatalf("provider cancellation identity: %v", err)
			}
			if maxActive.Load() != 1 || active.Load() != 0 {
				t.Fatalf("provider calls active/max = %d/%d", active.Load(), maxActive.Load())
			}
			if first == "Step" {
				close(release)
				if err := c.Step(t.Context()); err != nil {
					t.Fatalf("canceled provider retained admission: %v", err)
				}
			}
		})
	}
}

func TestCollectorAdmissionErrorAndCanceledInput(t *testing.T) {
	cause := errors.New("sample failure")
	calls := 0
	sampler := &phaseContractSampler{sample: func(context.Context) ([]Reading, error) {
		calls++
		if calls == 1 {
			return nil, cause
		}
		return nil, nil
	}}
	c := NewCollector("error-release", sampler, NewFakeClock(1, 1), CollectorConfig{}, nil)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := c.Step(ctx); !errors.Is(err, context.Canceled) || calls != 0 {
		t.Fatalf("canceled input touched sampler: %v calls %d", err, calls)
	}
	if err := c.Step(t.Context()); !errors.Is(err, cause) {
		t.Fatalf("sample cause lost: %v", err)
	}
	if err := c.Step(t.Context()); err != nil || calls != 2 {
		t.Fatalf("error retained admission: %v calls %d", err, calls)
	}
}

func TestCollectorAdmissionReentrantHookAndProvider(t *testing.T) {
	var c *Collector
	sampler := &phaseContractSampler{sample: func(context.Context) ([]Reading, error) {
		// This would deadlock if Sample held the collector state mutex.
		return nil, c.SetPhase("provider")
	}}
	c = NewCollector("reentrant", sampler, NewFakeClock(1, 1), CollectorConfig{}, nil)
	nested := false
	hookCalls := 0
	var hookErr error
	c.SetSampleHook(func(Sample) {
		hookCalls++
		if nested {
			return
		}
		nested = true
		if err := c.SetPhase("hook"); err != nil {
			hookErr = err
			return
		}
		hookErr = c.Step(t.Context())
	})
	done := make(chan error, 1)
	go func() { done <- c.Step(t.Context()) }()
	if err := awaitSampleResult(t, done); err != nil || hookErr != nil || hookCalls != 2 {
		t.Fatalf("reentrant hook/provider admission: outer=%v hook=%v calls=%d", err, hookErr, hookCalls)
	}
}
