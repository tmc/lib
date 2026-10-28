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
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// serveCancelAfterRead returns the original successful reading unchanged, only
// after the real Agent cancels the source generation through reload/shutdown.
type serveCancelAfterRead struct {
	base    nvtelemetry.Sampler
	entered chan struct{}
	calls   int
}

func (s *serveCancelAfterRead) Devices() []nvtelemetry.DeviceMetadata { return s.base.Devices() }
func (s *serveCancelAfterRead) Sample(ctx context.Context) ([]nvtelemetry.Reading, error) {
	r, err := s.base.Sample(ctx)
	s.calls++
	if s.calls == 2 && err == nil {
		close(s.entered)
		<-ctx.Done()
	}
	return r, err
}

func TestServeCancelAfterSuccessfulRead(t *testing.T) {
	for _, action := range []string{"reload", "shutdown"} {
		t.Run(action, func(t *testing.T) {
			l, dir, cfg := recoveryLifecycle(t)
			a, err := NewAgent(cfg, "", io.Discard, io.Discard)
			if err != nil {
				t.Fatal(err)
			}
			a.ctx, a.cancel = context.WithCancel(t.Context())
			entered := make(chan struct{})
			nextEntered := make(chan struct{})
			generation := 0
			var original *serveCancelAfterRead
			a.SetSamplerFactory(func(c *ParsedConfig) (nvtelemetry.Sampler, func(), error) {
				s, cleanup, err := l.sampler(c)
				generation++
				if err == nil && generation == 1 {
					ls := s.(*serveLifecycleSampler)
					original = &serveCancelAfterRead{base: ls.base, entered: entered}
					ls.base = original
				}
				if err == nil && generation == 2 {
					ls := s.(*serveLifecycleSampler)
					base := ls.base
					ls.base = serveCancelSampler{sample: func(ctx context.Context) ([]nvtelemetry.Reading, error) {
						r, err := base.Sample(ctx)
						if err == nil {
							close(nextEntered)
						}
						return r, err
					}}
				}
				return s, cleanup, err
			})
			defer func() {
				if err := a.Shutdown(context.Background()); err != nil {
					t.Error(err)
				}
			}()
			if err := a.ReloadConfig(cfg); err != nil {
				t.Fatal(err)
			}
			select {
			case <-entered:
			case <-time.After(2 * time.Second):
				t.Fatal("second successful source read not reached")
			}
			if action == "reload" {
				next := *cfg
				next.Interval = time.Hour
				if err := a.ReloadConfig(&next); err != nil {
					t.Fatal(err)
				}
			} else if err := a.Shutdown(context.Background()); err != nil {
				t.Fatal(err)
			}
			if original.calls != 2 {
				t.Fatalf("source acquisitions=%d", original.calls)
			}
			if l.retention.Status().Stores != 1 {
				t.Fatalf("stored generations=%+v", l.retention.Status())
			}
			entries, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			streams := 0
			for _, e := range entries {
				if e.IsDir() {
					continue
				}
				raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
				if err != nil {
					t.Fatal(err)
				}
				m, _, _, err := nvtelemetry.ReadStreamStrict(bytes.NewReader(raw), 1<<20)
				if err != nil {
					continue
				}
				streams++
				if m.TotalSamples != 2 || !m.IsComplete || m.LostRecords != 0 {
					t.Fatalf("successful acquisitions missing from retained stream: %+v", m)
				}
			}
			if streams != 1 {
				t.Fatalf("retained streams=%d", streams)
			}
			if action == "reload" {
				// The old source was cancelled during its successful read above.
				// Wait only for the independent new source to start before its
				// shutdown, avoiding an unrelated pre-Run startup transition.
				select {
				case <-nextEntered:
				case <-time.After(2 * time.Second):
					t.Fatal("replacement source did not start")
				}
				if err := a.Shutdown(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			if err := l.result(); err != nil {
				t.Fatalf("successful acquired readings not finalized: %v", err)
			}
		})
	}
}

type serveCancelSampler struct {
	sample func(context.Context) ([]nvtelemetry.Reading, error)
}

func (s serveCancelSampler) Devices() []nvtelemetry.DeviceMetadata { return nil }
func (s serveCancelSampler) Sample(ctx context.Context) ([]nvtelemetry.Reading, error) {
	return s.sample(ctx)
}

func TestServeCancelBeforeSourceAndProducerFailure(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	calls := 0
	failure := errors.New("actual producer failure")
	l := &serveLifecycle{}
	s := &serveLifecycleSampler{base: serveCancelSampler{sample: func(context.Context) ([]nvtelemetry.Reading, error) { calls++; return nil, failure }}, failure: l.fail}
	if _, err := s.Sample(ctx); !errors.Is(err, context.Canceled) || calls != 0 {
		t.Fatalf("cancelled source called: calls=%d error=%v", calls, err)
	}
	if _, err := s.Sample(t.Context()); !errors.Is(err, failure) || calls != 1 {
		t.Fatalf("producer error changed: calls=%d error=%v", calls, err)
	}
}

func TestServeCancelDurableHandoff(t *testing.T) {
	for _, closed := range []bool{false, true} {
		t.Run(map[bool]string{false: "successful admission", true: "genuine closed admission"}[closed], func(t *testing.T) {
			makeDelivery := func(logs bool) *nvtelemetry.DurableDelivery {
				cfg := nvtelemetry.DurableDeliveryConfig{SpoolDir: filepath.Join(t.TempDir(), "spool"), Owner: "constructed-cancel", MaxSpoolBatches: 2, MaxSpoolBytes: 1 << 20, Delivery: nvtelemetry.DeliveryConfig{Endpoint: "http://constructed.invalid/v1/metrics", Client: &http.Client{Transport: recoveryTransport(func(*http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("{}"))}, nil
				})}, MaxQueuedBatches: 2, MaxQueuedBytes: 1 << 20, MaxBatchBytes: 1 << 20, MaxBatchDataPoints: 100, MaxResponseBytes: 4096, MaxAttempts: 1, InitialBackoff: time.Millisecond, MaxBackoff: time.Millisecond, BatchTimeout: time.Second, RequestTimeout: time.Second, ShutdownTimeout: time.Second}}
				var d *nvtelemetry.DurableDelivery
				var err error
				if logs {
					d, err = nvtelemetry.NewLogsDurableDelivery(cfg)
				} else {
					d, err = nvtelemetry.NewDurableDelivery(cfg)
				}
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := d.Close(context.Background()); err != nil {
						t.Error(err)
					}
				})
				if closed {
					if err := d.Close(t.Context()); err != nil {
						t.Fatal(err)
					}
				}
				return d
			}
			metrics, logs := makeDelivery(false), makeDelivery(true)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			l := &serveLifecycle{}
			s := &serveLifecycleSampler{base: serveCancelSampler{sample: func(context.Context) ([]nvtelemetry.Reading, error) {
				cancel()
				return []nvtelemetry.Reading{{Name: nvtelemetry.MetricPowerUsageMW, Unit: nvtelemetry.UnitMilliwatts, Value: 123, Evidence: nvtelemetry.EvidenceMeasured, Backend: "constructed", DeviceUUID: "GPU-constructed", TimestampUnix: 1}}, nil
			}}, durableMetrics: metrics, durableLogs: logs, failure: l.fail}
			_, err := s.Sample(ctx)
			if closed {
				if !errors.Is(err, nvtelemetry.ErrDeliveryClosed) || !errors.Is(l.result(), nvtelemetry.ErrDeliveryClosed) {
					t.Fatalf("genuine failed admission hidden: sample=%v retained=%v", err, l.result())
				}
				return
			}
			if err != nil || l.result() != nil {
				t.Fatalf("successful handoff rejected: sample=%v retained=%v", err, l.result())
			}
			for _, d := range []*nvtelemetry.DurableDelivery{metrics, logs} {
				if err := d.Close(t.Context()); err != nil {
					t.Fatal(err)
				}
				if st := d.Stats(); st.EnqueuedBatches != 1 || st.SucceededBatches != 1 || st.PendingSpool != 0 || st.DroppedBatches != 0 {
					t.Fatalf("handoff accounting=%+v", st)
				}
			}
		})
	}
}

func TestServeCancelLocalDeadlineRemainsFailure(t *testing.T) {
	l, _, cfg := recoveryLifecycle(t)
	sampler, cleanup, err := l.sampler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	s := sampler.(*serveLifecycleSampler)
	// Replace only the local collector's sampler: it records the handoff's
	// finite deadline and waits for it, without source/vendor or I/O work.
	seen := false
	calls := 0
	local := serveCancelSampler{sample: func(ctx context.Context) ([]nvtelemetry.Reading, error) {
		calls++
		if calls == 1 {
			return s.retStep.Sample(ctx)
		}
		if _, ok := ctx.Deadline(); !ok {
			t.Error("local handoff has no deadline")
		}
		seen = true
		<-ctx.Done()
		return nil, ctx.Err()
	}}
	config := nvtelemetry.CollectorConfig{IntervalNanos: 1, MaxRecordBudget: 2, MaxByteBudget: 1 << 20, MaxBufferedRecords: 2}
	s.buffer.Reset()
	s.retCollector = nvtelemetry.NewCollector("constructed-deadline", local, nvtelemetry.RealClock{}, config, nvtelemetry.NewWriter(&s.buffer, config))
	if err := s.retCollector.InitStream(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Sample(t.Context()); err != nil {
		t.Fatal(err)
	}
	s.handoffTimeout = 10 * time.Millisecond
	if _, err := s.Sample(t.Context()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("local deadline hidden: %v", err)
	}
	if !seen || !errors.Is(l.result(), context.DeadlineExceeded) {
		t.Fatalf("deadline not sticky: seen=%v error=%v", seen, l.result())
	}
	cleanup()
	if l.retention.Status().Stores != 0 || l.retained != s {
		t.Fatalf("incomplete source generation published as a healthy prefix: stores=%d retained=%p", l.retention.Status().Stores, l.retained)
	}
}

func TestServeCancelExpiredLocalHandoffRetainsGeneration(t *testing.T) {
	l, _, cfg := recoveryLifecycle(t)
	sampler, cleanup, err := l.sampler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	s := sampler.(*serveLifecycleSampler)
	if _, err := s.Sample(t.Context()); err != nil {
		t.Fatal(err)
	}
	readings, err := s.base.Sample(t.Context()) // real successful acquisition
	if err != nil || len(readings) == 0 {
		t.Fatalf("source acquisition: %v", err)
	}
	ctx, cancel := context.WithDeadline(t.Context(), time.Unix(0, 0))
	defer cancel()
	// This is the same local handoff called by Sample, with a deadline that
	// elapsed before Step's entry guard rather than inside its sampler.
	if err := s.retainSample(ctx, readings); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired handoff: %v", err)
	}
	cleanup()
	if l.retention.Status().Stores != 0 || l.retained != s || !errors.Is(l.result(), context.DeadlineExceeded) {
		t.Fatalf("expired local acquisition lost: stores=%d retained=%p error=%v", l.retention.Status().Stores, l.retained, l.result())
	}
}
