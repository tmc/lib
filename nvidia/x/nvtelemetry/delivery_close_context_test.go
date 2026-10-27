package nvtelemetry_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type closeContextTransport func(*http.Request) (*http.Response, error)

func (f closeContextTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

var closeContextSignals = []struct {
	name, payload string
	new           func(nvtelemetry.DeliveryConfig) (*nvtelemetry.Delivery, error)
}{
	{"metrics", `{"resourceMetrics":[{"resource":{"attributes":[]},"scopeMetrics":[{"scope":{"name":"close-control"},"metrics":[{"name":"control","unit":"1","gauge":{"dataPoints":[{"timeUnixNano":"10","asDouble":1}]}}]}]}]}`, nvtelemetry.NewDelivery},
	{"logs", `{"resourceLogs":[{"resource":{"attributes":[]},"scopeLogs":[{"scope":{"name":"close-control"},"logRecords":[{"timeUnixNano":"10","body":{"stringValue":"controlled host record"}}]}]}]}`, func(c nvtelemetry.DeliveryConfig) (*nvtelemetry.Delivery, error) {
		c.Endpoint = "https://delivery.invalid/v1/logs"
		return nvtelemetry.NewLogsDelivery(c)
	}},
}

func closeContextConfig(transport http.RoundTripper) nvtelemetry.DeliveryConfig {
	return nvtelemetry.DeliveryConfig{Endpoint: "https://delivery.invalid/v1/metrics", Client: &http.Client{Transport: transport}, MaxQueuedBatches: 1, MaxQueuedBytes: 4096, MaxBatchBytes: 4096, MaxBatchDataPoints: 4, MaxResponseBytes: 512, MaxAttempts: 1, BatchTimeout: 10 * time.Second, RequestTimeout: 10 * time.Second, ShutdownTimeout: 5 * time.Second}
}
func closeContextResponse(r *http.Request, status int) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`)), Request: r}
}
func closeContextReceive[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(2 * time.Second):
		t.Fatal("controlled delivery did not complete")
		var zero T
		return zero
	}
}

// Public admission is the ordering barrier: while the first request is blocked,
// capacity1 prevents another item from being admitted. ErrDeliveryClosed proves
// the first Close has taken ownership of the shared drain before a waiter starts.
func waitCloseContextAdmission(t *testing.T, d *nvtelemetry.Delivery, payload []byte) {
	t.Helper()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		err := d.Enqueue(context.Background(), payload)
		if errors.Is(err, nvtelemetry.ErrDeliveryClosed) {
			return
		}
		if !errors.Is(err, nvtelemetry.ErrDeliveryFull) {
			t.Fatal("unexpected admission state", err)
		}
		select {
		case <-tick.C:
		case <-timer.C:
			t.Fatal("first Close did not stop admission")
		}
	}
}
func TestDeliveryLaterWaiterDoesNotCancelDrain(t *testing.T) {
	for _, signal := range closeContextSignals {
		for _, kind := range []string{"canceled", "deadline"} {
			t.Run(signal.name+"-"+kind, func(t *testing.T) {
				started := make(chan struct{})
				release := make(chan struct{})
				var once sync.Once
				transport := closeContextTransport(func(r *http.Request) (*http.Response, error) {
					close(started)
					select {
					case <-release:
					case <-r.Context().Done():
						return nil, r.Context().Err()
					}
					if err := r.Context().Err(); err != nil {
						return nil, err
					}
					return closeContextResponse(r, http.StatusOK), nil
				})
				d, err := signal.new(closeContextConfig(transport))
				if err != nil {
					t.Fatal(err)
				}
				owner, ownerCancel := context.WithTimeout(context.Background(), 3*time.Second)
				t.Cleanup(func() {
					ownerCancel()
					once.Do(func() { close(release) })
					ctx, cancel := context.WithTimeout(context.Background(), time.Second)
					defer cancel()
					_ = d.Close(ctx)
				})
				payload := []byte(signal.payload)
				if err := d.Enqueue(context.Background(), payload); err != nil {
					t.Fatal(err)
				}
				closeContextReceive(t, started)
				first := make(chan error, 1)
				go func() { first <- d.Close(owner) }()
				waitCloseContextAdmission(t, d, payload)
				var waiter context.Context
				var cancel context.CancelFunc
				if kind == "canceled" {
					waiter, cancel = context.WithCancel(context.Background())
					cancel()
				} else {
					waiter, cancel = context.WithTimeout(context.Background(), 10*time.Millisecond)
					defer cancel()
				}
				err = d.Close(waiter)
				if !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) || !errors.Is(err, waiter.Err()) {
					t.Fatal("later waiter lost its own cancellation", err, waiter.Err())
				}
				once.Do(func() { close(release) })
				if err := closeContextReceive(t, first); err != nil {
					t.Fatal("later waiter canceled the first caller's drain", err)
				}
				stats := d.Stats()
				if stats.EnqueuedBatches != 1 || stats.SucceededBatches != 1 || stats.FailedBatches != 0 || stats.PendingBatches != 0 || stats.PendingBytes != 0 {
					t.Fatal("valid owner drain lost its admitted batch", stats)
				}
			})
		}
	}
}
func TestDeliveryFirstCallerStillOwnsCancellation(t *testing.T) {
	for _, signal := range closeContextSignals {
		t.Run(signal.name, func(t *testing.T) {
			started := make(chan struct{})
			canceled := make(chan struct{})
			d, err := signal.new(closeContextConfig(closeContextTransport(func(r *http.Request) (*http.Response, error) {
				close(started)
				<-r.Context().Done()
				close(canceled)
				return nil, r.Context().Err()
			})))
			if err != nil {
				t.Fatal(err)
			}
			owner, cancel := context.WithCancel(context.Background())
			defer cancel()
			t.Cleanup(func() {
				cancel()
				ctx, stop := context.WithTimeout(context.Background(), time.Second)
				defer stop()
				_ = d.Close(ctx)
			})
			if err := d.Enqueue(context.Background(), []byte(signal.payload)); err != nil {
				t.Fatal(err)
			}
			closeContextReceive(t, started)
			result := make(chan error, 1)
			go func() { result <- d.Close(owner) }()
			waitCloseContextAdmission(t, d, []byte(signal.payload))
			cancel()
			if err := closeContextReceive(t, result); !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
				t.Fatal("first cancellation reported success", err)
			}
			closeContextReceive(t, canceled)
			if err := d.Close(context.Background()); !errors.Is(err, nvtelemetry.ErrDeliveryIncomplete) {
				t.Fatal("failed owner drain was not retained", err)
			}
			// Incomplete Close bounds waiting, not worker completion. The transport
			// signals cancellation before it returns, so final counters may still be
			// pending even after that signal and another incomplete Close.
			stats := waitCloseContextPending(t, d)
			if stats.SucceededBatches != 0 || stats.FailedBatches != 1 || stats.PendingBatches != 0 {
				t.Fatal("first cancellation outcome lost", stats)
			}
		})
	}
}
func TestDeliveryCompletedCloseIgnoresWaiterContext(t *testing.T) {
	for _, signal := range closeContextSignals {
		for _, status := range []int{http.StatusOK, http.StatusForbidden} {
			name := signal.name + "-success"
			if status != http.StatusOK {
				name = signal.name + "-failure"
			}
			t.Run(name, func(t *testing.T) {
				d, err := signal.new(closeContextConfig(closeContextTransport(func(r *http.Request) (*http.Response, error) { return closeContextResponse(r, status), nil })))
				if err != nil {
					t.Fatal(err)
				}
				if err := d.Enqueue(context.Background(), []byte(signal.payload)); err != nil {
					t.Fatal(err)
				}
				owner, ownerCancel := context.WithCancel(context.Background())
				original := d.Close(owner)
				ownerCancel()
				if status == http.StatusOK && original != nil || status != http.StatusOK && !errors.Is(original, nvtelemetry.ErrDeliveryIncomplete) {
					t.Fatal("unexpected stored outcome", original)
				}
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				// Once the drain has completed, the outcome is stable even when both the
				// done channel and each caller's cancellation channel are already ready.
				for i := 0; i < 64; i++ {
					runtime.Gosched()
					if got := d.Close(ctx); got != original {
						t.Fatal("completed Close outcome depends on waiter context", i, got, original)
					}
				}
			})
		}
	}
}

// Public pending counters are the completion witness for final batch accounting.
// Request cancellation alone is not that witness.
func waitCloseContextPending(t *testing.T, d *nvtelemetry.Delivery) nvtelemetry.DeliveryStats {
	t.Helper()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		stats := d.Stats()
		if stats.PendingBatches == 0 && stats.PendingBytes == 0 {
			return stats
		}
		select {
		case <-tick.C:
		case <-timer.C:
			t.Fatal("controlled delivery accounting remained pending", stats)
		}
	}
}
