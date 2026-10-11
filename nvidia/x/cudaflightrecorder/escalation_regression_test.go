package cudaflightrecorder

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"testing"
	"time"
)

type blockingUsage struct {
	entered chan struct{}
	release chan struct{}
}

func (s *blockingUsage) SampleUsage(ctx context.Context) (float64, uint64, error) {
	close(s.entered)
	select {
	case <-s.release:
		return 0, 0, nil
	case <-ctx.Done():
		return 0, 0, ctx.Err()
	}
}

func TestEscalationReservesAdmissionBeforeSampling(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	budgets := QualificationBudgets()
	budgets.HysteresisCount = 1
	sampler := &blockingUsage{make(chan struct{}), make(chan struct{})}
	coord, err := NewEscalationCoordinator(rec, budgets, WithResourceSampler(sampler), WithCUPTINegotiator(&FakeCUPTINegotiator{}))
	if err != nil {
		t.Fatal(err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := coord.Trigger(context.Background(), TriggerRequest{TenantID: "a"}); done <- err }()
	<-sampler.entered
	receipt, err := coord.Trigger(context.Background(), TriggerRequest{TenantID: "b"})
	if !errors.Is(err, ErrEscalationRefused) || receipt.RefusalReason != RefusalConcurrencyLimit {
		t.Fatalf("receipt=%+v err=%v", receipt, err)
	}
	if err := coord.Stop(); err != nil {
		t.Fatal(err)
	}
	close(sampler.release)
	<-done
	if coord.State() != StateStopped {
		t.Fatalf("state=%s", coord.State())
	}
	if coord.activeCaptures != 0 || len(coord.activeTenants) != 0 {
		t.Fatal("reservation leaked")
	}
}

type closeCountingFactory struct {
	mu     sync.Mutex
	closed int
	err    error
}

func (f *closeCountingFactory) CreateWriter(id, tenant string) (io.Writer, func() error, error) {
	return new(bytes.Buffer), func() error { f.mu.Lock(); defer f.mu.Unlock(); f.closed++; return f.err }, nil
}
func TestEscalationClosesWriterOnce(t *testing.T) {
	for _, closeErr := range []error{nil, errors.New("close failure")} {
		rec, err := New(Config{Capacity: 10})
		if err != nil {
			t.Fatal(err)
		}
		budgets := QualificationBudgets()
		budgets.HysteresisCount = 1
		factory := &closeCountingFactory{err: closeErr}
		coord, err := NewEscalationCoordinator(rec, budgets, WithResourceSampler(&FakeResourceSampler{}), WithCUPTINegotiator(&FakeCUPTINegotiator{}), WithCaptureWriterFactory(factory))
		if err != nil {
			t.Fatal(err)
		}
		if err := coord.Arm(); err != nil {
			t.Fatal(err)
		}
		_, err = coord.Trigger(context.Background(), TriggerRequest{})
		if closeErr != nil && !errors.Is(err, closeErr) {
			t.Fatalf("close error lost: %v", err)
		}
		if factory.closed != 1 {
			t.Fatalf("closed=%d", factory.closed)
		}
	}
}

type shortBundleWriter struct{}

func (shortBundleWriter) Write(p []byte) (int, error) { return len(p) / 2, nil }
func TestByteCountingWriterRejectsShortWrite(t *testing.T) {
	w := &byteCountingWriter{w: shortBundleWriter{}, limit: 10}
	n, err := w.Write([]byte("abcd"))
	if n != 2 || !errors.Is(err, io.ErrShortWrite) || w.written != 2 {
		t.Fatalf("n=%d err=%v written=%d", n, err, w.written)
	}
}

func TestEscalationRingBudgetAdmission(t *testing.T) {
	rec, err := New(Config{Capacity: 100})
	if err != nil {
		t.Fatal(err)
	}
	budgets := QualificationBudgets()
	budgets.MaxRingRecords = 10
	if _, err := NewEscalationCoordinator(rec, budgets); err == nil {
		t.Fatal("oversized recorder admitted")
	}
}

func TestEscalationRealClockUsesElapsedTime(t *testing.T) {
	clock := RealClock{}
	start := clock.NowMonotonic()
	if start < 0 || start > time.Hour.Nanoseconds() {
		t.Fatalf("monotonic time looks like wall time: %d", start)
	}
	time.Sleep(time.Millisecond)
	if clock.NowMonotonic() <= start {
		t.Fatal("clock did not advance")
	}
}

func TestEscalationRefusesOverlappingSnapshots(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	budgets := QualificationBudgets()
	budgets.MaxConcurrentCaptures = 2
	if _, err := NewEscalationCoordinator(rec, budgets); err == nil {
		t.Fatal("unsafe concurrent snapshots admitted")
	}
}

func TestDirectoryWriterNamesCannotEscape(t *testing.T) {
	dir := t.TempDir()
	factory := NewDirectoryWriterFactory(dir)
	writer, closeFn, err := factory.CreateWriter("../../../outside", "../../tenant/../../../")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Write([]byte("capture")); err != nil {
		t.Fatal(err)
	}
	if err := closeFn(); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].IsDir() {
		t.Fatalf("entries=%v", entries)
	}
}

func TestEscalationHysteresisIsTenantScoped(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	budgets := QualificationBudgets()
	budgets.HysteresisCount = 2
	coord, err := NewEscalationCoordinator(rec, budgets, WithResourceSampler(&FakeResourceSampler{}))
	if err != nil {
		t.Fatal(err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatal(err)
	}
	for _, tenant := range []string{"a", "b"} {
		receipt, err := coord.Trigger(context.Background(), TriggerRequest{TenantID: tenant, MetricName: "latency"})
		if !errors.Is(err, ErrEscalationRefused) || receipt.RefusalReason != RefusalHysteresisNotMet {
			t.Fatalf("tenant=%s receipt=%+v err=%v", tenant, receipt, err)
		}
	}
}

func TestEscalationMetadataTrackingBounded(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	budgets := QualificationBudgets()
	budgets.HysteresisCount = 2
	coord, err := NewEscalationCoordinator(rec, budgets, WithResourceSampler(&FakeResourceSampler{}))
	if err != nil {
		t.Fatal(err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1100; i++ {
		_, _ = coord.Trigger(context.Background(), TriggerRequest{TenantID: fmt.Sprintf("tenant-%d", i), MetricName: "latency"})
	}
	if len(coord.triggerHistory) > 1024 || len(coord.tenantHourlyCaptures) > 1024 {
		t.Fatalf("unbounded maps: triggers=%d tenants=%d", len(coord.triggerHistory), len(coord.tenantHourlyCaptures))
	}
}

func TestNilEscalationDependencies(t *testing.T) {
	rec, err := New(Config{Capacity: 8})
	if err != nil {
		t.Fatal(err)
	}
	var sampler *FakeResourceSampler
	var provider *CUPTIActivityCapture
	for _, option := range []EscalationOption{nil, WithResourceSampler(nil), WithResourceSampler(sampler), WithIncidentCapture(provider)} {
		if _, err := NewEscalationCoordinator(rec, Budgets{}, option); err == nil {
			t.Fatal("nil dependency accepted")
		}
	}
}
