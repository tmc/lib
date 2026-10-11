package cudaflightrecorder

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

type controlledActivitySession struct {
	result   cuptiactivity.Result
	closeErr error
	closes   int
}

func (s *controlledActivitySession) Close() error {
	s.closes++
	if s.closeErr == nil {
		s.result.Closed = true
	}
	return s.closeErr
}
func (s *controlledActivitySession) Result() cuptiactivity.Result { return s.result }

func testActivityCapture(t *testing.T, session incidentActivitySession, startErr error) *CUPTIActivityCapture {
	t.Helper()
	p, err := NewCUPTIActivityCapture(CUPTIActivityCaptureConfig{
		Activities:         cuptiactivity.Config{Kinds: []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel}, BufferBytes: 4096, MaxBuffers: 2, MaxRecords: 10},
		CollectionWindow:   time.Millisecond,
		ExclusiveOwnership: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	p.start = func(c cuptiactivity.Config) (incidentActivitySession, error) { return session, startErr }
	return p
}
func testActivityRecorder(t *testing.T) *Recorder {
	t.Helper()
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { rec.Stop() })
	return rec
}

func TestIncidentActivityCaptureRetainsCleanupAndLease(t *testing.T) {
	rec := testActivityRecorder(t)
	closeErr := errors.New("native flush failed")
	session := &controlledActivitySession{closeErr: closeErr}
	p := testActivityCapture(t, session, nil)
	released := 0
	report, err := p.Capture(context.Background(), rec, TriggerRequest{}, func() { released++ })
	if !errors.Is(err, closeErr) || !report.Started || !report.RetainedOwnership || session.closes != 1 || released != 0 {
		t.Fatalf("report=%+v err=%v closes=%d releases=%d", report, err, session.closes, released)
	}
	if _, err := p.Capture(context.Background(), rec, TriggerRequest{}, func() {}); err == nil {
		t.Fatal("pending cleanup replaced")
	}
	session.closeErr = nil
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if session.closes != 2 || released != 1 {
		t.Fatalf("closes=%d releases=%d", session.closes, released)
	}
}

func TestIncidentActivityStartupFailurePreservesCleanup(t *testing.T) {
	rec := testActivityRecorder(t)
	startErr := errors.New("activity enable failed")
	session := &controlledActivitySession{closeErr: errors.New("rollback failed")}
	p := testActivityCapture(t, session, startErr)
	report, err := p.Capture(context.Background(), rec, TriggerRequest{}, func() {})
	if !errors.Is(err, startErr) || report.Started || !report.RetainedOwnership || session.closes != 0 {
		t.Fatalf("report=%+v err=%v", report, err)
	}
	session.closeErr = nil
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestIncidentActivityCancellationClosesSession(t *testing.T) {
	rec := testActivityRecorder(t)
	session := &controlledActivitySession{}
	p := testActivityCapture(t, session, nil)
	ctx, cancel := context.WithCancel(context.Background())
	p.start = func(c cuptiactivity.Config) (incidentActivitySession, error) { cancel(); return session, nil }
	report, err := p.Capture(ctx, rec, TriggerRequest{}, func() {})
	if !errors.Is(err, context.Canceled) || !report.Closed || report.RetainedOwnership || session.closes != 1 {
		t.Fatalf("report=%+v err=%v", report, err)
	}
}

func TestIncidentActivityMissingLayoutRefusesBeforeCapture(t *testing.T) {
	rec := testActivityRecorder(t)
	layoutErr := &cuptiactivity.LayoutError{}
	p := testActivityCapture(t, nil, layoutErr)
	report, err := p.Capture(context.Background(), rec, TriggerRequest{}, func() {})
	if !errors.Is(err, layoutErr) || report.Started || report.RetainedOwnership {
		t.Fatalf("report=%+v err=%v", report, err)
	}
}

func TestIncidentActivityReportsLossesAndRawEvidence(t *testing.T) {
	rec := testActivityRecorder(t)
	session := &controlledActivitySession{result: cuptiactivity.Result{
		Records: []cuptiactivity.Record{{Kind: cupti.ActivityKindConcurrentKernel, Start: 10, End: 20, Complete: true, Name: "kernel"}}, DroppedRecords: 2,
	}}
	p := testActivityCapture(t, session, nil)
	report, err := p.Capture(context.Background(), rec, TriggerRequest{}, func() {})
	if err == nil || report.DroppedRecords != 2 || report.Records != 1 || !report.Closed {
		t.Fatalf("report=%+v err=%v", report, err)
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.extraFiles["incident-activity.json"]) == 0 || rec.count != 1 {
		t.Fatal("partial activity evidence omitted")
	}
}

func TestIncidentActivityRequiresExplicitOwnership(t *testing.T) {
	if _, err := NewCUPTIActivityCapture(CUPTIActivityCaptureConfig{}); err == nil {
		t.Fatal("missing ownership admitted")
	}
}

func TestTriggerRunsConfiguredActivityProvider(t *testing.T) {
	rec := testActivityRecorder(t)
	session := &controlledActivitySession{}
	provider := testActivityCapture(t, session, nil)
	factory := NewBufferWriterFactory()
	budgets := QualificationBudgets()
	budgets.HysteresisCount = 1
	coord, err := NewEscalationCoordinator(rec, budgets,
		WithResourceSampler(&FakeResourceSampler{}), WithCUPTINegotiator(&FakeCUPTINegotiator{}),
		WithCaptureWriterFactory(factory), WithIncidentCapture(provider))
	if err != nil {
		t.Fatal(err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatal(err)
	}
	receipt, err := coord.Trigger(context.Background(), TriggerRequest{Reason: "incident"})
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Capture.Started || !receipt.Capture.Closed || receipt.Capture.Provider != "cupti_activity" || session.closes != 1 {
		t.Fatalf("receipt=%+v closes=%d", receipt, session.closes)
	}
	if !bytes.Contains(factory.Buffers[receipt.IncidentID].Bytes(), []byte("incident-activity.json")) {
		t.Fatal("activity evidence missing from bundle")
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	if _, exists := rec.extraFiles["incident-activity.json"]; exists {
		t.Fatal("incident evidence leaked into later snapshots")
	}
}

type incidentLeaseControl struct{ releases int }

func (*incidentLeaseControl) CheckCompatibility(context.Context, uint32) error { return nil }
func (c *incidentLeaseControl) AcquireLease(context.Context, uint32, string) (func(), error) {
	return func() { c.releases++ }, nil
}

func TestTriggerRetainsLeaseUntilNativeCleanupSucceeds(t *testing.T) {
	rec := testActivityRecorder(t)
	closeErr := errors.New("flush failure")
	session := &controlledActivitySession{closeErr: closeErr}
	provider := testActivityCapture(t, session, nil)
	leases := &incidentLeaseControl{}
	budgets := QualificationBudgets()
	budgets.HysteresisCount = 1
	coord, err := NewEscalationCoordinator(rec, budgets,
		WithResourceSampler(&FakeResourceSampler{}), WithCUPTINegotiator(leases), WithIncidentCapture(provider))
	if err != nil {
		t.Fatal(err)
	}
	if err := coord.Arm(); err != nil {
		t.Fatal(err)
	}
	receipt, err := coord.Trigger(context.Background(), TriggerRequest{})
	if !errors.Is(err, closeErr) || receipt.Outcome != "failed" || !receipt.Capture.RetainedOwnership || leases.releases != 0 {
		t.Fatalf("receipt=%+v err=%v releases=%d", receipt, err, leases.releases)
	}
	session.closeErr = nil
	if err := provider.Close(); err != nil {
		t.Fatal(err)
	}
	if leases.releases != 1 {
		t.Fatalf("releases=%d", leases.releases)
	}
}
