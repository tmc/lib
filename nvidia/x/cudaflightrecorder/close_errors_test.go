package cudaflightrecorder

import (
	"context"
	"errors"
	"io"
	"testing"
)

type failedBundleWriter struct{ err error }

func (w failedBundleWriter) Write([]byte) (int, error) { return 0, w.err }

type failedBundleFactory struct {
	writeErr, closeErr error
	closes             int
}

func (f *failedBundleFactory) CreateWriter(string, string) (io.Writer, func() error, error) {
	return failedBundleWriter{f.writeErr}, func() error { f.closes++; return f.closeErr }, nil
}

func TestEscalationPreservesWriteAndCloseErrors(t *testing.T) {
	rec, err := New(Config{Capacity: 10})
	if err != nil {
		t.Fatal(err)
	}
	budgets := QualificationBudgets()
	budgets.HysteresisCount = 1
	writeErr, closeErr := errors.New("write failure"), errors.New("close failure")
	factory := &failedBundleFactory{writeErr: writeErr, closeErr: closeErr}
	coordinator, err := NewEscalationCoordinator(rec, budgets, WithResourceSampler(&FakeResourceSampler{}), WithCUPTINegotiator(&FakeCUPTINegotiator{}), WithCaptureWriterFactory(factory))
	if err != nil {
		t.Fatal(err)
	}
	defer coordinator.Stop()
	if err := coordinator.Arm(); err != nil {
		t.Fatal(err)
	}
	receipt, err := coordinator.Trigger(context.Background(), TriggerRequest{})
	if !errors.Is(err, writeErr) || !errors.Is(err, closeErr) || receipt.Outcome != "failed" || factory.closes != 1 {
		t.Fatalf("receipt=%+v err=%v closes=%d", receipt, err, factory.closes)
	}
}
