package cudahotswap

import (
	"context"
	"errors"
	"io"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// All kernel and graph operations in this file are host simulations.
func TestKernelCanceledCheckerRestoresGraphAndRetainsSubmittedCandidate(t *testing.T) {
	original := &serviceKernel{raw: 1}
	loader := &serviceLoader{kernel: original}
	s, _ := testService(t, loader)
	registerService(t, s)
	candidate := &serviceKernel{raw: 2}
	loader.kernel = candidate
	var installed driver.Function
	s.graphs.ops.patch = func(_ *cudagraph.Exec, _ *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		installed = fn
		return nil
	}
	if _, err := s.graphs.bind("kernel", new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream), driver.Dims(1, 1, 1), nil); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var pending *PendingLaunch
	rollbackAcquired := false
	s.acquire = func(acquireCtx context.Context, _ string) (io.Closer, error) {
		if err := acquireCtx.Err(); err != nil {
			return nil, err
		}
		if ctx.Err() != nil && !rollbackAcquired {
			deadline, ok := acquireCtx.Deadline()
			if !ok || time.Until(deadline) > 30*time.Second {
				t.Fatal("rollback acquire is not cooperatively bounded")
			}
			rollbackAcquired = true
		}
		return new(serviceCloser), nil
	}
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(ctx context.Context, _ State) error {
		var err error
		pending, err = s.Launch(ctx, "kernel", cudalaunch.Config{})
		if err != nil {
			return err
		}
		cancel()
		return context.Canceled
	}}, "cancel"); err != nil {
		t.Fatal(err)
	}
	result, err := s.Push(ctx, "kernel", Source{Code: "candidate", Entry: "entry"}, "unchecked")
	if !errors.Is(err, context.Canceled) || result == nil || !result.Published || !result.RolledBack || result.Selected.Version != 1 || installed != 1 {
		t.Fatalf("result=%+v err=%v installed=%d", result, err, installed)
	}
	if pending == nil || pending.kernel != candidate || candidate.calls != 0 || original.calls != 0 {
		t.Fatal("rollback invalidated submitted candidate owner")
	}
	next, err := s.Launch(t.Context(), "kernel", cudalaunch.Config{})
	if err != nil || next.kernel != original {
		t.Fatalf("new request uses unchecked candidate: %v", err)
	}
	if err := next.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := pending.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestKernelRollbackReservationSurvivesOtherName(t *testing.T) {
	for _, tc := range []struct {
		name      string
		remaining uint64
		wantOther bool
	}{{"enough", 3, true}, {"reserved boundary", 2, false}} {
		t.Run(tc.name, func(t *testing.T) {
			loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
			s, _ := testService(t, loader)
			registerService(t, s)
			loader.kernel = &serviceKernel{raw: 2}
			var other State
			var otherErr error
			if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(ctx context.Context, _ State) error {
				loader.kernel = &serviceKernel{raw: 3}
				other, otherErr = s.Register(ctx, "other", Source{Code: "other", Entry: "entry"}, "other")
				return errors.New("refused")
			}}, "reentrant"); err != nil {
				t.Fatal(err)
			}
			s.table.mu.Lock()
			s.table.next = math.MaxUint64 - tc.remaining
			s.table.mu.Unlock()
			r, err := s.Push(t.Context(), "kernel", Source{Code: "candidate", Entry: "entry"}, "unchecked")
			if err == nil || r == nil || !r.Published || !r.RolledBack || r.Selected.Version != 1 || r.Selected.Generation != math.MaxUint64 || (otherErr == nil) != tc.wantOther {
				t.Fatalf("r=%+v err=%v other=%+v otherErr=%v", r, err, other, otherErr)
			}
			s.table.mu.RLock()
			reserved := len(s.table.rollback)
			s.table.mu.RUnlock()
			if reserved != 0 {
				t.Fatal("reservation leaked")
			}
			if err := s.Close(t.Context()); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestKernelSuccessfulCheckerReleasesReservation(t *testing.T) {
	loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, _ := testService(t, loader)
	registerService(t, s)
	loader.kernel = &serviceKernel{raw: 2}
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { return nil }}, "accept"); err != nil {
		t.Fatal(err)
	}
	s.table.mu.Lock()
	s.table.next = math.MaxUint64 - 2
	s.table.mu.Unlock()
	if r, err := s.Push(t.Context(), "kernel", Source{Code: "candidate", Entry: "entry"}, "checked"); err != nil || r == nil || !r.Published || r.RolledBack {
		t.Fatalf("r=%+v err=%v", r, err)
	}
	loader.kernel = &serviceKernel{raw: 3}
	other, err := s.Register(t.Context(), "other", Source{Code: "other", Entry: "entry"}, "other")
	if err != nil || other.Generation != math.MaxUint64 {
		t.Fatalf("other=%+v err=%v", other, err)
	}
	if err := s.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestKernelCanceledCheckerGraphPatchFailureNotReportedAsRollback(t *testing.T) {
	loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, _ := testService(t, loader)
	registerService(t, s)
	loader.kernel = &serviceKernel{raw: 2}
	fail := false
	patchErr := errors.New("host graph rollback failure")
	var installed driver.Function
	s.graphs.ops.patch = func(_ *cudagraph.Exec, _ *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		installed = fn
		if fail && fn == 1 {
			return patchErr
		}
		return nil
	}
	if _, err := s.graphs.bind("kernel", new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream), driver.Dims(1, 1, 1), nil); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { fail = true; cancel(); return context.Canceled }}, "cancel"); err != nil {
		t.Fatal(err)
	}
	r, err := s.Push(ctx, "kernel", Source{Code: "candidate", Entry: "entry"}, "unchecked")
	if r == nil || !r.Published || r.RolledBack || r.Selected.Version != 2 || !errors.Is(err, patchErr) || !errors.Is(err, context.Canceled) || installed != 2 {
		t.Fatalf("r=%+v err=%v installed=%d", r, err, installed)
	}
	fail = false
	if err := s.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}
