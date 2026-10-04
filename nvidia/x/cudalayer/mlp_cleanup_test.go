package cudalayer

import (
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

type mlpTestCloser struct {
	calls int
	err   error
}

func TestMLPBackward_FreeCommitsBeforeContextRestoration(t *testing.T) {
	restoreFailure := errors.New("restore context failed")
	runs, frees := 0, 0
	free := mlpFreeBuffer(123, func(fn func() error) error { runs++; return errors.Join(fn(), restoreFailure) }, func(pointer driver.DevicePtr) error {
		if pointer != 123 {
			t.Fatalf("free pointer=%d", pointer)
		}
		frees++
		return nil
	})
	owner := &mlpCleanup{frees: []func() error{free}}
	if err := owner.Close(); !errors.Is(err, restoreFailure) {
		t.Fatalf("restoration failure lost: %v", err)
	}
	if frees != 1 || runs != 1 {
		t.Fatal("native free was not executed")
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if frees != 1 || runs != 1 {
		t.Fatalf("successful native free repeated: frees=%d runs=%d", frees, runs)
	}
}

func TestMLPBackward_FreeFailureRetainsNativePointer(t *testing.T) {
	nativeFailure := errors.New("native free failed")
	err := nativeFailure
	frees := 0
	free := mlpFreeBuffer(123, func(fn func() error) error { return fn() }, func(pointer driver.DevicePtr) error {
		if pointer != 123 {
			t.Fatalf("retry pointer=%d", pointer)
		}
		frees++
		return err
	})
	if e := free(); !errors.Is(e, nativeFailure) {
		t.Fatal("native failure hidden")
	}
	err = nil
	if e := free(); e != nil {
		t.Fatal(e)
	}
	if e := free(); e != nil {
		t.Fatal(e)
	}
	if frees != 2 {
		t.Fatalf("native calls=%d want failed attempt plus one successful retry", frees)
	}
}

func (c *mlpTestCloser) Close() error { c.calls++; return c.err }

func TestMLPBackward_CleanupRetainsUntilCompletion(t *testing.T) {
	failed := errors.New("device completion unavailable")
	syncErr := failed
	syncCalls, freeCalls := 0, 0
	module := &mlpTestCloser{}
	owner := &mlpCleanup{inFlight: true, synchronize: func() error { syncCalls++; return syncErr }, frees: []func() error{func() error { freeCalls++; return nil }}, closers: []io.Closer{module}}
	if e := owner.Close(); !errors.Is(e, failed) {
		t.Fatalf("synchronize error=%v", e)
	}
	if freeCalls != 0 || module.calls != 0 || !owner.inFlight {
		t.Fatalf("unconfirmed work freed: buffers=%d module=%d inflight=%t", freeCalls, module.calls, owner.inFlight)
	}
	launchErr := errors.New("backward launch failed")
	wrapped := &mlpCleanupError{cause: errors.Join(launchErr, failed), owner: owner}
	var retry io.Closer
	if !errors.As(fmt.Errorf("run: %w", wrapped), &retry) || !errors.Is(wrapped, launchErr) || !errors.Is(wrapped, failed) {
		t.Fatal("execution/cleanup causes or retry owner lost")
	}
	syncErr = nil
	if e := retry.Close(); e != nil {
		t.Fatal(e)
	}
	if freeCalls != 1 || module.calls != 1 || owner.inFlight || syncCalls != 2 {
		t.Fatalf("retry did not complete exact cleanup: %d/%d/%d", freeCalls, module.calls, syncCalls)
	}
	if e := retry.Close(); e != nil {
		t.Fatal(e)
	}
	if freeCalls != 1 || module.calls != 1 {
		t.Fatal("successful resources freed twice")
	}
}

func TestMLPBackward_CleanupPreservesAllFailureCausesAndRetries(t *testing.T) {
	freeFailure, moduleFailure := errors.New("free failed"), errors.New("module close failed")
	freeErr := freeFailure
	freeCalls, successCalls := 0, 0
	module := &mlpTestCloser{err: moduleFailure}
	owner := &mlpCleanup{frees: []func() error{func() error { freeCalls++; return freeErr }, func() error { successCalls++; return nil }}, closers: []io.Closer{module}}
	e := owner.Close()
	if !errors.Is(e, freeFailure) || !errors.Is(e, moduleFailure) {
		t.Fatalf("cleanup causes lost: %v", e)
	}
	freeErr = nil
	module.err = nil
	if e := owner.Close(); e != nil {
		t.Fatal(e)
	}
	if freeCalls != 2 || successCalls != 1 || module.calls != 2 {
		t.Fatal("retry did not retain only failed resources")
	}
}

type mlpPendingError struct {
	cause  error
	closer *mlpTestCloser
}

func (e *mlpPendingError) Close() error  { return e.closer.Close() }
func (e *mlpPendingError) Error() string { return e.cause.Error() }
func (e *mlpPendingError) Unwrap() error { return e.cause }

func TestMLPBackward_AdoptsCompilerAndLoaderCleanup(t *testing.T) {
	for _, stage := range []string{"compile", "load"} {
		t.Run(stage, func(t *testing.T) {
			original := errors.New(stage + " failure")
			retryFailure := errors.New(stage + " cleanup failure")
			c := &mlpTestCloser{err: retryFailure}
			err := mlpAdoptCleanup(fmt.Errorf("%s: %w", stage, &mlpPendingError{cause: original, closer: c}))
			var retry io.Closer
			if !errors.As(err, &retry) || !errors.Is(err, original) {
				t.Fatal("underlying failure owner lost")
			}
			if e := retry.Close(); !errors.Is(e, retryFailure) {
				t.Fatal("cleanup failure hidden")
			}
			c.err = nil
			if e := retry.Close(); e != nil {
				t.Fatal(e)
			}
			if c.calls != 2 {
				t.Fatal("cleanup was recreated or retried incorrectly")
			}
		})
	}
}

func TestMLPBackward_CompositeRequiresOwnedChildOffsets(t *testing.T) {
	m, e := NewFusedMLP(5, 7, 3, 2)
	if e != nil {
		t.Fatal(e)
	}
	if _, e := m.EmitBackward("unsafe_direct", 32); e == nil {
		t.Fatal("direct composite emission with unassigned context offsets accepted")
	}
}
