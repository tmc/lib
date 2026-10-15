package mxfp4

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

type cleanupOwner struct {
	calls int
	errs  []error
}

func (o *cleanupOwner) Close() error {
	o.calls++
	if len(o.errs) == 0 {
		return nil
	}
	err := o.errs[0]
	o.errs = o.errs[1:]
	return err
}

type cleanupLoadError struct {
	err   error
	owner *cleanupOwner
}

func (e *cleanupLoadError) Error() string { return e.err.Error() }
func (e *cleanupLoadError) Unwrap() error { return e.err }
func (e *cleanupLoadError) Close() error  { return e.owner.Close() }

func TestNewEngineWithReturnsNilWithoutLoadOwner(t *testing.T) {
	loadErr := errors.New("module load failed without an owner")
	e, err := newEngineWith(nil,
		func() (driver.Module, error) { return 0, loadErr },
		func(fn func() error) error { return fn() },
		func(driver.Module, string) (driver.Function, error) { return 0, nil },
	)
	if e != nil || !errors.Is(err, loadErr) {
		t.Fatalf("newEngineWith() = (%v, %v), want nil engine and load error", e, err)
	}
}

func TestNewEngineWithRetainsModuleReturnedWithLoadError(t *testing.T) {
	loadErr := errors.New("load returned a module and an error")
	var lookups int
	e, err := newEngineWith(nil,
		func() (driver.Module, error) { return 1, loadErr },
		func(fn func() error) error { return fn() },
		func(driver.Module, string) (driver.Function, error) { lookups++; return 0, nil },
	)
	if !errors.Is(err, loadErr) || e == nil || e.mod != 1 {
		t.Fatalf("newEngineWith() = (%v, %v), want a partial engine owning module 1 and load error", e, err)
	}
	if lookups != 0 {
		t.Fatalf("function lookups = %d, want 0 after load error", lookups)
	}

	var unloads int
	if err := e.closeWith(func(fn func() error) error { return fn() }, func(mod driver.Module) error {
		unloads++
		if mod != 1 {
			t.Fatalf("unload module = %d, want 1", mod)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if e.mod != 0 || unloads != 1 {
		t.Fatalf("after Close: mod = %d, unloads = %d, want 0 and 1", e.mod, unloads)
	}
}

func TestNewEngineWithRetainsRetryableLoadOwner(t *testing.T) {
	loadErr := errors.New("module load failed")
	initialCleanupErr := errors.New("initial unload failed")
	retryErr := errors.New("unload retry failed")
	owner := &cleanupOwner{errs: []error{retryErr, nil}}
	loadCleanup := &cleanupLoadError{err: errors.Join(loadErr, initialCleanupErr), owner: owner}
	e, err := newEngineWith(nil,
		func() (driver.Module, error) { return 0, loadCleanup },
		func(fn func() error) error { return fn() },
		func(driver.Module, string) (driver.Function, error) { return 0, nil },
	)
	if e == nil || !errors.Is(err, loadErr) || !errors.Is(err, initialCleanupErr) {
		t.Fatalf("newEngineWith() = (%v, %v), want partial owner and both load cleanup causes", e, err)
	}
	if e.loadOwner != loadCleanup {
		t.Fatal("newEngineWith did not retain the cleanup owner")
	}
	if err := e.closeWith(func(fn func() error) error { return fn() }, func(driver.Module) error {
		t.Fatal("unload called without a raw module")
		return nil
	}); !errors.Is(err, retryErr) {
		t.Fatalf("first Close error = %v, want %v", err, retryErr)
	}
	if e.loadOwner != loadCleanup || owner.calls != 1 {
		t.Fatalf("after failed Close: owner retained = %t, calls = %d", e.loadOwner == loadCleanup, owner.calls)
	}
	if err := e.closeWith(func(fn func() error) error { return fn() }, func(driver.Module) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if e.loadOwner != nil || owner.calls != 2 {
		t.Fatalf("after successful retry: owner = %v, calls = %d, want nil and 2", e.loadOwner, owner.calls)
	}
}

func TestNewEngineWithReturnsPartialEngineAfterFunctionLookupError(t *testing.T) {
	lookupErr := errors.New("function lookup failed")
	lookup := func(_ driver.Module, name string) (driver.Function, error) {
		if name == "MXFP4LinearGEMV" {
			return 0, lookupErr
		}
		return 2, nil
	}
	e, err := newEngineWith(nil,
		func() (driver.Module, error) { return 1, nil },
		func(fn func() error) error { return fn() }, lookup,
	)
	if e == nil || e.mod != 1 || !errors.Is(err, lookupErr) {
		t.Fatalf("newEngineWith() = (%v, %v), want partial engine retaining module 1 and lookup error", e, err)
	}
}

func TestEngineCloseRetiresModuleBeforeContextRestoreError(t *testing.T) {
	restoreErr := errors.New("context restoration failed")
	e := &Engine{mod: 1}
	var unloads int
	err := e.closeWith(func(fn func() error) error {
		if err := fn(); err != nil {
			return err
		}
		return restoreErr
	}, func(mod driver.Module) error {
		unloads++
		if mod != 1 {
			t.Fatalf("unload module = %d, want 1", mod)
		}
		return nil
	})
	if !errors.Is(err, restoreErr) || e.mod != 0 {
		t.Fatalf("Close() = %v, module = %d, want restore error and retired module", err, e.mod)
	}
	if err := e.closeWith(func(fn func() error) error { return fn() }, func(driver.Module) error {
		unloads++
		return fmt.Errorf("module was unloaded twice")
	}); err != nil {
		t.Fatal(err)
	}
	if unloads != 1 {
		t.Fatalf("unloads = %d, want 1", unloads)
	}
}

func TestEngineCloseRetainsModuleOnUnloadError(t *testing.T) {
	unloadErr := errors.New("module unload failed")
	restoreErr := errors.New("context restoration failed")
	e := &Engine{mod: 1}
	err := e.closeWith(func(fn func() error) error {
		if err := fn(); err != nil {
			return errors.Join(err, restoreErr)
		}
		return restoreErr
	}, func(driver.Module) error { return unloadErr })
	if !errors.Is(err, unloadErr) || !errors.Is(err, restoreErr) || e.mod != 1 {
		t.Fatalf("Close() = %v, module = %d, want both errors and retained module", err, e.mod)
	}
}
