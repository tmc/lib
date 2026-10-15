package cudadiagdemo

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

type nativeOwnerFixture struct{}

func TestNativeConstructorRetainsErrorResultAndRetriesCleanup(t *testing.T) {
	constructorErr := errors.New("constructor failed after creating owner")
	closeErr := errors.New("partial owner close failed")
	var calls []string
	cleanup := &nativeCleanup{}
	cleanup.add("lease", func() error { calls = append(calls, "lease"); return nil })
	cleanup.add("device", func() error { calls = append(calls, "device"); return nil })
	expectedOwner := &nativeOwnerFixture{}
	got, err := constructNativeOwner(cleanup, "partial module", func() (*nativeOwnerFixture, error) {
		return expectedOwner, constructorErr
	}, func(owner *nativeOwnerFixture) bool {
		return owner != nil
	}, func(owner **nativeOwnerFixture) error {
		if *owner != expectedOwner {
			t.Fatal("cleanup received a different owner")
		}
		calls = append(calls, "partial module")
		if len(calls) == 1 {
			return closeErr
		}
		return nil
	})
	if got != expectedOwner || !errors.Is(err, constructorErr) {
		t.Fatalf("constructed owner/error = %p, %v", got, err)
	}

	result := &NativeResult{cleanup: cleanup}
	joined := joinNativeCleanup(result, err)
	if !errors.Is(joined, constructorErr) || !errors.Is(joined, closeErr) || !result.CleanupPending {
		t.Fatalf("primary/cleanup error or ownership lost: %v pending=%t", joined, result.CleanupPending)
	}
	if want := []string{"partial module"}; !reflect.DeepEqual(calls, want) {
		t.Fatalf("failed drain released dependencies: %v, want %v", calls, want)
	}

	if err := result.Close(); err != nil || result.CleanupPending {
		t.Fatalf("cleanup retry = %v pending=%t", err, result.CleanupPending)
	}
	want := []string{"partial module", "partial module", "device", "lease"}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("cleanup retry order = %v, want %v", calls, want)
	}
	if err := result.Close(); err != nil || !reflect.DeepEqual(calls, want) {
		t.Fatalf("completed cleanup repeated: %v calls=%v", err, calls)
	}
}

func TestNativeConstructorDoesNotRegisterZeroOwner(t *testing.T) {
	constructorErr := errors.New("constructor refused")
	cleanup := &nativeCleanup{}
	owner, err := constructNativeOwner(cleanup, "empty", func() (*nativeOwnerFixture, error) {
		return nil, constructorErr
	}, func(owner *nativeOwnerFixture) bool {
		return owner != nil
	}, func(**nativeOwnerFixture) error {
		t.Fatal("zero owner was closed")
		return nil
	})
	if owner != nil || !errors.Is(err, constructorErr) || len(cleanup.actions) != 0 {
		t.Fatalf("zero owner result = %p, %v, actions=%d", owner, err, len(cleanup.actions))
	}
	result := &NativeResult{cleanup: cleanup}
	if got := joinNativeCleanup(result, err); !errors.Is(got, constructorErr) || result.CleanupPending {
		t.Fatalf("zero-owner cleanup changed result: %v pending=%t", got, result.CleanupPending)
	}
}

func TestNativeModuleContextRestoreKeepsCloseState(t *testing.T) {
	module := driver.Module(0x1234)
	restoreErr := errors.New("context restore failed")
	restoreFails := true
	withContextCalls, unloadCalls := 0, 0
	withContext := func(fn func() error) error {
		withContextCalls++
		if err := fn(); err != nil {
			return err
		}
		if restoreFails {
			restoreFails = false
			return restoreErr
		}
		return nil
	}
	unload := func(module *driver.Module) error {
		unloadCalls++
		*module = 0
		return nil
	}
	if err := closeNativeModule(withContext, &module, unload); !errors.Is(err, restoreErr) {
		t.Fatalf("restore error = %v", err)
	}
	if module != 0 || unloadCalls != 1 {
		t.Fatalf("successful unload state: module=%#x unloads=%d", module, unloadCalls)
	}
	if err := closeNativeModule(withContext, &module, unload); err != nil {
		t.Fatal(err)
	}
	if withContextCalls != 1 || unloadCalls != 1 {
		t.Fatalf("restoration retry double-unloaded: context=%d unload=%d", withContextCalls, unloadCalls)
	}
}
