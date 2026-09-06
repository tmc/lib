package cudnn

import (
	"errors"
	"os"
	"testing"
)

// TestNativeBindingRecovery qualifies loading real signatures and rejecting an
// intentionally oversized synthetic signature. It never calls that signature.
func TestNativeBindingRecovery(t *testing.T) {
	path := os.Getenv("CUDNN_BIND_LIBRARY")
	if path == "" {
		t.Skip("CUDNN_BIND_LIBRARY is required")
	}
	if Loaded() {
		t.Fatal("library loaded before explicit Load")
	}
	if got := BindFailures(); len(got) != 0 {
		t.Fatalf("before Load: %v", got)
	}
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Close(); err != nil {
			t.Errorf("close library: %v", err)
		}
	}()
	failures := BindFailures()
	if len(failures) != 0 {
		t.Fatalf("unexpected native failures: %v", failures)
	}
	if !HasnormalizationBackward() || fnnormalizationBackward == nil {
		t.Fatal("real normalization signature did not bind")
	}
	var oversized func(uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr, uintptr)
	// Use a resolved vendor symbol with a synthetic Go signature to exercise
	// RegisterFunc failure. The actual generated signature above is healthy;
	// parameter count alone does not determine ABI argument-slot consumption.
	const symbol = "cudnnNormalizationBackward"
	savedPresent, hadPresent := present[symbol]
	savedUnbound, hadUnbound := unbound[symbol]
	defer func() {
		if hadPresent {
			present[symbol] = savedPresent
		} else {
			delete(present, symbol)
		}
		if hadUnbound {
			unbound[symbol] = savedUnbound
		} else {
			delete(unbound, symbol)
		}
	}()
	delete(present, symbol)
	delete(unbound, symbol)
	register(&oversized, symbol)
	failures = BindFailures()
	if len(failures) != 1 {
		t.Fatalf("oversized signature failures: %v", failures)
	}
	if oversized != nil {
		t.Fatal("failed binding published function")
	}
	if HasnormalizationBackward() {
		t.Fatal("failed binding reported present")
	}
	var symbolErr *SymbolError
	if err := checkLibSym("cudnnNormalizationBackward"); !errors.As(err, &symbolErr) || !errors.Is(err, errArgumentSlots) {
		t.Fatalf("failure cause not propagated: %v", err)
	}
	f := failures[0]
	if f.Symbol != "cudnnNormalizationBackward" || f.Err != errArgumentSlots {
		t.Fatalf("unexpected failure: %+v", f)
	}

	version, err := GetVersion()
	if err != nil || version == 0 {
		t.Fatalf("healthy GetVersion: %d, %v", version, err)
	}
	t.Logf("loaded=%s version=%d rejected=%s cause=%v", path, version, f.Symbol, f.Err)
}
