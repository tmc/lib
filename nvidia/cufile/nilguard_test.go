//go:build linux

package cufile

import (
	"errors"
	"strings"
	"testing"
)

// mustPanic runs f and returns the value it panicked with. It fails the test
// if f returns normally.
func mustPanic(t *testing.T, name string, f func()) (v any) {
	t.Helper()
	defer func() {
		v = recover()
		if v == nil {
			t.Fatalf("%s returned normally; want panic", name)
		}
	}()
	f()
	return nil
}

// TestDirectReturnWrappersPanicDiagnostically checks that the five cuFile
// wrappers with a non-status return type report an unavailable library
// instead of calling a nil function value. Each is the first cuFile call in
// its own subtest process view: the guard runs before the function pointer is
// dereferenced, so a missing library surfaces as a descriptive error rather
// than a segmentation fault.
func TestDirectReturnWrappersPanicDiagnostically(t *testing.T) {
	if err := checkLib(); err == nil {
		t.Skip("libcufile is present on this host")
	}
	for _, tt := range []struct {
		name string
		call func()
	}{
		{"Handle.Read", func() { Handle(0).Read(0, 0, 0, 0) }},
		{"Handle.Write", func() { Handle(0).Write(0, 0, 0, 0) }},
		{"Handle.Deregister", func() { Handle(0).Deregister() }},
		{"BatchHandle.IODestroy", func() { BatchHandle(0).IODestroy() }},
		{"UseCount", func() { UseCount() }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v := mustPanic(t, tt.name, tt.call)
			err, ok := v.(error)
			if !ok {
				t.Fatalf("panicked with %v (%T), want error", v, v)
			}
			t.Logf("%s panicked with: %v", tt.name, err)
			if !strings.Contains(err.Error(), "cufile") {
				t.Errorf("panic %q does not name the library", err)
			}
		})
	}
}

// TestDirectReturnWrappersNameMissingSymbol checks that when the library
// loads but does not export the symbol, the panic identifies the symbol by
// name. It requires CUFILE_LIB_PATH to point at a shared object exporting no
// cuFile symbols; see stub.c in the test harness.
func TestDirectReturnWrappersNameMissingSymbol(t *testing.T) {
	if err := checkLib(); err != nil {
		t.Skipf("no loadable library: %v", err)
	}
	for _, tt := range []struct {
		name   string
		symbol string
		call   func()
	}{
		{"Handle.Read", "cuFileRead", func() { Handle(0).Read(0, 0, 0, 0) }},
		{"Handle.Write", "cuFileWrite", func() { Handle(0).Write(0, 0, 0, 0) }},
		{"Handle.Deregister", "cuFileHandleDeregister", func() { Handle(0).Deregister() }},
		{"BatchHandle.IODestroy", "cuFileBatchIODestroy", func() { BatchHandle(0).IODestroy() }},
		{"UseCount", "cuFileUseCount", func() { UseCount() }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v := mustPanic(t, tt.name, tt.call)
			err, ok := v.(error)
			if !ok {
				t.Fatalf("panicked with %v (%T), want error", v, v)
			}
			t.Logf("%s panicked with: %v", tt.name, err)
			var se *SymbolError
			if !errors.As(err, &se) {
				t.Fatalf("panic %v (%T), want *SymbolError", err, err)
			}
			if se.Symbol != tt.symbol {
				t.Errorf("SymbolError.Symbol = %q, want %q", se.Symbol, tt.symbol)
			}
		})
	}
}
