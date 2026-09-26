//go:build linux

package cufile

import (
	"errors"
	"strings"
	"testing"
)

// directReturns calls each of the five cuFile wrappers with a non-status
// return type and returns its error.
var directReturns = []struct {
	name   string
	symbol string
	call   func() error
}{
	{"Handle.Read", "cuFileRead", func() error { _, err := Handle(0).Read(0, 0, 0, 0); return err }},
	{"Handle.Write", "cuFileWrite", func() error { _, err := Handle(0).Write(0, 0, 0, 0); return err }},
	{"Handle.Deregister", "cuFileHandleDeregister", func() error { return Handle(0).Deregister() }},
	{"BatchHandle.IODestroy", "cuFileBatchIODestroy", func() error { return BatchHandle(0).IODestroy() }},
	{"UseCount", "cuFileUseCount", func() error { _, err := UseCount(); return err }},
}

// TestDirectReturnWrappersReportUnavailable checks that the five cuFile
// wrappers with a non-status return type report an unavailable library as an
// error instead of calling a nil function value.
func TestDirectReturnWrappersReportUnavailable(t *testing.T) {
	if err := checkLib(); err == nil {
		t.Skip("libcufile is present on this host")
	}
	for _, tt := range directReturns {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			if err == nil {
				t.Fatal("returned nil error with no library")
			}
			t.Logf("%s returned: %v", tt.name, err)
			if !strings.Contains(err.Error(), "cufile") {
				t.Errorf("error %q does not name the library", err)
			}
		})
	}
}

// TestDirectReturnWrappersNameMissingSymbol checks that when the library
// loads but does not export the symbol, the error identifies the symbol by
// name. It requires CUFILE_LIB_PATH to point at a shared object exporting no
// cuFile symbols; see stub.c in the test harness. A genuine libcufile exports
// them, so each case is skipped unless its symbol is actually absent.
func TestDirectReturnWrappersNameMissingSymbol(t *testing.T) {
	if err := checkLib(); err != nil {
		t.Skipf("no loadable library: %v", err)
	}
	for _, tt := range directReturns {
		t.Run(tt.name, func(t *testing.T) {
			if Has(tt.symbol) {
				t.Skipf("the loaded library exports %s", tt.symbol)
			}
			err := tt.call()
			t.Logf("%s returned: %v", tt.name, err)
			var se *SymbolError
			if !errors.As(err, &se) {
				t.Fatalf("error %v (%T), want *SymbolError", err, err)
			}
			if se.Symbol != tt.symbol {
				t.Errorf("SymbolError.Symbol = %q, want %q", se.Symbol, tt.symbol)
			}
		})
	}
}
