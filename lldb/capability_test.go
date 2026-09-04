package lldb

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func TestRequireReportsLoadError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing-lldb-bridge")
	if err := Load(path); err == nil {
		t.Fatal("Load succeeded for missing bridge")
	}

	err := Require("LLDBLastError")
	if err == nil {
		t.Fatal("Require succeeded after bridge load failure")
	}
	if !strings.Contains(err.Error(), path) {
		t.Fatalf("Require error %q does not name bridge %q", err, path)
	}
	var symbolErr *SymbolError
	if errors.As(err, &symbolErr) {
		t.Fatalf("Require error = %T, want bridge load error", err)
	}
	if Loaded() {
		t.Fatal("Loaded reports true after bridge load failure")
	}
	if got := Path(); got != "" {
		t.Fatalf("Path = %q after bridge load failure, want empty", got)
	}
}

func TestRequireReportsMissingSymbol(t *testing.T) {
	loadMu.Lock()
	oldReady := libReady.Load()
	oldPresent := present
	present = make(map[string]bool)
	libReady.Store(true)
	loadMu.Unlock()
	t.Cleanup(func() {
		loadMu.Lock()
		libReady.Store(false)
		present = oldPresent
		libReady.Store(oldReady)
		loadMu.Unlock()
	})

	err := Require("LLDBFutureSymbol")
	var symbolErr *SymbolError
	if !errors.As(err, &symbolErr) {
		t.Fatalf("Require error = %T, want *SymbolError", err)
	}
	if got, want := err.Error(), "lldb: symbol LLDBFutureSymbol unavailable"; got != want {
		t.Fatalf("Error = %q, want %q", got, want)
	}
}
