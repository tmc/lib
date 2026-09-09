//go:build darwin || linux

package driver

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

func TestBindFailuresFromNativeSymbols(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("C compiler unavailable")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "fixture.c")
	library := filepath.Join(dir, "fixture.so")
	if err := os.WriteFile(source, []byte("void alpha(void) {}\nvoid zeta(void) {}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(cc, "-shared", "-fPIC", source, "-o", library).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v: %s", err, out)
	}
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	if got := BindFailures(); got != nil {
		t.Fatalf("before load: %v", got)
	}
	if Loaded() {
		t.Fatal("diagnostics triggered a load")
	}
	if err := Load(library); err != nil {
		t.Fatal(err)
	}
	defer Close()
	args := make([]reflect.Type, 40)
	for i := range args {
		args[i] = reflect.TypeOf(uintptr(0))
	}
	fn := reflect.New(reflect.FuncOf(args, nil, false)).Interface()
	loadMu.Lock()
	register(fn, "zeta")
	register(fn, "alpha")
	register(fn, "missing")
	loadMu.Unlock()
	got := BindFailures()
	if len(got) != 2 || got[0].Symbol != "alpha" || got[1].Symbol != "zeta" {
		t.Fatalf("failures: %v", got)
	}
	for _, f := range got {
		if !errors.Is(f.Err, errArgumentSlots) {
			t.Fatalf("cause: %v", f)
		}
		if !errors.Is(checkLibSym(f.Symbol), f.Err) {
			t.Fatalf("call lost binding cause: %s", f.Symbol)
		}
	}
	if errors.Unwrap(checkLibSym("missing")) != nil {
		t.Fatal("absent symbol reported a bind failure")
	}
	got[0].Symbol = "changed"
	if BindFailures()[0].Symbol != "alpha" {
		t.Fatal("returned slice aliases loader state")
	}
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	if got := BindFailures(); got != nil {
		t.Fatalf("close retained failures: %v", got)
	}
}
