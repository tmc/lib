//go:build darwin || linux

package cudnn

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// stubDirect stands in for the cuDNN functions that return a value directly.
const stubDirect = `
#include <stddef.h>
#include <string.h>
#define WANT ((size_t)-1 - 17)
size_t cudnnGetVersion(void) { return WANT; }
size_t cudnnGetMaxDeviceVersion(void) { return WANT - 1; }
size_t cudnnGetCudartVersion(void) { return WANT - 2; }
void cudnnGetLastErrorString(char *p, size_t n) { if (n == 4) memcpy(p, "ok\0\0", 4); }
`

func TestDirectReturns(t *testing.T) {
	// Restore in a cleanup, not a defer: bindStub's cleanup restores the
	// stub state below, and cleanups run after defers, so a deferred
	// restore would be undone.
	oldReady, oldPresent := ready.Load(), present
	t.Cleanup(func() {
		ready.Store(oldReady)
		present = oldPresent
	})
	ready.Store(true)
	present = map[string]bool{}
	for _, tt := range []struct {
		name string
		call func() error
	}{
		{"cudnnGetVersion", func() error { _, e := GetVersion(); return e }},
		{"cudnnGetMaxDeviceVersion", func() error { _, e := GetMaxDeviceVersion(); return e }},
		{"cudnnGetCudartVersion", func() error { _, e := GetCudartVersion(); return e }},
		{"cudnnGetLastErrorString", func() error { return GetLastErrorString(nil, 0) }},
	} {
		err := tt.call()
		var absent *SymbolError
		if !errors.As(err, &absent) || absent.Symbol != tt.name {
			t.Errorf("%s: %v, want named SymbolError", tt.name, err)
		}
	}

	bindStub(t, stubDirect, map[string]*uintptr{
		"cudnnGetVersion":          &symGetVersion,
		"cudnnGetMaxDeviceVersion": &symGetMaxDeviceVersion,
		"cudnnGetCudartVersion":    &symGetCudartVersion,
		"cudnnGetLastErrorString":  &symGetLastErrorString,
	})
	// Preserve the full size_t width rather than converting it to a status or int32.
	want := ^uint(0) - 17
	for i, f := range []func() (uint, error){GetVersion, GetMaxDeviceVersion, GetCudartVersion} {
		if v, e := f(); e != nil || v != want-uint(i) {
			t.Errorf("getter %d = %d,%v", i, v, e)
		}
	}
	var buf [6]byte
	buf[0], buf[5] = 0xa5, 0x5a
	if err := GetLastErrorString(&buf[1], 4); err != nil {
		t.Fatal(err)
	}
	if buf != [6]byte{0xa5, 'o', 'k', 0, 0, 0x5a} {
		t.Fatalf("buffer bytes=%v", buf)
	}
}

// bindStub compiles csrc with the host C compiler and, until the test ends,
// binds each symbol in syms to it and marks it present. It skips the test
// when there is no compiler.
func bindStub(t *testing.T, csrc string, syms map[string]*uintptr) {
	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		t.Skip("no C compiler")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "stub.c")
	if err := os.WriteFile(src, []byte(csrc), 0o644); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "libstub.so")
	args := []string{"-shared", "-fPIC", "-o", path, src}
	if runtime.GOOS == "darwin" {
		arch := runtime.GOARCH
		if arch == "amd64" {
			arch = "x86_64"
		}
		args = append([]string{"-arch", arch}, args...)
	}
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("cc: %v\n%s", err, out)
	}
	h, err := openDynamicLibrary(path)
	if err != nil {
		t.Fatal(err)
	}
	oldReady, oldPresent := ready.Load(), present
	old := make(map[string]uintptr)
	present = make(map[string]bool)
	for name, p := range syms {
		old[name] = *p
		if *p, err = lookupLibrarySymbol(h, name); err != nil {
			t.Fatal(err)
		}
		present[name] = true
	}
	ready.Store(true)
	t.Cleanup(func() {
		for name, p := range syms {
			*p = old[name]
		}
		present = oldPresent
		ready.Store(oldReady)
		closeLibrary(h)
	})
}
