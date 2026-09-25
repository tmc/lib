//go:build darwin || linux

package nccl

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// stubCommInitAll writes one handle per device, derived from the device, so
// the test sees which storage the wrapper forwarded.
const stubCommInitAll = `
#include <stdint.h>
int ncclCommInitAll(uintptr_t *comms, int n, const int *devs) {
	if (n != 2 || devs[0] != 3 || devs[1] != 5) return 4;
	for (int i = 0; i < n; i++) comms[i] = devs[i] * 11;
	return 0;
}
`

// This checks wrapper storage forwarding, not native communicator execution.
func TestCommInitAllArray(t *testing.T) {
	bindStub(t, stubCommInitAll, map[string]*uintptr{"ncclCommInitAll": &symCommInitAll})
	storage := []Comm{99, 0, 0, 99}
	devices := []int32{3, 5}
	if err := CommInitAll(&storage[1], 2, &devices[0]); err != nil {
		t.Fatal(err)
	}
	want := []Comm{99, 33, 55, 99}
	for i := range want {
		if storage[i] != want[i] {
			t.Fatalf("storage = %v, want %v", storage, want)
		}
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
