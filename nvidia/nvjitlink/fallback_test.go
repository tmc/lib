package nvjitlink_test

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/nvjitlink"
)

// stubEnv marks the re-executed child process, which does the actual work.
// The library path is read once per process during the first load, so the
// stub has to be in place before the package under test is touched.
const stubEnv = "NVJITLINK_FALLBACK_STUB"

// TestNoFallbackFailsCleanly pins what happens against a toolkit that offers
// the loader nothing to fall back to.
//
// Every nvJitLink entry point is an inline function that forwards to a symbol
// carrying the toolkit version, so this package binds __nvJitLinkCreate_13_4.
// Libraries from CUDA 12.1 onward also export an unsuffixed nvJitLinkCreate,
// which the loader binds instead once the library's version checks out. CUDA
// 12.0 and 12.2 export neither that alias nor the _13_4 symbol, so by
// construction the fallback cannot rescue them and the call must fail.
//
// What is worth pinning is that it fails well: a *SymbolError naming the
// symbol that is missing, rather than a nil function dereference or a message
// that sends the reader looking for a load failure. That is the whole promise
// of resolving symbols lazily instead of at load time.
//
// The export surface here matches the real 12.0 redistributable, whose only
// spelling of the entry point is __nvJitLinkCreate_12_0.
func TestNoFallbackFailsCleanly(t *testing.T) {
	if os.Getenv(stubEnv) == "" {
		reexecWithStub(t, "testdata/stub120.c")
		return
	}
	if _, err := nvjitlink.Create(); err == nil {
		t.Fatal("Create succeeded against a library exporting no usable symbol")
	} else if se := new(nvjitlink.SymbolError); !errors.As(err, &se) {
		t.Fatalf("Create returned %v (%T), want *SymbolError", err, err)
	} else if se.Symbol != "__nvJitLinkCreate_13_4" {
		t.Errorf("SymbolError.Symbol = %q, want %q", se.Symbol, "__nvJitLinkCreate_13_4")
	} else {
		t.Logf("Create failed cleanly: %v", err)
	}
	// The unsuffixed alias is the only thing that could have rescued this,
	// and a 12.0 library does not have it. Record that the loader agrees.
	if nvjitlink.HasCreate() {
		t.Error("HasCreate reports a symbol the stub does not export")
	}
}

// TestFallbackAfterVersionCheck pins that a library whose version checks out
// binds the unversioned alias of an entry point it does not export under the
// versioned name: a 13.x library older than the header.
func TestFallbackAfterVersionCheck(t *testing.T) {
	if os.Getenv(stubEnv) == "" {
		reexecWithStub(t, "testdata/stubalias.c", "-DNVJITLINK_STUB_MAJOR=13")
		return
	}
	h, err := nvjitlink.Create()
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if h == 0 {
		t.Error("Create returned a zero handle")
	}
	if !nvjitlink.HasCreate() {
		t.Error("HasCreate = false after Create bound through nvJitLinkCreate")
	}
	if err := h.Destroy(); err != nil {
		t.Errorf("Destroy: %v", err)
	}
}

// TestNoFallbackWithoutVersionCheck pins that the unversioned alias is not
// bound in a library whose version could not be checked. The alias names no
// train, so binding it there could call into a library these bindings were
// not generated for.
func TestNoFallbackWithoutVersionCheck(t *testing.T) {
	if os.Getenv(stubEnv) == "" {
		reexecWithStub(t, "testdata/stubalias.c")
		return
	}
	_, err := nvjitlink.Create()
	var se *nvjitlink.SymbolError
	if !errors.As(err, &se) {
		t.Fatalf("Create returned %v (%T), want *SymbolError", err, err)
	}
	if se.Symbol != "__nvJitLinkCreate_13_4" {
		t.Errorf("SymbolError.Symbol = %q, want %q", se.Symbol, "__nvJitLinkCreate_13_4")
	}
	if nvjitlink.HasCreate() {
		t.Error("HasCreate = true in a library whose version was not checked")
	}
	if !nvjitlink.Loaded() {
		t.Error("the library was not admitted")
	}
}

// TestVersionMismatchRejected pins that a library reporting another major is
// refused outright, fallback or not.
func TestVersionMismatchRejected(t *testing.T) {
	if os.Getenv(stubEnv) == "" {
		reexecWithStub(t, "testdata/stubalias.c", "-DNVJITLINK_STUB_MAJOR=11", "-DNVJITLINK_STUB_MINOR=2")
		return
	}
	_, err := nvjitlink.Create()
	var vm *nvjitlink.VersionMismatchError
	if !errors.As(err, &vm) {
		t.Fatalf("Create returned %v (%T), want *VersionMismatchError", err, err)
	}
	if vm.Got != 11 || vm.Want != 13 {
		t.Errorf("VersionMismatchError Got, Want = %d, %d; want 11, 13", vm.Got, vm.Want)
	}
	if nvjitlink.Loaded() {
		t.Error("a library of the wrong major stayed loaded")
	}
}

// reexecWithStub builds a stub library from the C source src, compiled with
// the extra cc arguments args, and runs the calling test again against it.
func reexecWithStub(t *testing.T, src string, args ...string) {
	t.Helper()
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skipf("no C compiler to build the stub: %v", err)
	}
	stub := filepath.Join(t.TempDir(), "libnvJitLink.so")
	build := exec.Command(cc, append(append([]string{"-shared", "-fPIC", "-o", stub}, args...), src)...)
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build the stub library: %v\n%s", err, out)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("locate test binary: %v", err)
	}
	child := exec.Command(self, "-test.run", "^"+t.Name()+"$", "-test.v")
	child.Env = append(os.Environ(), stubEnv+"=1", "NVJITLINK_LIB_PATH="+stub)
	out, err := child.CombinedOutput()
	t.Logf("child output:\n%s", out)
	if err != nil {
		t.Fatalf("re-executed test failed: %v", err)
	}
}
