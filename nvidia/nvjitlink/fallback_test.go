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
// Every nvJitLink entry point is a macro that expands to a symbol carrying the
// toolkit version, so this package is generated against __nvJitLinkCreate_13_0.
// Libraries from CUDA 12.1 onward also export an unsuffixed nvJitLinkCreate,
// which register lists as a second spelling and binds instead. CUDA 12.0 and
// 12.2 export neither that alias nor the _13_0 symbol, so by construction the
// fallback cannot rescue them and the call must fail.
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
		reexecWithStub(t)
		return
	}
	if _, err := nvjitlink.Create(nil); err == nil {
		t.Fatal("Create succeeded against a library exporting no usable symbol")
	} else if se := new(nvjitlink.SymbolError); !errors.As(err, &se) {
		t.Fatalf("Create returned %v (%T), want *SymbolError", err, err)
	} else if se.Symbol != "__nvJitLinkCreate_13_2" {
		t.Errorf("SymbolError.Symbol = %q, want %q", se.Symbol, "__nvJitLinkCreate_13_2")
	} else {
		t.Logf("Create failed cleanly: %v", err)
	}
	// The unsuffixed alias is the only thing that could have rescued this,
	// and a 12.0 library does not have it. Record that the loader agrees.
	if nvjitlink.HasCreate() {
		t.Error("HasCreate reports a symbol the stub does not export")
	}
}

// reexecWithStub builds the CUDA 12.0 export surface from testdata and runs
// this test again against it.
func reexecWithStub(t *testing.T) {
	t.Helper()
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skipf("no C compiler to build the stub: %v", err)
	}
	stub := filepath.Join(t.TempDir(), "libnvJitLink.so.12")
	build := exec.Command(cc, "-shared", "-fPIC", "-o", stub, "testdata/stub120.c")
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build the stub library: %v\n%s", err, out)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("locate test binary: %v", err)
	}
	child := exec.Command(self, "-test.run", "TestNoFallbackFailsCleanly", "-test.v")
	child.Env = append(os.Environ(), stubEnv+"=1", "NVJITLINK_LIB_PATH="+stub)
	out, err := child.CombinedOutput()
	t.Logf("child output:\n%s", out)
	if err != nil {
		t.Fatalf("re-executed test failed: %v", err)
	}
}
