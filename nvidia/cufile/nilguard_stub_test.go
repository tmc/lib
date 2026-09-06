//go:build linux

package cufile

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// stubEnv names the environment variable that marks the re-executed child
// process. The parent sets it along with CUFILE_LIB_PATH.
const stubEnv = "CUFILE_NILGUARD_STUB"

// TestMissingSymbolWithStubLibrary drives TestDirectReturnWrappersNameMissingSymbol
// against a library that loads but exports no cuFile symbols, which is the only
// configuration in which that test does more than skip. On a host with a real
// libcufile every case skips, and on a host with none the library never loads;
// neither reaches the guard being tested.
//
// The stub is built from testdata/stub.c and handed to a re-executed copy of
// this test binary through CUFILE_LIB_PATH, because the library path is read
// once per process during the first load.
func TestMissingSymbolWithStubLibrary(t *testing.T) {
	if os.Getenv(stubEnv) != "" {
		t.Skip("child process; the re-executed run does the work")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skipf("no C compiler to build the stub: %v", err)
	}
	stub := filepath.Join(t.TempDir(), "libcufile_stub.so")
	build := exec.Command(cc, "-shared", "-fPIC", "-o", stub, "testdata/stub.c")
	if out, err := build.CombinedOutput(); err != nil {
		t.Skipf("cannot build the stub library: %v\n%s", err, out)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("locate test binary: %v", err)
	}
	child := exec.Command(self, "-test.run", "TestDirectReturnWrappersNameMissingSymbol", "-test.v")
	child.Env = append(os.Environ(), stubEnv+"=1", "CUFILE_LIB_PATH="+stub)
	out, err := child.CombinedOutput()
	t.Logf("child output:\n%s", out)
	if err != nil {
		t.Fatalf("re-executed test failed: %v", err)
	}
	// A run in which every case skipped proves nothing, so require that the
	// child actually reached the guard for all five wrappers.
	if n := strings.Count(string(out), "unavailable"); n != 5 {
		t.Errorf("child reported %d unavailable symbols, want 5", n)
	}
	if strings.Contains(string(out), "SKIP") {
		t.Error("child skipped a case; the stub library exported a cuFile symbol")
	}
}
