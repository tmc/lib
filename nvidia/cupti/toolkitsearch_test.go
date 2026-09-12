package cupti

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// firstLibraryName returns the candidate the toolkit search tries first, and
// skips the test where there is none. toolkitLibraryPaths only ever tries the
// names libraryNames returns, so a test that stubs a fixed soname stops
// exercising the search the moment the candidate list changes.
func firstLibraryName(t *testing.T) string {
	t.Helper()
	names := libraryNames()
	if len(names) == 0 {
		t.Skipf("libraryNames has no entry for %s", runtime.GOOS)
	}
	return names[0]
}

// stubLibrary compiles a shared library into dir and returns its path. The
// directory is a fresh temporary one, so it is not on the dynamic linker's
// search path and can only be found by full path.
func stubLibrary(t *testing.T, dir, name string) string {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skipf("libraryNames has no entry for %s", runtime.GOOS)
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skipf("no C compiler: %v", err)
	}
	src := filepath.Join(t.TempDir(), "stub.c")
	if err := os.WriteFile(src, []byte("int cuptiGetVersion(void) { return 0; }\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, name)
	cmd := exec.Command(cc, "-shared", "-fPIC", "-o", out, src)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("cc -shared: %v: %s", err, b)
	}
	return out
}

// TestToolkitLibraryPathsFindsLibraryOffTheLinkerPath is the regression test
// for the load failure on a correctly installed toolkit: the CUDA packages put
// libcupti in extras/CUPTI/lib64, which ld.so does not search, so the bare
// sonames libraryNames returns cannot resolve unless something has already
// added that directory to LD_LIBRARY_PATH.
//
// The stub is built into a fresh temporary directory, reachable only by full
// path, and the test removes it again as its own negative control: the
// assertion is only meaningful if it can fail.
func TestToolkitLibraryPathsFindsLibraryOffTheLinkerPath(t *testing.T) {
	root := t.TempDir()
	libDir := filepath.Join(root, "extras", "CUPTI", "lib64")
	stub := stubLibrary(t, libDir, firstLibraryName(t))

	t.Setenv("CUDA_HOME", root)
	t.Setenv("CUDA_PATH", "")

	paths := pathsUnder(toolkitLibraryPaths(), root)
	if len(paths) == 0 {
		t.Fatalf("toolkitLibraryPaths found nothing under CUDA_HOME=%s", root)
	}
	if paths[0] != stub {
		t.Errorf("toolkitLibraryPaths[0] = %q, want %q", paths[0], stub)
	}
	// The path must be openable, not merely present.
	h, err := openDynamicLibrary(stub)
	if err != nil || h == 0 {
		t.Fatalf("openDynamicLibrary(%q) = %d, %v", stub, h, err)
	}

	// Negative control: with the file gone the same call must find nothing.
	if err := os.Remove(stub); err != nil {
		t.Fatal(err)
	}
	if paths := pathsUnder(toolkitLibraryPaths(), root); len(paths) != 0 {
		t.Errorf("toolkitLibraryPaths returned %v after the library was removed", paths)
	}
}

// TestToolkitSearchHintNamesTheDirectory covers the diagnostic floor. Without
// it the failure is an opaque "cannot open libcupti", which reads as an absent
// toolkit rather than a toolkit installed exactly where NVIDIA puts it.
func TestToolkitSearchHintNamesTheDirectory(t *testing.T) {
	t.Setenv("CUDA_HOME", "/opt/cuda-test")
	t.Setenv("CUDA_PATH", "")

	hint := toolkitSearchHint()
	for _, want := range []string{"/opt/cuda-test/extras/CUPTI/lib64", "CUPTI_LIB_PATH", "LD_LIBRARY_PATH"} {
		if !strings.Contains(hint, want) {
			t.Errorf("hint does not mention %q: %s", want, hint)
		}
	}
	// The conventional root is searched even with no environment set, so the
	// hint is never empty and the message never degrades to silence.
	t.Setenv("CUDA_HOME", "")
	if hint := toolkitSearchHint(); !strings.Contains(hint, "/usr/local/cuda/extras/CUPTI/lib64") {
		t.Errorf("hint with no CUDA_HOME does not name the conventional root: %s", hint)
	}
}

// TestToolkitSearchDoesNotPreemptTheLinker checks the ordering constraint: a
// library the dynamic linker already resolves must still load by soname, so
// this change cannot redirect a working installation to a different copy.
func TestToolkitSearchDoesNotPreemptTheLinker(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("libraryNames has no entry for %s", runtime.GOOS)
	}
	var resolvable string
	for _, name := range libraryNames() {
		if h, err := openDynamicLibrary(name); err == nil && h != 0 {
			resolvable = name
			break
		}
	}
	if resolvable == "" {
		t.Skip("no libcupti on the linker path; nothing to preempt")
	}

	// Point the toolkit search at a directory holding a different library.
	root := t.TempDir()
	stubLibrary(t, filepath.Join(root, "extras", "CUPTI", "lib64"), firstLibraryName(t))
	t.Setenv("CUDA_HOME", root)
	t.Setenv("CUDA_PATH", "")

	_, path, err := openLibrary("")
	if err != nil {
		t.Fatalf("openLibrary: %v", err)
	}
	if path != resolvable {
		t.Errorf("openLibrary used %q, want the soname %q the linker resolves", path, resolvable)
	}
}

// pathsUnder returns the candidates found beneath root, so an assertion is not
// perturbed by a real CUDA toolkit installed on the test machine: toolkitRoots
// always includes the conventional /usr/local/cuda.
func pathsUnder(paths []string, root string) []string {
	var found []string
	for _, p := range paths {
		if strings.HasPrefix(p, root+string(filepath.Separator)) {
			found = append(found, p)
		}
	}
	return found
}
