package nvvm

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestToolkitLibraryPathsFindsLibraryOffTheLinkerPath is the nvvm half of the
// same regression: the CUDA packages install libnvvm in nvvm/lib64, which the
// dynamic linker does not search, so the bare sonames libraryNames returns
// cannot resolve on an otherwise correct installation.
//
// The stub is built into a fresh temporary directory, reachable only by full
// path, and removing it again is the test's own negative control.
func TestToolkitLibraryPathsFindsLibraryOffTheLinkerPath(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("libraryNames has no entry for %s", runtime.GOOS)
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skipf("no C compiler: %v", err)
	}
	src := filepath.Join(t.TempDir(), "stub.c")
	if err := os.WriteFile(src, []byte("int nvvmVersion(void) { return 0; }\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	libDir := filepath.Join(root, "nvvm", "lib64")
	if err := os.MkdirAll(libDir, 0o700); err != nil {
		t.Fatal(err)
	}
	stub := filepath.Join(libDir, "libnvvm.so")
	if b, err := exec.Command(cc, "-shared", "-fPIC", "-o", stub, src).CombinedOutput(); err != nil {
		t.Skipf("cc -shared: %v: %s", err, b)
	}

	t.Setenv("CUDA_HOME", root)
	t.Setenv("CUDA_PATH", "")

	paths := pathsUnder(toolkitLibraryPaths(), root)
	if len(paths) == 0 || paths[0] != stub {
		t.Fatalf("toolkitLibraryPaths = %v, want %q first", paths, stub)
	}
	if h, err := openDynamicLibrary(stub); err != nil || h == 0 {
		t.Fatalf("openDynamicLibrary(%q) = %d, %v", stub, h, err)
	}
	if hint := toolkitSearchHint(); !strings.Contains(hint, libDir) {
		t.Errorf("hint does not name %s: %s", libDir, hint)
	}

	if err := os.Remove(stub); err != nil {
		t.Fatal(err)
	}
	if paths := pathsUnder(toolkitLibraryPaths(), root); len(paths) != 0 {
		t.Errorf("toolkitLibraryPaths returned %v after the library was removed", paths)
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
