package sanitizer

import (
	"errors"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestSearchPathsLinuxOrder pins the Linux candidate list: environment roots
// first, in CUDA_HOME, CUDA_PATH, CUDA_ROOT order, then the default prefixes,
// each expanded to <root>/compute-sanitizer/libsanitizer-public.so and each
// appearing once.
//
// It runs the Linux branch on any host, because the platform this package
// targets is not the platform its tests run on.
func TestSearchPathsLinuxOrder(t *testing.T) {
	env := map[string]string{
		"CUDA_ROOT": "/opt/cuda", // also a default prefix: must not appear twice
		"CUDA_HOME": "/custom/cuda",
	}
	got := searchPaths("linux", func(k string) string { return env[k] })

	want := []string{
		"/custom/cuda/compute-sanitizer/libsanitizer-public.so",
		"/opt/cuda/compute-sanitizer/libsanitizer-public.so",
		"/usr/local/cuda/compute-sanitizer/libsanitizer-public.so",
		"/usr/local/cuda-13/compute-sanitizer/libsanitizer-public.so",
		"/usr/local/cuda-12/compute-sanitizer/libsanitizer-public.so",
		"/usr/lib/cuda/compute-sanitizer/libsanitizer-public.so",
	}
	if len(got) != len(want) {
		t.Fatalf("searchPaths returned %d paths, want %d:\n%s", len(got), len(want), strings.Join(got, "\n"))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("path %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestSearchPathsAllAbsolute checks that every candidate is absolute and names
// the library file itself, so a relative CUDA_HOME cannot make the loader
// resolve against the current directory.
func TestSearchPathsAllAbsolute(t *testing.T) {
	got := searchPaths("linux", func(string) string { return "" })
	if len(got) == 0 {
		t.Fatal("searchPaths(linux) returned no candidates")
	}
	for _, p := range got {
		if !filepath.IsAbs(p) {
			t.Errorf("candidate %q is not absolute", p)
		}
		if filepath.Base(p) != toolkitLibName {
			t.Errorf("candidate %q does not end in %s", p, toolkitLibName)
		}
	}
}

// TestSearchPathsUnsupportedOS checks that a platform Compute Sanitizer does
// not ship for yields no candidates rather than a list of paths that cannot
// exist.
func TestSearchPathsUnsupportedOS(t *testing.T) {
	for _, goos := range []string{"darwin", "windows", "plan9"} {
		if got := searchPaths(goos, func(string) string { return "" }); got != nil {
			t.Errorf("searchPaths(%q) = %v, want nil", goos, got)
		}
	}
}

// TestNotFoundErrorListsEveryPath checks that the failure message names every
// candidate tried, which is the whole point of the type: the library lives off
// the loader's search path, so the usual fix is to see which prefixes were
// checked and point SANITIZER_LIB_PATH somewhere else.
func TestNotFoundErrorListsEveryPath(t *testing.T) {
	err := &NotFoundError{Tried: []LoadAttempt{
		{Path: "/a/compute-sanitizer/libsanitizer-public.so", Err: errors.New("no such file")},
		{Path: "/b/compute-sanitizer/libsanitizer-public.so", Err: errors.New("permission denied")},
	}}
	msg := err.Error()
	for _, want := range []string{"/a/compute-sanitizer", "/b/compute-sanitizer", "no such file", "permission denied", "2 path(s)"} {
		if !strings.Contains(msg, want) {
			t.Errorf("NotFoundError.Error() = %q, missing %q", msg, want)
		}
	}
}

// TestLoadFromToolkitFailsClosed checks that LoadFromToolkit reports a typed,
// enumerated failure instead of panicking when the library is absent, and that
// it does not leave the package believing something is loaded.
//
// On a machine that does have a toolkit install this test observes the success
// path instead; it never asserts that loading must fail.
func TestLoadFromToolkitFailsClosed(t *testing.T) {
	err := LoadFromToolkit()
	if err == nil {
		if !Loaded() {
			t.Error("LoadFromToolkit returned nil but Loaded() is false")
		}
		t.Skipf("library loaded from %s: the failure path is not observable here", Path())
	}
	var nf *NotFoundError
	if !errors.As(err, &nf) {
		t.Fatalf("LoadFromToolkit error = %v (%T), want *NotFoundError", err, err)
	}
	if len(nf.Tried) == 0 {
		t.Error("NotFoundError.Tried is empty; the error must name what it tried")
	}
	if runtime.GOOS == "linux" && len(nf.Tried) != len(SearchPaths()) {
		t.Errorf("tried %d paths, SearchPaths reports %d", len(nf.Tried), len(SearchPaths()))
	}
	if Loaded() {
		t.Error("Loaded() is true after LoadFromToolkit failed")
	}
}
