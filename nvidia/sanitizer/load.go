package sanitizer

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// toolkitLibName is the file name of the Compute Sanitizer public library
// inside a CUDA toolkit install. The shipped ELF's DT_SONAME is exactly this,
// with no version suffix: unlike libcudart.so.13 there is no versioned
// alternative to fall back to.
const toolkitLibName = "libsanitizer-public.so"

// toolkitSubdir is the directory the library lives in inside a toolkit
// install, relative to the toolkit root. It is not lib64, and it is not on
// ld.so's default search path.
const toolkitSubdir = "compute-sanitizer"

// toolkitEnvVars are the environment variables that conventionally name a CUDA
// toolkit root.
var toolkitEnvVars = []string{"CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"}

// toolkitRoots are the install prefixes tried when no environment variable
// names one.
var toolkitRoots = []string{
	"/usr/local/cuda",
	"/usr/local/cuda-13",
	"/usr/local/cuda-12",
	"/opt/cuda",
	"/usr/lib/cuda",
}

// NotFoundError reports that the library could not be opened, listing every
// candidate that was tried and why each failed.
//
// Reporting the whole list matters here more than for the other NVIDIA
// libraries: this one is not on the loader's search path, so the usual failure
// is a correct install in a prefix the list does not cover, and the fix is to
// set SANITIZER_LIB_PATH to a path the message already shows the shape of.
type NotFoundError struct {
	// Tried holds one entry per candidate, in the order attempted.
	Tried []LoadAttempt
}

// LoadAttempt records one candidate path and the error opening it produced.
type LoadAttempt struct {
	Path string
	Err  error
}

func (e *NotFoundError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "sanitizer: %s not found; tried %d path(s)", toolkitLibName, len(e.Tried))
	for _, a := range e.Tried {
		fmt.Fprintf(&b, "\n\t%s: %v", a.Path, a.Err)
	}
	return b.String()
}

// SearchPaths returns the absolute paths [LoadFromToolkit] tries, in order.
//
// A CUDA toolkit keeps the Compute Sanitizer library in
// <root>/compute-sanitizer/, beside the compute-sanitizer launcher, rather
// than in <root>/lib64. Nothing puts that directory on the loader's default
// search path, so the bare soname that [Load] tries first usually fails on an
// otherwise complete install. The roots come from CUDA_HOME, CUDA_PATH and
// CUDA_ROOT when set, then from the usual install prefixes.
//
// The list is empty on GOOS values Compute Sanitizer does not ship for.
// Duplicates are removed, so an environment variable naming a default prefix
// does not produce the same path twice.
func SearchPaths() []string { return searchPaths(runtime.GOOS, os.Getenv) }

// searchPaths is SearchPaths with the platform and environment injected, so
// the Linux ordering can be tested on a machine that is not Linux. This
// package's tests cannot run on the platform the bindings target, which makes
// the seam worth having.
func searchPaths(goos string, getenv func(string) string) []string {
	if goos != "linux" {
		return nil
	}
	var roots []string
	for _, env := range toolkitEnvVars {
		if v := getenv(env); v != "" {
			roots = append(roots, v)
		}
	}
	roots = append(roots, toolkitRoots...)

	seen := make(map[string]bool, len(roots))
	paths := make([]string, 0, len(roots))
	for _, root := range roots {
		p := filepath.Join(root, toolkitSubdir, toolkitLibName)
		if seen[p] {
			continue
		}
		seen[p] = true
		paths = append(paths, p)
	}
	return paths
}

// LoadFromToolkit opens the library from a CUDA toolkit install, trying every
// path [SearchPaths] reports until one succeeds.
//
// Use it when the library is installed but not reachable by name, which is the
// normal case: see [SearchPaths]. It is an alternative to [Load], not a
// replacement: [Load] with an explicit path, or the SANITIZER_LIB_PATH
// environment variable, still wins where the install is somewhere else
// entirely.
//
// If the library is already loaded, LoadFromToolkit returns nil without
// opening anything, exactly as [Load] does. If no candidate opens, it returns
// *NotFoundError naming every path tried; it never panics and never falls back
// to a library that is not this one.
func LoadFromToolkit() error {
	if Loaded() {
		return nil
	}
	paths := SearchPaths()
	if len(paths) == 0 {
		return &NotFoundError{Tried: []LoadAttempt{{
			Path: toolkitLibName,
			Err:  fmt.Errorf("compute sanitizer does not ship for %s", runtime.GOOS),
		}}}
	}
	tried := make([]LoadAttempt, 0, len(paths))
	for _, p := range paths {
		err := Load(p)
		if err == nil {
			return nil
		}
		tried = append(tried, LoadAttempt{Path: p, Err: err})
	}
	return &NotFoundError{Tried: tried}
}
