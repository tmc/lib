package nvrtcx

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// libraryIdentity describes the file behind the NVRTC library opened as
// name, for the compiler cache key: its resolved path, size, modification
// time and, where the system reports them, device and inode. Replacing the
// file or rewriting it in place changes the result. It returns "" when the
// file cannot be located.
func libraryIdentity(name string) string {
	path := resolvedLibrary(name)
	if path == "" {
		return ""
	}
	fi, err := os.Stat(path)
	if err != nil || !fi.Mode().IsRegular() {
		return ""
	}
	return fmt.Sprintf("file=%q size=%d mtime=%d%s", path, fi.Size(), fi.ModTime().UnixNano(), fileSerial(fi))
}

var resolvedLibraries sync.Map // opened name -> resolved path

// resolvedLibrary returns the file the library opened as name resolves to.
// A loaded library cannot be replaced by another path within a process, so
// a successful resolution is remembered; the file's identity is not.
func resolvedLibrary(name string) string {
	if p, ok := resolvedLibraries.Load(name); ok {
		return p.(string)
	}
	var path string
	if filepath.IsAbs(name) {
		path, _ = filepath.EvalSymlinks(name)
	} else {
		path = mappedLibrary(name)
	}
	if path != "" {
		resolvedLibraries.Store(name, path)
	}
	return path
}

// mappedLibraryIn returns the one file in a /proc/self/maps table that can
// be the library dlopen found for soname, or "" if there is not exactly one.
// A file named soname, or soname with a longer version, is preferred; only
// if there is none does any version of the same library match, because
// "libcuda.so.1" links to "libcuda.so.580.95.05". libnvrtc-builtins never
// matches libnvrtc.
func mappedLibraryIn(maps, soname string) string {
	if path, n := mappedWithPrefix(maps, soname, soname); n > 0 {
		return path
	}
	stem := soname
	if i := strings.Index(soname, ".so."); i >= 0 {
		stem = soname[:i+len(".so")]
	}
	path, _ := mappedWithPrefix(maps, soname, stem)
	return path
}

// mappedWithPrefix returns the number of distinct mapped files whose base
// name is soname or starts with prefix and a dot, and the file if there is
// exactly one.
func mappedWithPrefix(maps, soname, prefix string) (string, int) {
	seen := map[string]bool{}
	found := ""
	for line := range strings.SplitSeq(maps, "\n") {
		// Only the pathname field can contain a slash.
		i := strings.Index(line, " /")
		if i < 0 {
			continue
		}
		path := strings.TrimSuffix(line[i+1:], " (deleted)")
		base := filepath.Base(path)
		if base != soname && !strings.HasPrefix(base, prefix+".") {
			continue
		}
		seen[path] = true
		found = path
	}
	if len(seen) != 1 {
		return "", len(seen)
	}
	return found, 1
}
