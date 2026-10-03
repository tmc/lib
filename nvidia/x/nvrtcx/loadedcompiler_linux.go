//go:build linux

package nvrtcx

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

func checkedMappedCompilerPath(name string) (string, bool, error) {
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return "", true, fmt.Errorf("nvrtcx: loaded compiler mappings: %w", err)
	}
	return checkedCompilerMapping(string(maps), name)
}

func checkedCompilerMapping(maps, name string) (string, bool, error) {
	path := ""
	if filepath.IsAbs(name) {
		var err error
		path, err = filepath.EvalSymlinks(name)
		if err != nil {
			return "", true, fmt.Errorf("nvrtcx: loaded compiler path: %w", err)
		}
	} else {
		path = mappedLibraryIn(maps, name)
	}
	if path == "" {
		return "", true, fmt.Errorf("nvrtcx: loaded compiler %q absent or ambiguous", name)
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", true, fmt.Errorf("nvrtcx: loaded compiler file: %w", err)
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok || !info.Mode().IsRegular() {
		return "", true, fmt.Errorf("nvrtcx: loaded compiler has no regular file identity")
	}
	matched := false
	for _, line := range strings.Split(maps, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		// The path begins after five whitespace-delimited fields and may contain
		// spaces. Locate it without reconstructing or normalizing its bytes.
		at := strings.Index(line, " /")
		if at < 0 {
			continue
		}
		raw := line[at+1:]
		p := strings.TrimSuffix(raw, " (deleted)")
		if p != path {
			continue
		}
		if raw != p {
			return "", true, fmt.Errorf("nvrtcx: loaded compiler file was deleted")
		}
		major, minor, ok := strings.Cut(fields[3], ":")
		if !ok {
			return "", true, fmt.Errorf("nvrtcx: malformed compiler mapping")
		}
		ma, e1 := strconv.ParseUint(major, 16, 32)
		mi, e2 := strconv.ParseUint(minor, 16, 32)
		ino, e3 := strconv.ParseUint(fields[4], 10, 64)
		if e1 != nil || e2 != nil || e3 != nil || ino != st.Ino || ma != uint64(unix.Major(uint64(st.Dev))) || mi != uint64(unix.Minor(uint64(st.Dev))) {
			return "", true, fmt.Errorf("nvrtcx: compiler path no longer names the loaded inode")
		}
		matched = true
	}
	if !matched {
		return "", true, fmt.Errorf("nvrtcx: compiler file is not mapped")
	}
	return path, true, nil
}
