package cudacache

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// LoadedLibraryPath returns the file this process has mapped for the shared
// library opened as name, which is a path or a bare soname such as
// "libnvrtc.so.13".
//
// On Linux it reads /proc/self/maps. It reports ErrToolIdentityChanged when
// the mapped file was unlinked, or when its path now names a different file
// (device or inode) than the one mapped: the bytes on disk are then not the
// bytes the process runs. Elsewhere there is no mapping table to read: an
// absolute name that is a regular file is returned as it resolves, and a bare
// soname reports an error wrapping cudaerrors.ErrUnsupported.
func LoadedLibraryPath(name string) (string, error) {
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return "", fmt.Errorf("loaded library %s: %w", name, err)
	}
	var ms []mapping
	if filepath.IsAbs(name) {
		target, err := filepath.EvalSymlinks(name)
		if err != nil {
			return "", fmt.Errorf("loaded library %s: %w", name, err)
		}
		ms = mappedFiles(maps, func(path string) bool { return path == target })
	} else {
		ms = libraryMatches(maps, name)
	}
	m, err := uniqueMapping(name, ms)
	if err != nil {
		return "", err
	}
	if m.deleted {
		return "", fmt.Errorf("loaded library %s: %s was deleted after it was mapped: %w", name, m.path, ErrToolIdentityChanged)
	}
	id, err := StatTool(m.path)
	if err != nil {
		return "", fmt.Errorf("loaded library %s: %w: %w", name, ErrToolIdentityChanged, err)
	}
	if id.Inode != m.inode || uint64(unix.Major(id.Device)) != m.major || uint64(unix.Minor(id.Device)) != m.minor {
		return "", fmt.Errorf("loaded library %s: %s is no longer the mapped file (inode %d, mapped %d): %w", name, m.path, id.Inode, m.inode, ErrToolIdentityChanged)
	}
	return m.path, nil
}
