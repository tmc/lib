//go:build !linux

package cudacache

import (
	"fmt"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
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
	if !filepath.IsAbs(name) {
		return "", fmt.Errorf("loaded library %s: no mapping table on this system: %w", name, cudaerrors.ErrUnsupported)
	}
	target, err := filepath.EvalSymlinks(name)
	if err != nil {
		return "", fmt.Errorf("loaded library %s: %w", name, err)
	}
	if _, err := StatTool(target); err != nil {
		return "", fmt.Errorf("loaded library %s: %w", name, err)
	}
	return target, nil
}
