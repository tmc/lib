//go:build darwin || linux

package cudadiageval

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"

	"golang.org/x/sys/unix"
)

// Each component is opened relative to a held directory descriptor. O_NOFOLLOW
// refuses a symlink atomically; O_NONBLOCK prevents a FIFO open from blocking.
type registrationRoot struct{ fd int }

func registrationOpenRoot(path string) (registrationRoot, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK, 0)
	if err != nil {
		return registrationRoot{}, err
	}
	return registrationRoot{fd}, nil
}
func (r registrationRoot) close() error { return unix.Close(r.fd) }
func (r registrationRoot) open(ctx context.Context, path string) (*os.File, error) {
	if !fs.ValidPath(path) || path == "." {
		return nil, errors.New("verify registration: invalid artifact path")
	}
	parent := r.fd
	owned := false
	parts := strings.Split(path, "/")
	for i, name := range parts {
		if err := ctx.Err(); err != nil {
			if owned {
				return nil, errors.Join(err, unix.Close(parent))
			}
			return nil, err
		}
		last := i == len(parts)-1
		flags := unix.O_RDONLY | unix.O_CLOEXEC | unix.O_NOFOLLOW | unix.O_NONBLOCK
		if !last {
			flags |= unix.O_DIRECTORY
		}
		fd, err := unix.Openat(parent, name, flags, 0)
		if owned {
			if closeErr := unix.Close(parent); closeErr != nil {
				if err == nil {
					err = unix.Close(fd)
				}
				return nil, errors.Join(closeErr, err)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("verify registration: no-follow artifact open refused: %w", err)
		}
		if last {
			return os.NewFile(uintptr(fd), path), nil
		}
		parent, owned = fd, true
	}
	return nil, errors.New("verify registration: invalid artifact path")
}
