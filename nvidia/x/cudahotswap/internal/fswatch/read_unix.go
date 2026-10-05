//go:build unix

package fswatch

import (
	"errors"
	"fmt"
	"io"
	"os"

	"golang.org/x/sys/unix"
)

// ReadFile reads a regular file without following a final symlink. It refuses
// nonregular files before reading and bounds both the initial size and growth.
func ReadFile(path string, limit int64) ([]byte, error) {
	if limit < 0 || limit > 8<<20 {
		return nil, fmt.Errorf("read file: limit must be between 0 and 8 MiB")
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NONBLOCK|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	info, err := f.Stat()
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return nil, errors.Join(fmt.Errorf("read file: nonregular file or size limit exceeded"), f.Close())
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	err = errors.Join(err, f.Close())
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("read file: grew beyond size limit")
	}
	return data, nil
}
