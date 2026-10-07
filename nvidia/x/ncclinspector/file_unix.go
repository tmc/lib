//go:build linux || darwin

package ncclinspector

import (
	"errors"
	"os"
	"syscall"
)

func openOrdinary(path string) (*os.File, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	info, err := f.Stat()
	if err == nil && !info.Mode().IsRegular() {
		err = errors.New("inspector: nonregular file")
	}
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	return f, nil
}
