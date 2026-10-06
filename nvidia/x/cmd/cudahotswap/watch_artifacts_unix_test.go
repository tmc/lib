//go:build linux || darwin

package main

import (
	"errors"
	"io"
	"os"
	"syscall"
)

// Open nonblocking and no-follow before checking the descriptor's regular type.
// A replaced pathname cannot turn this read into an unbounded FIFO wait.
func watchReadArtifact(path string, limit int64) (data []byte, err error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	defer func() { err = errors.Join(err, f.Close()) }()
	before, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Size() <= 0 || before.Size() > limit {
		return nil, errors.New("artifact is not bounded ordinary file")
	}
	data, err = io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if len(data) > int(limit) || int64(len(data)) != before.Size() || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || !os.SameFile(before, after) {
		return nil, errors.New("artifact changed while reading")
	}
	return data, nil
}
