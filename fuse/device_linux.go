//go:build linux

package fuse

import (
	"fmt"
	"os"
	"syscall"
)

type Device struct {
	file *os.File
}

func OpenDevice(path string) (*Device, error) {
	if path == "" {
		path = "/dev/fuse"
	}
	f, err := os.OpenFile(path, os.O_RDWR|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("fuse open device %s: %w", path, err)
	}
	return &Device{file: f}, nil
}

func (d *Device) FD() int {
	if d == nil || d.file == nil {
		return -1
	}
	return int(d.file.Fd())
}

func (d *Device) Read(p []byte) (int, error) {
	if d == nil || d.file == nil {
		return 0, ErrClosed
	}
	return d.file.Read(p)
}

func (d *Device) Write(p []byte) (int, error) {
	if d == nil || d.file == nil {
		return 0, ErrClosed
	}
	return d.file.Write(p)
}

func (d *Device) Close() error {
	if d == nil || d.file == nil {
		return ErrClosed
	}
	f := d.file
	d.file = nil
	if err := f.Close(); err != nil {
		return fmt.Errorf("fuse close device: %w", err)
	}
	return nil
}
