//go:build linux

package nbd

import (
	"fmt"
	"os"
	"syscall"
)

const ioctlType = 0xab

var (
	requestSetSock      = ioctlRequest(ioctlType, 0)
	requestSetBlockSize = ioctlRequest(ioctlType, 1)
	requestSetSize      = ioctlRequest(ioctlType, 2)
	requestDoIt         = ioctlRequest(ioctlType, 3)
	requestClearSock    = ioctlRequest(ioctlType, 4)
	requestClearQueue   = ioctlRequest(ioctlType, 5)
	requestSetSizeBlock = ioctlRequest(ioctlType, 7)
	requestDisconnect   = ioctlRequest(ioctlType, 8)
	requestSetTimeout   = ioctlRequest(ioctlType, 9)
	requestSetFlags     = ioctlRequest(ioctlType, 10)
)

const (
	iocNrbits   = 8
	iocTypebits = 8
	iocSizebits = 14
	iocDirbits  = 2

	iocNrshift   = 0
	iocTypeshift = iocNrshift + iocNrbits
	iocSizeshift = iocTypeshift + iocTypebits
	iocDirshift  = iocSizeshift + iocSizebits

	iocNone = 0
)

func ioctlRequest(typ, nr uintptr) uintptr { return ioc(iocNone, typ, nr, 0) }
func ioc(dir, typ, nr, size uintptr) uintptr {
	return (dir << iocDirshift) | (typ << iocTypeshift) | (nr << iocNrshift) | (size << iocSizeshift)
}

// KernelDevice is an open Linux /dev/nbdX descriptor.
type KernelDevice struct {
	file *os.File
}

// OpenKernelDevice opens a Linux /dev/nbdX node.
func OpenKernelDevice(path string) (*KernelDevice, error) {
	f, err := os.OpenFile(path, os.O_RDWR|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("nbd open kernel device %s: %w", path, err)
	}
	return &KernelDevice{file: f}, nil
}

func (d *KernelDevice) FD() int {
	if d == nil || d.file == nil {
		return -1
	}
	return int(d.file.Fd())
}

func (d *KernelDevice) Close() error {
	if d == nil || d.file == nil {
		return ErrClosed
	}
	f := d.file
	d.file = nil
	if err := f.Close(); err != nil {
		return fmt.Errorf("nbd close kernel device: %w", err)
	}
	return nil
}

func (d *KernelDevice) SetSocket(fd int) error {
	return d.ioctl(requestSetSock, uintptr(fd), "set socket")
}
func (d *KernelDevice) SetBlockSize(size uint64) error {
	return d.ioctl(requestSetBlockSize, uintptr(size), "set block size")
}
func (d *KernelDevice) SetSize(size uint64) error {
	return d.ioctl(requestSetSize, uintptr(size), "set size")
}
func (d *KernelDevice) SetSizeBlocks(blocks uint64) error {
	return d.ioctl(requestSetSizeBlock, uintptr(blocks), "set size blocks")
}
func (d *KernelDevice) SetFlags(flags uint64) error {
	return d.ioctl(requestSetFlags, uintptr(flags), "set flags")
}
func (d *KernelDevice) SetTimeout(seconds uint64) error {
	return d.ioctl(requestSetTimeout, uintptr(seconds), "set timeout")
}
func (d *KernelDevice) Disconnect() error  { return d.ioctl(requestDisconnect, 0, "disconnect") }
func (d *KernelDevice) ClearSocket() error { return d.ioctl(requestClearSock, 0, "clear socket") }
func (d *KernelDevice) ClearQueue() error  { return d.ioctl(requestClearQueue, 0, "clear queue") }
func (d *KernelDevice) DoIt() error        { return d.ioctl(requestDoIt, 0, "do it") }

func (d *KernelDevice) ioctl(req uintptr, arg uintptr, op string) error {
	if d == nil || d.file == nil {
		return ErrClosed
	}
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(d.FD()), req, arg)
	if errno != 0 {
		return fmt.Errorf("nbd %s: %w", op, errno)
	}
	return nil
}
