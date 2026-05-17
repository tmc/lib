package kvm

import (
	"os"
	"runtime"
	"sync"
)

const APIVersion = 12

// Device is an open Linux UAPI handle.
type Device struct {
	file *os.File
}

// VM is an open Linux UAPI handle.
type VM struct {
	file     *os.File
	mmapSize int
	mu       sync.Mutex
	regions  map[uint32][]byte
	pins     map[uint32]*runtime.Pinner
}

// VCPU is an open Linux UAPI handle.
type VCPU struct {
	file *os.File
	run  []byte
}

// DeviceFD is an open Linux UAPI handle.
type DeviceFD struct {
	file *os.File
}

// FD returns the underlying file descriptor.
func (x *Device) FD() int {
	if x == nil || x.file == nil {
		return -1
	}
	return int(x.file.Fd())
}

// FD returns the underlying file descriptor.
func (x *VM) FD() int {
	if x == nil || x.file == nil {
		return -1
	}
	return int(x.file.Fd())
}

// FD returns the underlying file descriptor.
func (x *VCPU) FD() int {
	if x == nil || x.file == nil {
		return -1
	}
	return int(x.file.Fd())
}

// FD returns the underlying file descriptor.
func (x *DeviceFD) FD() int {
	if x == nil || x.file == nil {
		return -1
	}
	return int(x.file.Fd())
}
