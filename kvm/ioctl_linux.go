//go:build linux

package kvm

import (
	"syscall"
	"unsafe"
)

const ioctlType = 0xAE

const (
	iocNrbits   = 8
	iocTypebits = 8
	iocSizebits = 14
	iocDirbits  = 2

	iocNrshift   = 0
	iocTypeshift = iocNrshift + iocNrbits
	iocSizeshift = iocTypeshift + iocTypebits
	iocDirshift  = iocSizeshift + iocSizebits

	iocNone  = 0
	iocWrite = 1
	iocRead  = 2
)

var (
	requestGetAPIVersion       = io(ioctlType, 0x00)
	requestCreateVM            = io(ioctlType, 0x01)
	requestCheckExtension      = io(ioctlType, 0x03)
	requestGetVCPUMMapSize     = io(ioctlType, 0x04)
	requestCreateVCPU          = io(ioctlType, 0x41)
	requestSetUserMemoryRegion = iow(ioctlType, 0x46, unsafe.Sizeof(UserspaceMemoryRegion{}))
	requestIRQLine             = iow(ioctlType, 0x61, unsafe.Sizeof(IRQLevel{}))
	requestGetOneReg           = iow(ioctlType, 0xab, unsafe.Sizeof(OneReg{}))
	requestSetOneReg           = iow(ioctlType, 0xac, unsafe.Sizeof(OneReg{}))
	requestARMVCPUInit         = iow(ioctlType, 0xae, unsafe.Sizeof(VCPUInit{}))
	requestARMPreferredTarget  = ior(ioctlType, 0xaf, unsafe.Sizeof(VCPUInit{}))
	requestCreateDevice        = iowr(ioctlType, 0xe0, unsafe.Sizeof(CreateDevice{}))
	requestSetDeviceAttr       = iow(ioctlType, 0xe1, unsafe.Sizeof(DeviceAttr{}))
	requestGetDeviceAttr       = iow(ioctlType, 0xe2, unsafe.Sizeof(DeviceAttr{}))
	requestRun                 = io(ioctlType, 0x80)
)

type runHeader struct {
	RequestInterruptWindow     uint8
	ImmediateExit              uint8
	Padding1                   [6]uint8
	ExitReason                 uint32
	ReadyForInterruptInjection uint8
	IfFlag                     uint8
	Flags                      uint16
	Cr8                        uint64
	ApicBase                   uint64
}

const runHeaderSize = int(unsafe.Sizeof(runHeader{}))

func io(typ, nr uintptr) uintptr                 { return ioc(iocNone, typ, nr, 0) }
func iow(typ, nr uintptr, size uintptr) uintptr  { return ioc(iocWrite, typ, nr, size) }
func ior(typ, nr uintptr, size uintptr) uintptr  { return ioc(iocRead, typ, nr, size) }
func iowr(typ, nr uintptr, size uintptr) uintptr { return ioc(iocRead|iocWrite, typ, nr, size) }
func ioc(dir, typ, nr, size uintptr) uintptr {
	return (dir << iocDirshift) | (typ << iocTypeshift) | (nr << iocNrshift) | (size << iocSizeshift)
}

func ioctl(fd int, req uintptr, arg uintptr) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, arg)
	if errno != 0 {
		return errno
	}
	return nil
}

func ioctlRet(fd int, req uintptr, arg uintptr) (int, error) {
	r0, _, errno := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, arg)
	if errno != 0 {
		return 0, errno
	}
	return int(r0), nil
}
