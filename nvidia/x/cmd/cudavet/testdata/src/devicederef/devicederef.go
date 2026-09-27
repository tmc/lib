package devicederef

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func direct(d driver.DevicePtr) float32 {
	p := (*float32)(unsafe.Pointer(uintptr(d))) // want `unsafe.Pointer of a device address makes it a Go pointer; keep it a DevicePtr`
	return *p
}

func derived(d driver.DevicePtr) unsafe.Pointer {
	u := uintptr(d) + 16
	return unsafe.Pointer(u) // want `unsafe.Pointer of a device address makes it a Go pointer; keep it a DevicePtr`
}

func added(d driver.DevicePtr) unsafe.Pointer {
	return unsafe.Add(nil, d) // want `unsafe.Add\(nil, d\) of a device address makes it a Go pointer; keep it a DevicePtr`
}

func offset(d driver.DevicePtr) driver.DevicePtr {
	return d + 16
}

func host(buf []float32) unsafe.Pointer {
	return unsafe.Pointer(&buf[0])
}
