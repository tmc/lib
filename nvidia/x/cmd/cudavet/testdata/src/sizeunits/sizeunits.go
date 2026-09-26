package sizeunits

import (
	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/driver"
)

func bytes(h cusolver.Handle, a driver.DevicePtr, n int32) (driver.DevicePtr, error) {
	lwork, err := h.DgetrfBufferSize(n, n, a, n)
	if err != nil {
		return 0, err
	}
	return driver.MemAllocBytes(uint(lwork)) // want `lwork from DgetrfBufferSize counts float64 elements, not bytes; allocate MemAlloc\[float64\]\(lwork\)`
}

func wrongType(h cusolver.Handle, a driver.DevicePtr, n int32) (driver.DevicePtr, error) {
	lwork, err := h.DgetrfBufferSize(n, n, a, n)
	if err != nil {
		return 0, err
	}
	return driver.MemAlloc[float32](int(lwork)) // want `lwork from DgetrfBufferSize counts float64 elements, not float32s; allocate MemAlloc\[float64\]\(lwork\)`
}

func elements(h cusolver.Handle, a driver.DevicePtr, n int32) (driver.DevicePtr, error) {
	lwork, err := h.DgetrfBufferSize(n, n, a, n)
	if err != nil {
		return 0, err
	}
	return driver.MemAlloc[float64](int(lwork))
}
