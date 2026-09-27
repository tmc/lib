package devptruintptr

import (
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/custatevec"
	"github.com/tmc/lib/nvidia/driver"
)

func encode(m *driver.TensorMap, d driver.DevicePtr) error {
	return driver.TensorMapEncodeTiled(m, driver.TensorMapDataTypeFloat32, 2, d, nil, nil, nil, nil, 0, 0, 0, 0)
}

func freeHost(d driver.DevicePtr) error {
	return driver.MemFreeHost(uintptr(d)) // want `device pointer passed to driver.MemFreeHost as uintptr; the binding does not type p as device memory`
}

func norm(h cublas.Handle, x driver.DevicePtr) (float32, error) {
	return h.Snrm2(4, x, 1)
}

func typed(h cublas.Handle, x, y driver.DevicePtr) error {
	a := float32(2)
	return h.Saxpy(4, &a, x, 1, y, 1)
}

// custatevec names no DevicePtr: uintptr is its policy for all device memory.
func initialize(h custatevec.Handle, sv driver.DevicePtr) error {
	return h.InitializeStateVector(uintptr(sv), custatevec.C64F, 3, custatevec.StateVectorTypeZero)
}
