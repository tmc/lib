package devptruintptr

import (
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/custatevec"
	"github.com/tmc/lib/nvidia/driver"
)

func encode(m *driver.TensorMap, d driver.DevicePtr) error {
	return driver.TensorMapEncodeTiled(m, 0, 2, uintptr(d), nil, nil, nil, nil) // want `device pointer passed to driver.TensorMapEncodeTiled as uintptr; the binding does not type globalAddress as device memory`
}

func norm(h cublas.Handle, x driver.DevicePtr) (float32, error) {
	return h.Snrm2(4, uintptr(x), 1) // want `device pointer passed to cublas.Snrm2 as uintptr; the binding does not type x as device memory`
}

func typed(h cublas.Handle, x, y driver.DevicePtr) error {
	a := float32(2)
	return h.Saxpy(4, &a, x, 1, y, 1)
}

// custatevec names no DevicePtr: uintptr is its policy for all device memory.
func initialize(h custatevec.Handle, sv driver.DevicePtr) error {
	return h.InitializeStateVector(uintptr(sv), custatevec.C64F, 3, custatevec.StateVectorTypeZero)
}
