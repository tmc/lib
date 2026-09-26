package rawenum

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/cufft"
	"github.com/tmc/lib/nvidia/cusolver"
	"github.com/tmc/lib/nvidia/driver"
)

const opN cusolver.Operation = 0 // want `literal 0 of type cusolver.Operation; use cusolver.OpN`

func trans() cusolver.Operation {
	return cusolver.Operation(1) // want `literal 1 of type cusolver.Operation; use cusolver.OpT`
}

func named() cusolver.Operation {
	return cusolver.OpT
}

func unnamed() cusolver.Operation {
	return cusolver.Operation(7)
}

// fft leaves cufft.Direction to fftdirection.
func fft(p cufft.Plan, a driver.DevicePtr) error {
	return p.ExecC2C(a, a, -1)
}

// flags passes an untyped 0 for no flags; the context gives the type.
func flags(h cusolver.Handle, a driver.DevicePtr) error {
	return h.Dgetrs(0, 4, 1, a, 4, a, a, 4, a)
}

// size wants a value of the type, whatever its name.
func size() uintptr {
	return unsafe.Sizeof(cusolver.Operation(0))
}
