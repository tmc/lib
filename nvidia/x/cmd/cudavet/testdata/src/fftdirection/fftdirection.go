package fftdirection

import (
	"github.com/tmc/lib/nvidia/cufft"
	"github.com/tmc/lib/nvidia/driver"
)

func forward(p cufft.Plan, a, b driver.DevicePtr) error {
	return p.ExecC2C(a, b, -1) // want `direction -1 written as a number; use cufft.Forward`
}

func inverse(p cufft.Plan, a, b driver.DevicePtr) error {
	return p.ExecC2C(a, b, cufft.Direction(1)) // want `direction 1 written as a number; use cufft.Inverse`
}

func named(p cufft.Plan, a, b driver.DevicePtr) error {
	if err := p.ExecC2C(a, b, cufft.Forward); err != nil {
		return err
	}
	return p.ExecC2C(b, a, cufft.Inverse)
}
