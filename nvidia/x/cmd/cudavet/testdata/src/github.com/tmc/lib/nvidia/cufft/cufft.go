// Package cufft is a declaration-only stand-in for the real cufft package.
package cufft

import "github.com/tmc/lib/nvidia/driver"

type Direction int32

const (
	Forward Direction = -1
	Inverse Direction = 1
)

type Type int32

const C2C Type = 0x29

type Handle int32

func Plan1d(nx int32, type_ Type, batch int32) (Handle, error) { return 0, nil }
func (h Handle) Destroy() error                                { return nil }

func (h Handle) ExecC2C(idata driver.DevicePtr, odata driver.DevicePtr, direction Direction) error {
	return nil
}
