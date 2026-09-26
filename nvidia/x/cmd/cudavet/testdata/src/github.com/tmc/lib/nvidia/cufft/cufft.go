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

type Plan uintptr

func Plan1d(nx int32, typ Type, batch int32) (Plan, error) { return 0, nil }
func (p Plan) Destroy() error                              { return nil }

func (p Plan) ExecC2C(idata, odata driver.DevicePtr, direction Direction) error { return nil }
