// Package cusolver is a declaration-only stand-in for the real cusolver
// package.
package cusolver

import "github.com/tmc/lib/nvidia/driver"

type Operation int32

const (
	OpN Operation = 0
	OpT Operation = 1
)

type Handle uintptr

func Create() (Handle, error)  { return 0, nil }
func (h *Handle) Close() error { return nil }

func (h Handle) DgetrfBufferSize(m, n int32, a driver.DevicePtr, lda int32) (int32, error) {
	return 0, nil
}

func (h Handle) Dgetrf(m, n int32, a driver.DevicePtr, lda int32, workspace, devIpiv, devInfo driver.DevicePtr) error {
	return nil
}

func (h Handle) Dgetrs(trans Operation, n, nrhs int32, a driver.DevicePtr, lda int32, devIpiv, b driver.DevicePtr, ldb int32, devInfo driver.DevicePtr) error {
	return nil
}
