// Package cublas is a declaration-only stand-in for the real cublas package.
package cublas

import "github.com/tmc/lib/nvidia/driver"

type CallError struct{ Result Status }

func (e *CallError) Error() string { return "" }

type Status int32

func (s Status) Error() string { return "" }

type Operation int32

const (
	OpN Operation = 0
	OpT Operation = 1
)

type DataType int32

const R32F DataType = 0

type ComputeType int32

const (
	Compute32F ComputeType = 68
	Compute64F ComputeType = 70
)

type Handle uintptr

func Create() (Handle, error)                    { return 0, nil }
func (h *Handle) Close() error                   { return nil }
func (h Handle) SetStream(s driver.Stream) error { return nil }

func (h Handle) Saxpy(n int32, alpha *float32, x driver.DevicePtr, incx int32, y driver.DevicePtr, incy int32) error {
	return nil
}

func (h Handle) Snrm2(n int32, x uintptr, incx int32) (float32, error) { return 0, nil }

func (h Handle) GemmEx(transa, transb Operation, m, n, k int32, alpha uintptr, a driver.DevicePtr, aType DataType, lda int32, b driver.DevicePtr, bType DataType, ldb int32, beta uintptr, c driver.DevicePtr, cType DataType, ldc int32, computeType ComputeType) error {
	return nil
}
