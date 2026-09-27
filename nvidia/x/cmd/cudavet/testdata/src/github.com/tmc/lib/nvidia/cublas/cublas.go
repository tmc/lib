// Package cublas is a declaration-only stand-in for the real cublas package.
package cublas

import "github.com/tmc/lib/nvidia/driver"

type CallError struct {
	Func   string
	Result Result
}

func (e *CallError) Error() string { return "" }
func (e *CallError) Unwrap() error { return e.Result }

type Result int32

func (r Result) Error() string { return "" }

type Operation int32

const (
	OpN Operation = 0
	OpT Operation = 1
)

type DataType int32

const (
	R32F DataType = 0
	R64F DataType = 1
)

type ComputeType int32

const (
	Compute32F ComputeType = 68
	Compute64F ComputeType = 70
)

type GemmAlgo int32

const GemmDefault GemmAlgo = -1

type Scalar struct{}

func Float16BitsScalar(b uint16) Scalar      { return Scalar{} }
func Float32Scalar(x float32) Scalar         { return Scalar{} }
func Float64Scalar(x float64) Scalar         { return Scalar{} }
func Complex64Scalar(x complex64) Scalar     { return Scalar{} }
func Complex128Scalar(x complex128) Scalar   { return Scalar{} }
func Int32Scalar(x int32) Scalar             { return Scalar{} }
func DeviceScalar(p driver.DevicePtr) Scalar { return Scalar{} }

type Handle uintptr

func Create() (Handle, error)                           { return 0, nil }
func (h *Handle) Close() error                          { return nil }
func (h Handle) SetStream(streamId driver.Stream) error { return nil }

func (h Handle) Saxpy(n int32, alpha *float32, x driver.DevicePtr, incx int32, y driver.DevicePtr, incy int32) error {
	return nil
}

func (h Handle) Snrm2(n int32, x driver.DevicePtr, incx int32) (float32, error) { return 0, nil }

func (h Handle) GemmEx(transa Operation, transb Operation, m int32, n int32, k int32, alpha Scalar, a driver.DevicePtr, atype DataType, lda int32, b driver.DevicePtr, btype DataType, ldb int32, beta Scalar, c driver.DevicePtr, ctype DataType, ldc int32, computeType ComputeType, algo GemmAlgo) error {
	return nil
}
