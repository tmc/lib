// Package cublaslt is a declaration-only stand-in for the real cublaslt package.
package cublaslt

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

type Handle uintptr

type MatmulDesc uintptr

type MatrixLayout uintptr

type MatmulAlgo struct {
	Data [8]uint64
}

type ComputeType int32

const (
	Compute32F ComputeType = 68
	Compute64F ComputeType = 70
)

type DataType int32

const (
	R32F DataType = 0
	R64F DataType = 1
	R16F DataType = 2
	C32F DataType = 4
)

type Scalar struct{}

func Float16BitsScalar(b uint16) Scalar      { return Scalar{} }
func Float32Scalar(x float32) Scalar         { return Scalar{} }
func Float64Scalar(x float64) Scalar         { return Scalar{} }
func Complex64Scalar(x complex64) Scalar     { return Scalar{} }
func Complex128Scalar(x complex128) Scalar   { return Scalar{} }
func Int32Scalar(x int32) Scalar             { return Scalar{} }
func DeviceScalar(p driver.DevicePtr) Scalar { return Scalar{} }

type MatmulDescAttributes int32

const (
	MatmulDescScaleType   MatmulDescAttributes = 1
	MatmulDescPointerMode MatmulDescAttributes = 2
)

func (m MatmulDesc) SetAttribute(attr MatmulDescAttributes, buf unsafe.Pointer, sizeInBytes uint) error {
	return nil
}

func NewMatmulDesc(computeType ComputeType, scaleType DataType) (MatmulDesc, error) {
	return 0, nil
}

func (h Handle) Matmul(computeDesc MatmulDesc, alpha Scalar, a driver.DevicePtr, adesc MatrixLayout, b driver.DevicePtr, bdesc MatrixLayout, beta Scalar, c driver.DevicePtr, cdesc MatrixLayout, d driver.DevicePtr, ddesc MatrixLayout, algo *MatmulAlgo, workspace driver.DevicePtr, workspaceSizeInBytes uint, stream driver.Stream) error {
	return nil
}
