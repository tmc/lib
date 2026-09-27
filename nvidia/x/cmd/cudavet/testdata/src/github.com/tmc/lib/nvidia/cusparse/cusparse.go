// Package cusparse is a declaration-only stand-in for the real cusparse package.
package cusparse

import "github.com/tmc/lib/nvidia/driver"

type Handle uintptr

type ConstSpMatDescr uintptr

type ConstDnVecDescr uintptr

type DnVecDescr uintptr

type Operation int32

const OperationNonTranspose Operation = 0

type SpMVAlg int32

const SpmvAlgDefault SpMVAlg = 0

type DataType int32

const (
	R32F DataType = 0
	R64F DataType = 1
	C32F DataType = 4
)

type Scalar struct{}

func Float32Scalar(x float32) Scalar       { return Scalar{} }
func Float64Scalar(x float64) Scalar       { return Scalar{} }
func Complex64Scalar(x complex64) Scalar   { return Scalar{} }
func Complex128Scalar(x complex128) Scalar { return Scalar{} }
func Int32Scalar(x int32) Scalar           { return Scalar{} }

func (h Handle) SpMV(opA Operation, alpha Scalar, matA ConstSpMatDescr, vecX ConstDnVecDescr, beta Scalar, vecY DnVecDescr, computeType DataType, alg SpMVAlg, externalBuffer driver.DevicePtr) error {
	return nil
}
