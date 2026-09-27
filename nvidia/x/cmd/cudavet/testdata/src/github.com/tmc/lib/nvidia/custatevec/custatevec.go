// Package custatevec is a declaration-only stand-in for the real custatevec
// package, which binds all device memory as uintptr.
package custatevec

type Handle uintptr

type DataType int32

const C64F DataType = 5

type StateVectorType int32

const StateVectorTypeZero StateVectorType = 0

func Create() (Handle, error) { return 0, nil }

func (h Handle) InitializeStateVector(sv uintptr, svDataType DataType, nIndexBits uint32, svType StateVectorType) error {
	return nil
}
