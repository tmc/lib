package gonccl

import "unsafe"

type DType int

const (
	DTypeInvalid DType = iota
	Bool
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float32
	Float64
	Complex64
)

type ReduceOp int

const (
	Sum ReduceOp = iota
	Max
	Min
	Prod
)

type DeviceKind int

const (
	DeviceUnknown DeviceKind = iota
	DeviceHost
	DeviceCUDA
	DeviceROCM
	DeviceNetwork
	DeviceApple
)

type Device struct {
	Kind  DeviceKind
	Index int
	ID    string
}

type Buffer struct {
	Ptr    unsafe.Pointer
	Bytes  []byte
	Count  uint
	DType  DType
	Device Device
	Stream uintptr
	Owner  any
}

func (b Buffer) DevicePointer() unsafe.Pointer {
	if b.Ptr != nil {
		return b.Ptr
	}
	if len(b.Bytes) == 0 {
		return nil
	}
	return unsafe.Pointer(&b.Bytes[0])
}
