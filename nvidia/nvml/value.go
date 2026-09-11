package nvml

import (
	"encoding/binary"
	"math"
	"math/bits"
	"runtime"
)

// Float64 reads dVal. Use it for ValueTypeDouble; Value carries no discriminator.
func (v Value) Float64() float64 { return math.Float64frombits(v.Data[0]) }

// Int32 reads siVal. Use it for ValueTypeSignedInt.
func (v Value) Int32() int32 { return int32(v.Uint32()) }

// Uint32 reads uiVal. Use it for ValueTypeUnsignedInt.
func (v Value) Uint32() uint32 {
	var data [8]byte
	binary.NativeEndian.PutUint64(data[:], v.Data[0])
	return binary.NativeEndian.Uint32(data[:])
}

// UnsignedLong reads ulVal and widens it to uint64. Use it for
// ValueTypeUnsignedLong. C unsigned long is 32 bits on Windows and on
// 32-bit targets, and 64 bits on other supported targets.
func (v Value) UnsignedLong() uint64 {
	if runtime.GOOS == "windows" || bits.UintSize == 32 {
		return uint64(v.Uint32())
	}
	return v.Uint64()
}

// Uint64 reads ullVal. Use it for ValueTypeUnsignedLongLong.
func (v Value) Uint64() uint64 { return v.Data[0] }

// Int64 reads sllVal. Use it for ValueTypeSignedLongLong.
func (v Value) Int64() int64 { return int64(v.Data[0]) }

// Uint16 reads usVal. Use it for ValueTypeUnsignedShort.
func (v Value) Uint16() uint16 {
	var data [8]byte
	binary.NativeEndian.PutUint64(data[:], v.Data[0])
	return binary.NativeEndian.Uint16(data[:])
}
