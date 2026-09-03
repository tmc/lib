package metal

import (
	"unsafe"

	"github.com/tmc/apple/metal"
)

// Scalar is the set of types a Metal kernel scalar argument may have.
//
// Go's int and uint are deliberately absent. They are 64 bits wide on Apple
// silicon, while MSL's int and uint are 32, so binding one writes twice the
// bytes the kernel reads and leaves the kernel with a value it never agreed
// to. Convert to a fixed-width type at the call site instead.
type Scalar interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// SetScalar binds v into enc's argument table at index. Kernel scalars are
// small and change on every dispatch, so copying them inline is cheaper than
// allocating a buffer per argument per dispatch.
func SetScalar[T Scalar](enc metal.MTLComputeCommandEncoder, v T, index int) {
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&v)), unsafe.Sizeof(v)), uint(index))
}

// Threads returns the one-dimensional size covering n threads, for
// dispatching by thread count. An n below one becomes one: Metal rejects an
// empty grid.
func Threads(n int) metal.MTLSize {
	if n < 1 {
		n = 1
	}
	return metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}
}

// Threadgroups returns the one-dimensional size covering n threads in groups
// of group threads each, rounding up so a partial final group still runs. A
// group below one is treated as one.
func Threadgroups(n, group int) metal.MTLSize {
	if group < 1 {
		group = 1
	}
	if n < 1 {
		n = 1
	}
	return metal.MTLSize{Width: uint((n + group - 1) / group), Height: 1, Depth: 1}
}
