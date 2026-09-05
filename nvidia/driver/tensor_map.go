package driver

import "unsafe"

// NewTensorMap returns a zeroed tensor-map descriptor aligned to 128 bytes.
// Use it with TensorMapEncodeTiled, TensorMapEncodeIm2col, or
// TensorMapEncodeIm2colWide before passing the descriptor to a kernel.
// A TensorMap allocated with new or declared as a variable has only Go's
// natural alignment and does not satisfy the CUDA alignment contract.
// Keep the returned pointer, rather than copying its value to another variable.
func NewTensorMap() *TensorMap {
	// The interior pointer keeps the entire backing allocation alive.
	storage := new([255]byte)
	base := unsafe.Pointer(storage)
	offset := -uintptr(base) & 127
	return (*TensorMap)(unsafe.Add(base, offset))
}
