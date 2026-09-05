package cublasmp_test

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublasmp"
	"github.com/tmc/lib/nvidia/nccl"
)

// TestUnavailableSurfaceIsStructured checks that every entry point of the
// bound surface reports *LibraryError when libcublasmp cannot be opened,
// rather than panicking on a nil function pointer. It covers the public
// wrappers as well, which is where a missing load check would hide.
//
// The assertions only mean something on a machine without the library, so the
// test skips where the library is installed rather than calling into a GPU.
func TestUnavailableSurfaceIsStructured(t *testing.T) {
	if err := cublasmp.Load(""); err == nil {
		t.Skip("libcublasmp is installed: the unavailable path is not reachable in this process")
	}
	alpha := float32(1)
	calls := []struct {
		name string
		call func() error
	}{
		{"Create", func() error { _, err := cublasmp.NewHandle(cublas.Stream(0)); return err }},
		{"Destroy", func() error { return cublasmp.Handle(0).Destroy() }},
		{"SetStream", func() error { return cublasmp.Handle(0).SetStream(cublas.Stream(0)) }},
		{"GetStream", func() error { _, err := cublasmp.Handle(0).GetStream(); return err }},
		{"GetVersion", func() error { _, err := cublasmp.GetVersion(); return err }},
		{"GridCreate", func() error {
			_, err := cublasmp.NewGrid(1, 1, cublasmp.ColMajor, nccl.Comm(0))
			return err
		}},
		{"GridDestroy", func() error { return cublasmp.Grid(0).Destroy() }},
		{"BufferRegister", func() error { return cublasmp.Grid(0).Register(0, 1) }},
		{"BufferDeregister", func() error { return cublasmp.Grid(0).Deregister(0) }},
		{"Malloc", func() error { _, err := cublasmp.Grid(0).Malloc(1); return err }},
		{"Free", func() error { return cublasmp.Grid(0).Free(0) }},
		{"MatrixDescriptorCreate", func() error {
			_, err := cublasmp.NewMatrixDescriptor(1, 1, 1, 1, 0, 0, 1, cublas.R32F, cublasmp.Grid(0))
			return err
		}},
		{"MatrixDescriptorDestroy", func() error { return cublasmp.MatrixDescriptor(0).Destroy() }},
		{"MatrixDescriptorInit", func() error {
			return cublasmp.MatrixDescriptor(0).Init(1, 1, 1, 1, 0, 0, 1, cublas.R32F, cublasmp.Grid(0))
		}},
		{"GemmBufferSize", func() error {
			_, _, err := cublasmp.Handle(0).GemmBufferSize(
				cublas.OpN, cublas.OpN, 1, 1, 1,
				unsafe.Pointer(&alpha), 0, 1, 1, cublasmp.MatrixDescriptor(0),
				0, 1, 1, cublasmp.MatrixDescriptor(0),
				unsafe.Pointer(&alpha), 0, 1, 1, cublasmp.MatrixDescriptor(0),
				cublas.Compute32F)
			return err
		}},
		{"Gemm", func() error {
			return cublasmp.Handle(0).Gemm(
				cublas.OpN, cublas.OpN, 1, 1, 1,
				unsafe.Pointer(&alpha), 0, 1, 1, cublasmp.MatrixDescriptor(0),
				0, 1, 1, cublasmp.MatrixDescriptor(0),
				unsafe.Pointer(&alpha), 0, 1, 1, cublasmp.MatrixDescriptor(0),
				cublas.Compute32F, 0, 0, unsafe.Pointer(&alpha), 0)
		}},
	}
	for _, c := range calls {
		err := c.call()
		if err == nil {
			t.Errorf("%s: returned nil with no library loaded", c.name)
			continue
		}
		var libErr *cublasmp.LibraryError
		if !errors.As(err, &libErr) {
			t.Errorf("%s: error = %v (%T), want *LibraryError", c.name, err, err)
		}
	}
	// Nothing above may have opened a sibling library.
	if cublas.Loaded() || nccl.Loaded() {
		t.Error("a failed cublasmp call loaded another library's handle")
	}
}
