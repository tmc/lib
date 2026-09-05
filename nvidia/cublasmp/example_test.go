package cublasmp_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublasmp"
	"github.com/tmc/lib/nvidia/nccl"
)

// Example reports the version of the loaded library and compares it against
// the header these bindings were generated from. It returns early when
// libcublasmp is not installed.
func Example() {
	version, err := cublasmp.GetVersion()
	if err != nil {
		fmt.Println("cuBLASMp unavailable:", err)
		return
	}
	fmt.Println("library version:", version)
	if version != cublasmp.HeaderVersion {
		fmt.Println("header version:", cublasmp.HeaderVersion)
	}
}

// ExampleHandle_Gemm shows the full call sequence for one distributed GEMM:
// communicator, process grid, matrix descriptors, workspace query, and the
// multiplication itself. Every rank of the job runs this same code with the
// same nprow, npcol, and layout, and enters each collective in the same order.
//
// It returns early on the first error, which on a machine without a NCCL job
// is the communicator. comm, deviceA, deviceB, and deviceC stand for a
// communicator and device buffers the caller already owns; this package
// allocates none of them.
func ExampleHandle_Gemm() {
	var (
		comm             nccl.Comm // from nccl.NewCommInitRank on this rank
		deviceA, deviceB uintptr   // local tiles of A and B, on this rank's device
		deviceC          uintptr   // local tile of C, written in place
		nprow, npcol               = int64(1), int64(1)
		m, n, k          int64     = 128, 128, 128
		mb, nb           int64     = 32, 32
	)

	grid, err := cublasmp.NewGrid(nprow, npcol, cublasmp.ColMajor, comm)
	if err != nil {
		fmt.Println("grid:", err)
		return
	}
	defer grid.Destroy()

	// The stream is bound at handle creation; every call on this handle is
	// ordered on it, and the caller synchronizes it before reading C.
	h, err := cublasmp.NewHandle(cublas.Stream(0))
	if err != nil {
		fmt.Println("handle:", err)
		return
	}
	defer h.Destroy()

	descA, err := cublasmp.NewMatrixDescriptor(m, k, mb, nb, 0, 0, m, cublas.R32F, grid)
	if err != nil {
		fmt.Println("descriptor:", err)
		return
	}
	defer descA.Destroy()
	descB, err := cublasmp.NewMatrixDescriptor(k, n, mb, nb, 0, 0, k, cublas.R32F, grid)
	if err != nil {
		fmt.Println("descriptor:", err)
		return
	}
	defer descB.Destroy()
	descC, err := cublasmp.NewMatrixDescriptor(m, n, mb, nb, 0, 0, m, cublas.R32F, grid)
	if err != nil {
		fmt.Println("descriptor:", err)
		return
	}
	defer descC.Destroy()

	// alpha and beta are host pointers, so unsafe.Pointer, not a device
	// address. Their element type must match the compute type.
	alpha, beta := float32(1), float32(0)

	// Submatrix indices are one-based, as in ScaLAPACK.
	const ia, ja, ib, jb, ic, jc = 1, 1, 1, 1, 1, 1

	deviceBytes, hostBytes, err := h.GemmBufferSize(
		cublas.OpN, cublas.OpN, m, n, k,
		unsafe.Pointer(&alpha), deviceA, ia, ja, descA,
		deviceB, ib, jb, descB,
		unsafe.Pointer(&beta), deviceC, ic, jc, descC,
		cublas.Compute32F)
	if err != nil {
		fmt.Println("workspace:", err)
		return
	}

	// Registering the device workspace with the grid is what makes the
	// overlap algorithms selectable; NVIDIA documents this, and this package
	// does not measure it. Grid.Malloc allocates and registers in one call.
	deviceWork, err := grid.Malloc(deviceBytes)
	if err != nil {
		fmt.Println("workspace:", err)
		return
	}
	defer grid.Free(deviceWork)
	hostWork := make([]byte, hostBytes)

	if err := h.Gemm(
		cublas.OpN, cublas.OpN, m, n, k,
		unsafe.Pointer(&alpha), deviceA, ia, ja, descA,
		deviceB, ib, jb, descB,
		unsafe.Pointer(&beta), deviceC, ic, jc, descC,
		cublas.Compute32F,
		deviceWork, deviceBytes, unsafe.Pointer(&hostWork[0]), hostBytes); err != nil {
		fmt.Println("gemm:", err)
		return
	}
	// hostWork must stay live and unmoved until the stream is synchronized.
	fmt.Println("submitted")
}
