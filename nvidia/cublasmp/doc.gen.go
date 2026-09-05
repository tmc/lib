// Package cublasmp provides Go bindings for cuBLASMp, NVIDIA's multi-process,
// multi-GPU dense linear algebra library, libcublasmp.
//
// The bound surface is the library handle, the NCCL-backed process grid,
// 2D block-cyclic matrix descriptors, the GEMM workspace query, and the
// distributed general matrix-matrix product. cuBLASMp's other level-3
// routines and its descriptor-driven Matmul are declared in the pinned
// header but are not bound here.
//
// Scalars, transpose modes, compute types, and streams use the types in
// [github.com/tmc/lib/nvidia/cublas]; the communicator uses
// [github.com/tmc/lib/nvidia/nccl]. This package loads and probes
// libcublasmp independently of both, and holds no reference to a cuBLAS or
// NCCL handle of its own.
//
// # Ownership
//
// Device and rank mapping. A [Handle] is bound to the CUDA device and
// stream current when [NewHandle] is called; the caller sets the device
// before creating the handle and keeps it current for every later call on
// that handle. A [Grid] is bound to the NCCL communicator passed to
// [NewGrid], which fixes the rank set; the caller owns the communicator and
// must keep it alive for the grid's whole lifetime and destroy it only after
// [Grid.Destroy]. cuBLASMp derives a rank's (row, column) position from its
// NCCL rank and the grid layout, so the caller is responsible for the
// rank-to-device assignment that makes that position meaningful.
//
// Local tile layouts. A [MatrixDescriptor] describes a global matrix in 2D
// block-cyclic distribution; the local buffer it refers to is the caller's
// and is never allocated, freed, or bounds-checked by this package. The
// leading dimension must be at least the calling rank's local row count for
// the given grid and blocking. A descriptor borrows its grid: the grid must
// outlive every descriptor built from it.
//
// Collective ordering. [NewGrid], [Grid.Destroy], [Grid.Register],
// [Grid.Deregister], [Grid.Malloc], [Grid.Free], and the GEMM entry points
// are collective over the grid's communicator. Every rank must call them in
// the same order with matching arguments; a rank that skips one or reorders
// two deadlocks the job. Calls are issued on the handle's stream, so a rank
// must not change the handle's stream between the ranks' entries into a
// collective.
//
// Workspace lifetime. [Handle.GemmBufferSize] returns the device and host
// workspace sizes for one specific call shape. The workspaces are
// caller-allocated and caller-owned, must be at least the returned sizes,
// and must stay valid and unaliased until the work submitted to the stream
// has completed, not merely until [Handle.Gemm] returns. Registering the
// device workspace with [Grid.Register] (or allocating it with
// [Grid.Malloc]) is what makes the overlapping algorithms available; an
// unregistered workspace restricts the library to its non-overlapping path.
// Pin any Go memory the host workspace points into for that whole interval;
// a uintptr does not keep an allocation alive.
//
// Coordinated cleanup. Destroy in reverse order of creation and
// collectively: complete or synchronize the stream, then
// [MatrixDescriptor.Destroy], then [Grid.Deregister] or [Grid.Free] for
// every buffer registered with the grid, then [Grid.Destroy], then
// [Handle.Destroy], and only then destroy the NCCL communicator. Skipping a
// collective destroy on one rank hangs the others.
//
// A call made when libcublasmp is missing, or when the loaded library
// predates a symbol, returns a structured error rather than panicking. See
// [Load], [Loaded], and [SymbolError].
//
// Argument meanings, distribution rules, and the algorithm descriptions
// summarized above are NVIDIA's; see the [cuBLASMp documentation].
//
// # Process grid
//
//   - [Grid.Deregister]: Deregisters a previously registered device memory buffer from NCCL symmetric memory.
//   - [Grid.Register]: Registers a device memory buffer with NCCL symmetric memory for a process grid.
//   - [Grid.Free]: Deregisters and frees device memory allocated with [Grid.Malloc].
//   - [NewGrid]: Creates and initializes a process grid.
//   - [Grid.Destroy]: Destroys a process grid.
//   - [Grid.Malloc]: Allocates and registers device memory with NCCL symmetric memory.
//
// # Library handle
//
//   - [NewHandle]: Initializes the cuBLASMp library handle.
//   - [Handle.Destroy]: Destroys the cuBLASMp library handle.
//   - [Handle.GetStream]: Gets the current CUDA stream used by the cuBLASMp library.
//   - [GetVersion]: Returns the version number of the cuBLASMp library.
//   - [Handle.SetStream]: Sets the CUDA stream for cuBLASMp computations.
//
// # Distributed GEMM
//
//   - [Handle.Gemm]: Performs a distributed matrix-matrix multiplication (C = alpha * op(A) * op(B) + beta * C).
//   - [Handle.GemmBufferSize]: Computes the workspace size required by [Handle.Gemm].
//
// # Matrix descriptors
//
//   - [NewMatrixDescriptor]: Creates and initializes a matrix descriptor.
//   - [MatrixDescriptor.Destroy]: Destroys a matrix descriptor.
//   - [MatrixDescriptor.Init]: Initializes the values of an existing matrix descriptor.
//
// [cuBLASMp documentation]: https://docs.nvidia.com/cuda/cublasmp/
package cublasmp

// Code generated by gen-cublasmp from cublasmp.h. DO NOT EDIT.
