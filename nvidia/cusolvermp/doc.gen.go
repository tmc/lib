// Package cusolvermp provides Go bindings for NVIDIA cuSOLVERMp, the distributed dense
// linear algebra library, libcusolverMp.
//
// Status codes, streams, data types and matrix operations use the types in
// [github.com/tmc/lib/nvidia/cusolver]; communicators use [github.com/tmc/lib/nvidia/nccl].
// This package opens and probes libcusolverMp on its own and shares no loader
// state with either of them.
//
// # Scope
//
// Handle lifecycle, process grids, matrix descriptors, workspace queries and the
// LU factorize-and-solve path ([Getrf] and [Getrs]) are bound. The QR family, the
// least-squares solver, the symmetric and generalized eigensolvers, the SVD, the
// polar decomposition and the Newton-Schulz iteration are declared by the header
// but deliberately not bound here.
//
// # Rank and device mapping
//
// Every rank calls [Create] with the device ordinal it owns, and that binding is
// fixed for the life of the handle. A handle is not portable between ranks, is not
// safe for concurrent use, and must be destroyed by the rank that created it.
// [CreateDeviceGrid] takes an already-initialized NCCL communicator whose rank
// count equals numRowDevices*numColDevices; the mapping argument decides whether a
// communicator rank becomes a grid row or a grid column first. Ranks that disagree
// about the grid shape or the mapping do not fail locally, they diverge.
//
// # Local layouts
//
// A distributed matrix is 2D block-cyclic. Each rank stores only its own local
// block and describes the global matrix with [CreateMatrixDesc]. [NUMROC] computes
// how many rows or columns a given process coordinate owns, which is what the
// local leading dimension has to accommodate; passing a leading dimension smaller
// than the local row count is a silent memory error, not a status code. Matrix
// indices IA and JA are one-based, following ScaLAPACK.
//
// [MatrixScatterH2D] and [MatrixGatherD2H] move a matrix between one rank's host
// buffer and the distributed blocks. NVIDIA's header marks both as shipped for
// early access and not part of the official release.
//
// # Collective ordering
//
// Every call that takes a [Grid] or a [MatrixDescriptor] is collective over the
// communicator behind that grid. All ranks must reach the same call, with matching
// arguments, in the same order. A rank that returns early on a local error, or that
// skips a call because a local check failed, leaves the remaining ranks waiting.
// Handle an error by propagating it to every rank and tearing down together.
//
// # Workspace lifetime
//
// [GetrfBufferSize] and [GetrsBufferSize] return a device size and a host size.
// The caller allocates both, keeps them alive until the stream has completed the
// operation, and does not share one workspace between two operations in flight.
// The sizes depend on the arguments they were queried with: re-query whenever the
// matrix, the descriptors, or the compute type change.
//
// # Coordinated cleanup
//
// Synchronize the stream, then destroy in reverse order of creation: matrix
// descriptors, then the grid, then the handle, and only then the communicator and
// the device buffers. Every rank performs the same teardown. Destroying a grid
// while a descriptor still refers to it, or freeing device memory the library is
// still reading, is undefined behavior that this package cannot detect.
//
// # Memory
//
// [DevicePtr] and [HostPtr] are plain addresses. They are not Go pointers, they do
// not keep an allocation alive, and the garbage collector cannot see through them.
// Do not pass the address of Go memory: these calls are asynchronous on the stream
// and the library reads the buffers after the call returns.
//
// [Malloc] and [Free] are the grid's own allocator, which registers the block for
// collective transfer as it allocates. [BufferRegister] and [BufferDeregister] do
// the registration alone, for memory the caller allocated elsewhere; the caller
// keeps ownership and must deregister before freeing.
//
// The index below covers the wrappers generated from the header's declarations.
// The memory, layout and factorization entry points listed above are written by
// hand, next to the ownership rule each one carries, and do not appear in it.
//
// See the [cuSOLVERMp documentation] for the block-cyclic layout rules and the
// supported precision combinations.
//
// # Lifecycle
//
//   - [NewHandle]: creates a cuSOLVERMp handle bound to one device and one stream
//   - [Handle.CreateDeviceGrid]: describes the process grid the distributed matrices are laid out over
//   - [Grid.CreateMatrixDesc]: describes a 2D block-cyclic distributed matrix over a grid
//   - [Handle.Destroy]: releases a handle created by [NewHandle]
//   - [Grid.Destroy]: releases a grid created by [Handle.CreateDeviceGrid]
//   - [MatrixDescriptor.Destroy]: releases a descriptor created by [Grid.CreateMatrixDesc]
//   - [Handle.GetStream]: returns the stream operations on handle are enqueued on
//   - [Handle.GetVersion]: returns the version of the loaded library
//   - [Handle.SetStream]: redirects subsequent operations on handle to stream
//
// # Logging
//
//   - [LoggerForceDisable]: turns the library's logging off for the rest of the process
//   - [LoggerOpenFile]: sends the library's log to a file
//   - [LoggerSetLevel]: sets the library's log verbosity
//   - [LoggerSetMask]: selects which log categories the library emits
//
// # Other
//
//   - [Handle.GetMathMode]: returns the arithmetic mode set by [Handle.SetMathMode]
//   - [Handle.SetMathMode]: selects the arithmetic the library may use
//
// [cuSOLVERMp documentation]: https://docs.nvidia.com/cuda/cusolvermp/
package cusolvermp

// Code generated by gen-cusolvermp from cusolverMp.h. DO NOT EDIT.
