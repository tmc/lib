//go:build linux

// Package nvshmem binds the host-callable API of the NVSHMEM library,
// libnvshmem_host.so.3, using purego. NVSHMEM gives the processing elements
// (PEs) of a multi-GPU job a symmetric heap: an allocation that every PE makes
// at the same time and can address on every other PE.
//
// # Host and device
//
// NVSHMEM is two APIs that share one set of headers. Only one of them is here.
//
// The device API is what most NVSHMEM code is written against: nvshmem_put,
// nvshmem_get, nvshmem_wait_until, the block- and warp-scoped collectives, the
// tile API. Those are __device__ functions. They are compiled into your CUDA
// kernels from libnvshmem_device.a and they execute on the GPU. There is no
// symbol for a program to call and no way for purego, cgo, or any other FFI
// mechanism to reach them: a function that runs on the device is not an entry
// point on the host. Device-side NVSHMEM code belongs in a .cu file compiled
// by nvcc against libnvshmem_device.a. Load the resulting cubin with the CUDA
// driver API, register it with [CUModuleInit] or [CULibraryInit] so NVSHMEM
// can patch its device state into the module, and launch it yourself.
//
// The host API is what this package binds: everything libnvshmem_host.so.3
// exports with C linkage. That is more than the name "host API" suggests.
// Alongside the setup and query calls, the host library exports host-callable
// remote memory access ([PutMem], [GetMem]), signal operations, barriers, and
// collectives, both blocking and stream-ordered. A Go program can move data
// between PEs without writing a kernel at all.
//
// # What is deliberately not bound
//
// Three groups of declarations in the public headers are unreachable and are
// therefore absent from this package rather than present and broken.
//
// nvshmem_init, nvshmem_init_thread, nvshmem_finalize and nvshmemx_init_attr
// are static inline functions in the headers. They call nvshmemi_init_thread
// and nvshmemi_finalize, which libnvshmem_host.so.3 compiles as C++ with local
// binding: the names are absent from its dynamic symbol table, so dlsym cannot
// find them under any spelling. Use [InitWithUniqueID] or [InitWithMPIComm],
// which call the exported nvshmemx_hostlib_init_attr, and [Finalize], which
// calls the exported nvshmemx_hostlib_finalize.
//
// nvshmemx_collective_launch and nvshmemx_collective_launch_query_gridsize are
// unreachable for the same reason, and this one costs real functionality. A
// kernel that uses device-side NVSHMEM collectives or synchronization must be
// launched collectively, and from Go it cannot be. A kernel that only issues
// non-blocking device-side RMA can be launched normally, with the CUDA driver
// API, after [CUModuleInit]. If your kernel needs a collective launch, launch
// it from a C++ program, not this package.
//
// The typed API - nvshmem_int_put, nvshmem_float_sum_reduce and roughly 1300
// more - is host-exported and reachable, but is not bound. The byte-granular
// [PutMem] and [GetMem] cover the same ground for data movement. The typed
// reductions do not have a byte-granular equivalent; if you need one, bind it.
//
// # Bootstrap
//
// NVSHMEM cannot initialize alone. Before the symmetric heap exists, the PEs
// have to agree on how many of them there are and how to reach each other,
// and NVSHMEM delegates that to a bootstrap plugin it dlopens at init.
//
// [InitWithUniqueID] uses nvshmem_bootstrap_uid.so.3, which links nothing
// outside libc and libstdc++ and needs no launcher. It is the path a Go
// program can drive end to end: one PE calls [NewUniqueID], distributes the
// result to the others by whatever means you already have, and every PE calls
// [InitWithUniqueID] with its own rank, the total rank count, and that same
// ID. Distributing the ID is your job. NVSHMEM does not do it, and this
// package does not either.
//
// [InitWithMPIComm] uses nvshmem_bootstrap_mpi.so.3, which links libmpi.so.40
// - the Open MPI ABI. It takes an MPI communicator as an opaque pointer, so
// the process must already be an MPI rank under mpirun, and the MPI library
// in the process has to be the one the plugin was built against. Handle values
// are not portable across MPI implementations.
//
// The remaining plugins NVSHMEM ships - PMI, PMI2, PMIx, and OpenSHMEM - are
// selected by environment and launcher, not by an argument to this package.
//
// Every PE must select its CUDA device before initializing. NVSHMEM binds the
// symmetric heap to the current context's device.
//
// # Ownership
//
// Symmetric-heap addresses are device addresses and are spelled uintptr, not
// unsafe.Pointer, so that Go memory cannot be passed where a remote GPU will
// dereference it. [Malloc], [Calloc] and [Align] return an address that is
// valid on every PE at the same offset and that only [Free] releases. Every PE
// must make the same allocation calls in the same order: these are collective,
// and a PE that skips one desynchronizes the heap for the whole job.
//
// An address the caller passes to a stream-ordered or non-blocking operation
// must stay allocated until the operation completes - [QuietOnStream] or a
// CUDA stream synchronize for the former, [Quiet] for the latter. Returning
// from the call is not completion.
//
// Host buffers registered with [BufferRegister] stay registered, and must stay
// mapped, until [BufferUnregister]. Pin any Go allocation for the whole
// interval; the library keeps the address.
//
// [Finalize] tears down the symmetric heap and the transports. Every address
// [Malloc] returned is invalid afterwards, on every PE.
//
// # Availability
//
// The library is opened on first use, or by an explicit [Load]. Import does
// not open it. Each wrapper checks its own symbol, so a package built against
// 3.7.2 still runs against an older library: a call whose symbol is absent
// returns a *[SymbolError] rather than panicking. [Has] reports availability
// in advance. Set NVSHMEM_LIB_PATH, or pass a path to [Load], to choose the
// library.
//
// Generated by gen-nvshmem from the pinned NVSHMEM 3.7.2 public headers.
//
// # Symmetric heap
//
//   - [Align].
//   - [Calloc].
//   - [Free].
//   - [Malloc].
//   - [Ptr].
//   - [BufferRegister].
//   - [BufferRegisterSymmetric].
//   - [BufferRegisterSymmetricAt].
//   - [BufferUnregister].
//   - [BufferUnregisterAll].
//   - [BufferUnregisterSymmetric].
//   - [MulticastPtr].
//
// # Collectives
//
//   - [AllToAllMem].
//   - [BroadcastMem].
//   - [FCollectMem].
//   - [AllToAllMemOnStream].
//   - [BroadcastMemOnStream].
//   - [FCollectMemOnStream].
//
// # Synchronization
//
//   - [Barrier].
//   - [BarrierAll].
//   - [Fence].
//   - [Quiet].
//   - [SyncAll].
//   - [BarrierAllOnStream].
//   - [BarrierOnStream].
//   - [Flush].
//   - [FlushOnStream].
//   - [QuietOnStream].
//   - [SyncAllOnStream].
//
// # Remote memory access
//
//   - [GetMem].
//   - [GetMemNBI].
//   - [PutMem].
//   - [PutMemNBI].
//   - [SignalFetch].
//   - [GetMemNBIOnStream].
//   - [GetMemOnStream].
//   - [PutMemNBIOnStream].
//   - [PutMemOnStream].
//   - [PutMemSignalNBIOnStream].
//   - [PutMemSignalOnStream].
//   - [SignalOpOnStream].
//   - [SignalWaitUntilOnStream].
//
// # Library lifecycle
//
//   - [GlobalExit].
//   - [QueryInitStatus].
//
// # PE and version queries
//
//   - [MyPE].
//   - [NPEs].
//
// # Teams
//
//   - [TeamDestroy].
//   - [TeamGetConfig].
//   - [TeamMyPE].
//   - [TeamNPEs].
//   - [TeamSplit2D].
//   - [TeamSplitStrided].
//   - [TeamSync].
//   - [TeamTranslatePE].
//   - [TeamGetUniqueID].
//   - [TeamInit].
//   - [TeamSyncOnStream].
//
// # Device code registration
//
//   - [CULibraryFinalize].
//   - [CULibraryInit].
//   - [CUModuleFinalize].
//   - [CUModuleInit].
//
// # Bootstrap
//
//   - [GetUniqueID].
//   - [SetAttrMPICommArgs].
//   - [SetAttrUniqueIDArgs].
//
// [NVSHMEM documentation]: https://docs.nvidia.com/nvshmem/api/index.html
package nvshmem

// Code generated by gen-nvshmem from nvshmem_host.h. DO NOT EDIT.
