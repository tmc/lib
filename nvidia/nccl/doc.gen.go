// Package nccl provides low-level Go bindings for the NVIDIA Collective Communications Library (NCCL).
//
// NCCL provides optimized primitives for multi-GPU and multi-node collective
// communication: [Comm.AllReduce], [Comm.Broadcast], [Comm.ReduceScatter],
// [Comm.AllGather], [Comm.AlltoAll], [Comm.Send]/[Comm.Recv], and one-sided
// [Window.PutSignal]/[Comm.Signal]/[Comm.WaitSignal] operations.
//
// This package maps every function from nccl.h to a Go function using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libnccl.
// All functions return errors wrapping ncclResult_t with descriptive messages
// from ncclGetErrorString.
//
// The host-side NCCL device communicator and GIN setup APIs are also exposed.
// Device-side GIN entry points are CUDA-kernel-only and cannot be called through
// purego; kernels using those entry points must be compiled separately and launched
// by the host application.
//
// # Quick Start
//
//   // Initialize communicators across GPUs.
//   id, _ := nccl.GetUniqueId()
//   comm, _ := nccl.CommInitRank(nranks, id, rank)
//   defer nccl.CommDestroy(comm)
//
//   // AllReduce on a CUDA stream.
//   nccl.AllReduce(sendBuf, recvBuf, count, nccl.Float32, nccl.Sum, comm, stream)
//
// # Collective Operations
//
//   - [Comm.AllGather]: Gathers sendcount values from all GPUs and leaves identical copies of the result in each recvbuff, receiving data from rank i at offset i*sendcount.
//   - [Comm.AllReduce]: Reduces data arrays of length count in sendbuff using the op operation and leaves identical copies of the result in each recvbuff.
//   - [Comm.AlltoAll]: Each rank sends count values to all other ranks and receives count values from all other ranks.
//   - [Comm.Broadcast]: Copies count elements from sendbuff on the root rank to all ranks’ recvbuff.
//   - [Comm.Gather]: Each rank sends count elements from sendbuff to the root rank.
//   - [Comm.Reduce]: Reduce data arrays of length count in sendbuff into recvbuff on the root rank using the op operation.
//   - [Comm.ReduceScatter]: Reduce data in sendbuff from all GPUs using the op operation and leave the reduced result scattered over the devices so that the recvbuff on rank i will contain the i-th block of the result.
//   - [Comm.Scatter]: Each rank receives count elements from the root rank.
//
// # Communicator Lifecycle
//
//   - [Comm.Abort]: [Comm.Abort] frees resources that are allocated to a communicator object comm and aborts any uncompleted operations before destroying the communicator.
//   - [Comm.Destroy]: Destroy a communicator object comm.
//   - [Comm.Finalize]: Finalize a communicator object comm.
//   - [Comm.Resume]: Resume all previously suspended resources on communicator comm.
//   - [Comm.Revoke]: Revokes in-flight operations on a communicator without destroying resources.
//   - [Comm.Suspend]: Suspend communicator operations to free resources.
//
// # Communicator Queries
//
//   - [Comm.Count]: Returns in count the number of ranks in the NCCL communicator comm.
//   - [Comm.CuDevice]: Returns in device the CUDA device associated with the NCCL communicator comm.
//   - [Comm.GetAsyncError]: Queries the progress and potential errors of asynchronous NCCL operations.
//   - [Comm.GetUniqueId]: The [Comm.GetUniqueId] function generates a unique identifier for growing an existing communicator exactly once.
//   - [Comm.MemStats]: Query communicator memory statistics.
//   - [Comm.UserRank]: Returns in rank the rank of the caller in the NCCL communicator comm.
//
// # Buffer Registration
//
//   - [Comm.Deregister]: Deregister buffer represented by handle under communicator comm.
//   - [Comm.Register]: Registers the buffer buff with size under communicator comm for zero-copy communication; handle is returned for future deregistration.
//
// # Communicator Management
//
//   - [Comm.Grow]: The [Comm.Grow] function creates a new communicator by adding new ranks to an existing communicator.
//   - [Comm.Shrink]: The [Comm.Shrink] function creates a new communicator by removing specified ranks from an existing communicator.
//   - [Comm.Split]: The [Comm.Split] is a collective function and creates a set of new communicators from an existing one.
//
// # Communicator Setup
//
//   - [CommInitAll]: Creates a clique of communicators (single process version) in a blocking way.
//   - [NewCommInitRank]: Creates a new communicator (multi thread/process version).
//   - [NewCommInitRankConfig]: This function works the same way as [NewCommInitRank] but accepts a configuration argument of extra attributes for the communicator.
//   - [NewCommInitRankScalable]: This function works the same way as [NewCommInitRankConfig] but accepts a list of ncclUniqueIds instead of a single one.
//   - [GetUniqueId]: Generates an Id to be used in [NewCommInitRank].
//
// # Window Registration
//
//   - [Comm.WindowDeregister]: Deregister NCCL window represented by win under communicator comm.
//   - [Comm.WindowRegister]: Collectively register local buffer buff with size under communicator comm into NCCL window.
//
// # Error Handling
//
//   - [Comm.GetLastError]: constchar*[Comm.GetLastError]( [Comm] comm) 
//
// # Version
//
//   - [GetVersion]: The [GetVersion] function returns the version number of the currently linked NCCL library.
//
// # Group Semantics
//
//   - [GroupEnd]: End a group call.
//   - [GroupSimulateEnd]: Simulate a [GroupEnd] call and return NCCL’s simulation info in a structure passed as an argument.
//   - [GroupStart]: Start a group call.
//
// # Memory Management
//
//   - [MemAlloc]: Allocate a GPU buffer with size.
//   - [MemFree]: Free memory allocated by [MemAlloc].
//
// # One-Sided Communication
//
//   - [Window.PutSignal]: Write data from localbuff to rank peer’s registered memory window peerWin at offset peerWinOffset and subsequently updating a remote signal.
//   - [Comm.Signal]: Send a signal to rank peer without transferring data.
//   - [Comm.WaitSignal]: Wait for signals as described in the signal descriptor array.
//
// # Point-to-Point
//
//   - [Comm.Recv]: Receive data from rank peer into recvbuff.
//   - [Comm.Send]: Send data from sendbuff to rank peer.
//
// # Reduction Operations
//
//   - [Comm.RedOpCreatePreMulSum]: Creates a new reduction operator which pre-multiplies input values by a given scalar locally before reducing them with peer values via summation.
//   - [Comm.RedOpDestroy]: Destroys the reduction operator op.
//
// [NCCL Documentation]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
package nccl

// Code generated by gen-nvidia from nccl.h. DO NOT EDIT.
