// Package driver provides low-level Go bindings for the NVIDIA CUDA driver API.
//
// This package maps every function from cuda.h to a Go function using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libcuda.
// All functions return errors wrapping CUresult with descriptive messages
// from cuGetErrorName and cuGetErrorString.
//
// # Array Management
//
//   - [NewArray3D]: Creates a 3D CUDA array.
//   - [Array.Get3DDescriptor]: Get a 3D CUDA array descriptor.
//   - [NewArray]: Creates a 1D or 2D CUDA array.
//   - [Array.Destroy]: Destroys a CUDA array.
//   - [Array.GetDescriptor]: Get a 1D or 2D CUDA array descriptor.
//   - [Array.GetMemoryRequirements]: Returns the memory requirements of a CUDA array.
//   - [Array.GetPlane]: Gets a CUDA array plane from a CUDA array.
//   - [Array.GetSparseProperties]: Returns the layout properties of a sparse CUDA array.
//
// # Checkpointing
//
//   - [CheckpointProcessCheckpoint]: Checkpoint a CUDA process's GPU memory contents
//   - [CheckpointProcessGetRestoreThreadId]: Returns the restore thread ID for a CUDA process
//   - [CheckpointProcessGetState]: Returns the process state of a CUDA process
//   - [CheckpointProcessLock]: Lock a running CUDA process
//   - [CheckpointProcessRestore]: Restore a CUDA process's GPU memory contents from its last checkpoint
//   - [CheckpointProcessUnlock]: Unlock a CUDA process to allow CUDA API calls
//
// # Debugging
//
//   - [CoredumpCallbackHandle.CoredumpDeregisterCompleteCallback]: Deregister a previously registered coredump complete callback
//   - [CoredumpCallbackHandle.CoredumpDeregisterStartCallback]: Deregister a previously registered coredump start callback
//   - [CoredumpGetAttribute]: Allows caller to fetch a coredump attribute value for the current context
//   - [CoredumpGetAttributeGlobal]: Allows caller to fetch a coredump attribute value for the entire application
//   - [CoredumpRegisterCompleteCallback]: Register a callback to be invoked when a GPU coredump completes
//   - [CoredumpRegisterStartCallback]: Register a callback to be invoked when a GPU coredump begins
//   - [CoredumpSetAttribute]: Allows caller to set a coredump attribute value for the current context
//   - [CoredumpSetAttributeGlobal]: Allows caller to set a coredump attribute value globally
//
// # Context Management
//
//   - [NewContext]: Create a CUDA context.
//   - [Context.Destroy]: Destroy a CUDA context.
//   - [Context.DisablePeerAccess]: Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
//   - [Context.EnablePeerAccess]: Enables direct access to memory allocations in a peer context.
//   - [NewContextFromGreenCtx]: Converts a green context into the primary context
//   - [Context.GetApiVersion]: Gets the context's API version.
//   - [CtxGetCacheConfig]: Returns the preferred cache configuration for the current context.
//   - [NewContextGetCurrent]: Returns the CUDA context bound to the calling CPU thread.
//   - [Context.GetDevResource]: Get context resources
//   - [CtxGetDevice]: Returns the device handle for the current context.
//   - [Context.GetDeviceV2]: Returns the device handle for the specified context.
//   - [CtxGetExecAffinity]: Returns the execution affinity setting for the current context.
//   - [CtxGetFlags]: Returns the flags for the current context.
//   - [Context.GetId]: Returns the unique Id associated with the context supplied.
//   - [CtxGetLimit]: Returns resource limits.
//   - [CtxGetStreamPriorityRange]: Returns numerical values that correspond to the least and greatest stream priorities.
//   - [NewContextPopCurrent]: Pops the current CUDA context from the current CPU thread.
//   - [Context.PushCurrent]: Pushes a context on the current CPU thread.
//   - [Context.RecordEvent]: Records an event.
//   - [CtxResetPersistingL2Cache]: Resets all persisting lines in cache to normal status.
//   - [CtxSetCacheConfig]: Sets the preferred cache configuration for the current context.
//   - [Context.SetCurrent]: Binds the specified CUDA context to the calling CPU thread.
//   - [CtxSetFlags]: Sets the flags for the current context.
//   - [CtxSetLimit]: Set resource limits.
//   - [CtxSynchronize]: Block for the current context's tasks to complete.
//   - [Context.SynchronizeV2]: Block for the specified context's tasks to complete.
//   - [Context.WaitEvent]: Make a context wait on an event.
//   - [NewGreenCtx]: Creates a green context with a specified set of resources.
//   - [GreenCtx.Destroy]: Destroys a green context
//   - [GreenCtx.GetDevResource]: Get green context resources
//   - [GreenCtx.GetId]: Returns the unique Id associated with the green context supplied
//   - [GreenCtx.RecordEvent]: Records an event.
//   - [GreenCtx.StreamCreate]: Create a stream for use in the green context
//   - [GreenCtx.WaitEvent]: Make a green context wait on an event
//
// # External Resources
//
//   - [ExternalMemory.Destroy]: Destroys an external memory object.
//   - [ExternalSemaphore.Destroy]: Destroys an external semaphore.
//   - [ExternalMemory.GetMappedBuffer]: Maps a buffer onto an imported memory object.
//   - [ExternalMemory.GetMappedMipmappedArray]: Maps a CUDA mipmapped array onto an external memory object.
//   - [ImportExternalMemory]: Imports an external memory object.
//   - [ImportExternalSemaphore]: Imports an external semaphore.
//   - [Stream.SignalExternalSemaphoresAsync]: Signals a set of external semaphore objects.
//   - [Stream.WaitExternalSemaphoresAsync]: Waits on a set of external semaphore objects.
//
// # Device Management
//
//   - [DevResourceGenerateDesc]: Generate a resource descriptor
//   - [DevSmResourceSplit]: Splits a [DevResourceTypeSm] resource into structured groups.
//   - [DevSmResourceSplitByCount]: Splits [DevResourceTypeSm] resources.
//   - [Device.CanAccessPeer]: Queries if a device may directly access a peer device's memory.
//   - [NewDeviceGet]: Returns a handle to a compute device.
//   - [Device.GetAttribute]: Returns information about the device.
//   - [NewDeviceGetByPCIBusId]: Returns a handle to a compute device.
//   - [DeviceGetCount]: Returns the number of compute-capable devices.
//   - [Device.GetDefaultMemPool]: Returns the default mempool of a device.
//   - [Device.GetDevResource]: Get device resources
//   - [Device.GetExecAffinitySupport]: Returns information about the execution affinity support of the device.
//   - [Device.GetGraphMemAttribute]: Query asynchronous allocation attributes related to graphs.
//   - [Device.GetHostAtomicCapabilities]: Queries details about atomic operations supported between the device and host.
//   - [Device.GetLuid]: Return an LUID and device node mask for the device.
//   - [Device.GetMemPool]: Gets the current mempool for a device.
//   - [Device.GetName]: Returns an identifier string for the device.
//   - [Device.GetNvSciSyncAttributes]: Return NvSciSync attributes that this device can support.
//   - [Device.GetP2PAtomicCapabilities]: Queries details about atomic operations supported between two devices.
//   - [Device.GetP2PAttribute]: Queries attributes of the link between two devices.
//   - [Device.GetPCIBusId]: Returns a PCI Bus Id string for the device.
//   - [Device.GetTexture1DLinearMaxWidth]: Returns the maximum number of elements allocatable in a 1D linear texture for a given texture element size.
//   - [Device.GetUuid]: Return an UUID for the device.
//   - [Device.GraphMemTrim]: Free unused memory that was cached on the specified device for use with graphs back to the OS.
//   - [Device.PrimaryCtxGetState]: Get the state of the primary context.
//   - [Device.PrimaryCtxRelease]: Release the primary context on the GPU.
//   - [Device.PrimaryCtxReset]: Destroy all allocations and reset all state on the primary context.
//   - [Device.PrimaryCtxRetain]: Retain the primary context on the GPU.
//   - [Device.PrimaryCtxSetFlags]: Set flags for the primary context.
//   - [Device.RegisterAsyncNotification]: Registers a callback function to receive async notifications.
//   - [Device.SetGraphMemAttribute]: Set asynchronous allocation attributes related to graphs.
//   - [Device.SetMemPool]: Sets the current memory pool of a device.
//   - [Device.TotalMem]: Returns the total amount of memory on the device.
//   - [Device.UnregisterAsyncNotification]: Unregisters an async notification callback.
//
// # Driver Queries
//
//   - [DriverGetVersion]: Returns the latest CUDA version supported by driver.
//   - [GetProcAddress]: Returns the requested driver API function pointer.
//
// # Event Management
//
//   - [NewEvent]: Creates an event.
//   - [Event.Destroy]: Destroys an event.
//   - [Event.ElapsedTime]: Computes the elapsed time between two events.
//   - [Event.Query]: Queries an event's status.
//   - [Event.Record]: Records an event.
//   - [Event.RecordWithFlags]: Records an event.
//   - [Event.Synchronize]: Waits for an event to complete.
//
// # RDMA
//
//   - [FlushGPUDirectRDMAWrites]: Blocks until remote writes are visible to the specified scope.
//
// # Function Management
//
//   - [Function.GetAttribute]: Returns information about a function.
//   - [Function.GetModule]: Returns a module handle.
//   - [Function.GetName]: Returns the function name for a [Function] handle.
//   - [Function.GetParamCount]: Returns the number of parameters used by the function.
//   - [Function.GetParamInfo]: Returns the offset and size of a kernel parameter in the device-side parameter layout.
//   - [Function.IsLoaded]: Returns if the function is loaded.
//   - [Function.Load]: Loads a function.
//   - [Function.SetAttribute]: Sets information about a function.
//   - [Function.SetCacheConfig]: Sets the preferred cache configuration for a device function.
//
// # Graph Management
//
//   - [Graph.AddBatchMemOpNode]: Creates a batch memory operation node and adds it to a graph.
//   - [Graph.AddChildGraphNode]: Creates a child graph node and adds it to a graph.
//   - [Graph.AddDependencies]: Adds dependency edges to a graph.
//   - [Graph.AddEmptyNode]: Creates an empty node and adds it to a graph.
//   - [Graph.AddEventRecordNode]: Creates an event record node and adds it to a graph.
//   - [Graph.AddEventWaitNode]: Creates an event wait node and adds it to a graph.
//   - [Graph.AddExternalSemaphoresSignalNode]: Creates an external semaphore signal node and adds it to a graph.
//   - [Graph.AddExternalSemaphoresWaitNode]: Creates an external semaphore wait node and adds it to a graph.
//   - [Graph.AddHostNode]: Creates a host execution node and adds it to a graph.
//   - [Graph.AddKernelNode]: Creates a kernel execution node and adds it to a graph.
//   - [Graph.AddMemAllocNode]: Creates an allocation node and adds it to a graph.
//   - [Graph.AddMemFreeNode]: Creates a memory free node and adds it to a graph.
//   - [Graph.AddMemcpyNode]: Creates a memcpy node and adds it to a graph.
//   - [Graph.AddMemsetNode]: Creates a memset node and adds it to a graph.
//   - [Graph.AddNode]: Adds a node of arbitrary type to a graph.
//   - [GraphNode.GraphBatchMemOpNodeGetParams]: Returns a batch mem op node's parameters.
//   - [GraphNode.GraphBatchMemOpNodeSetParams]: Sets a batch mem op node's parameters.
//   - [NewGraphChildGraphNodeGetGraph]: Gets a handle to the embedded graph of a child graph node.
//   - [Graph.Clone]: Clones a graph.
//   - [Graph.ConditionalHandleCreate]: Create a conditional handle.
//   - [NewGraph]: Creates a graph.
//   - [Graph.DebugDotPrint]: Write a DOT file describing graph structure.
//   - [Graph.Destroy]: Destroys a graph.
//   - [GraphNode.GraphDestroyNode]: Remove a node from the graph.
//   - [GraphNode.GraphEventRecordNodeGetEvent]: Returns the event associated with an event record node.
//   - [GraphNode.GraphEventRecordNodeSetEvent]: Sets an event record node's event.
//   - [GraphNode.GraphEventWaitNodeGetEvent]: Returns the event associated with an event wait node.
//   - [GraphNode.GraphEventWaitNodeSetEvent]: Sets an event wait node's event.
//   - [GraphExec.BatchMemOpNodeSetParams]: Sets the parameters for a batch mem op node in the given graphExec.
//   - [GraphExec.ChildGraphNodeSetParams]: Updates node parameters in the child graph node in the given graphExec.
//   - [GraphExec.Destroy]: Destroys an executable graph.
//   - [GraphExec.EventRecordNodeSetEvent]: Sets the event for an event record node in the given graphExec.
//   - [GraphExec.EventWaitNodeSetEvent]: Sets the event for an event wait node in the given graphExec.
//   - [GraphExec.ExternalSemaphoresSignalNodeSetParams]: Sets the parameters for an external semaphore signal node in the given graphExec.
//   - [GraphExec.ExternalSemaphoresWaitNodeSetParams]: Sets the parameters for an external semaphore wait node in the given graphExec.
//   - [GraphExec.GetFlags]: Query the instantiation flags of an executable graph.
//   - [GraphExec.GetId]: Returns the id of a given graph exec.
//   - [GraphExec.HostNodeSetParams]: Sets the parameters for a host node in the given graphExec.
//   - [GraphExec.KernelNodeSetParams]: Sets the parameters for a kernel node in the given graphExec.
//   - [GraphExec.MemcpyNodeSetParams]: Sets the parameters for a memcpy node in the given graphExec.
//   - [GraphExec.MemsetNodeSetParams]: Sets the parameters for a memset node in the given graphExec.
//   - [GraphExec.NodeSetParams]: Update a graph node's parameters in an instantiated graph.
//   - [GraphExec.Update]: Check whether an executable graph can be updated with a graph and perform the update if possible.
//   - [GraphNode.GraphExternalSemaphoresSignalNodeGetParams]: Returns an external semaphore signal node's parameters.
//   - [GraphNode.GraphExternalSemaphoresSignalNodeSetParams]: Sets an external semaphore signal node's parameters.
//   - [GraphNode.GraphExternalSemaphoresWaitNodeGetParams]: Returns an external semaphore wait node's parameters.
//   - [GraphNode.GraphExternalSemaphoresWaitNodeSetParams]: Sets an external semaphore wait node's parameters.
//   - [Graph.GetEdges]: Returns a graph's dependency edges.
//   - [Graph.GetId]: Returns the id of a given graph.
//   - [Graph.GetNodes]: Returns a graph's nodes.
//   - [Graph.GetRootNodes]: Returns a graph's root nodes.
//   - [GraphNode.GraphHostNodeGetParams]: Returns a host node's parameters.
//   - [GraphNode.GraphHostNodeSetParams]: Sets a host node's parameters.
//   - [Graph.Instantiate]: Creates an executable graph from a graph.
//   - [Graph.InstantiateWithParams]: Creates an executable graph from a graph.
//   - [GraphNode.GraphKernelNodeCopyAttributes]: Copies attributes from source node to destination node.
//   - [GraphNode.GraphKernelNodeGetAttribute]: Queries node attribute.
//   - [GraphNode.GraphKernelNodeGetParams]: Returns a kernel node's parameters.
//   - [GraphNode.GraphKernelNodeSetAttribute]: Sets node attribute.
//   - [GraphNode.GraphKernelNodeSetParams]: Sets a kernel node's parameters.
//   - [GraphExec.GraphLaunch]: Launches an executable graph in a stream.
//   - [GraphNode.GraphMemAllocNodeGetParams]: Returns a memory alloc node's parameters.
//   - [GraphNode.GraphMemFreeNodeGetParams]: Returns a memory free node's parameters.
//   - [GraphNode.GraphMemcpyNodeGetParams]: Returns a memcpy node's parameters.
//   - [GraphNode.GraphMemcpyNodeSetParams]: Sets a memcpy node's parameters.
//   - [GraphNode.GraphMemsetNodeGetParams]: Returns a memset node's parameters.
//   - [GraphNode.GraphMemsetNodeSetParams]: Sets a memset node's parameters.
//   - [GraphNode.FindInClone]: Finds a cloned version of a node.
//   - [GraphNode.GetContainingGraph]: Returns the graph that contains a given graph node.
//   - [GraphNode.GetDependencies]: Returns a node's dependencies.
//   - [GraphNode.GetDependentNodes]: Returns a node's dependent nodes.
//   - [GraphNode.GetEnabled]: Query whether a node in the given graphExec is enabled.
//   - [GraphNode.GetLocalId]: Returns the local node id of a given graph node.
//   - [GraphNode.GetParams]: Return a graph node's parameters.
//   - [GraphNode.GetToolsId]: Returns an id used by tools to identify a given node.
//   - [GraphNode.GetType]: Returns a node's type.
//   - [GraphNode.SetEnabled]: Enables or disables the specified node in the given graphExec.
//   - [GraphNode.SetParams]: Update a graph node's parameters.
//   - [Graph.ReleaseUserObject]: Release a user object reference from a graph.
//   - [Graph.RemoveDependencies]: Removes dependency edges from a graph.
//   - [Graph.RetainUserObject]: Retain a reference to a user object from a graph.
//   - [GraphExec.GraphUpload]: Uploads an executable graph in a stream.
//   - [Stream.GraphicsMapResources]: Map graphics resources for access by CUDA
//   - [GraphicsResource.GetMappedMipmappedArray]: Get a mipmapped array through which to access a mapped graphics resource.
//   - [GraphicsResource.GetMappedPointer]: Get a device pointer through which to access a mapped graphics resource.
//   - [GraphicsResource.SetMapFlags]: Set usage flags for mapping a graphics resource
//   - [GraphicsResource.GraphicsSubResourceGetMappedArray]: Get an array through which to access a subresource of a mapped graphics resource.
//   - [Stream.GraphicsUnmapResources]: Unmap graphics resources.
//   - [GraphicsResource.GraphicsUnregisterResource]: Unregisters a graphics resource for access by CUDA
//
// # Initialization
//
//   - [Init]: Initialize the CUDA driver API Initializes the driver API and must be called before any other function from the driver API in the current process.
//
// # IPC
//
//   - [DevicePtr.IpcCloseMemHandle]: Attempts to close memory mapped with [IpcOpenMemHandle].
//   - [Event.IpcGetEventHandle]: Gets an interprocess handle for a previously allocated event.
//   - [DevicePtr.IpcGetMemHandle]: Gets an interprocess memory handle for an existing device memory allocation.
//   - [IpcOpenEventHandle]: Opens an interprocess event handle for use in the current process.
//   - [IpcOpenMemHandle]: Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
//
// # Kernel Management
//
//   - [Kernel.GetAttribute]: Returns information about a kernel
//   - [Kernel.GetFunction]: Returns a function handle
//   - [Kernel.GetLibrary]: Returns a library handle
//   - [Kernel.GetName]: Returns the function name for a [Kernel] handle
//   - [Kernel.GetParamCount]: Returns the number of parameters used by the kernel
//   - [Kernel.GetParamInfo]: Returns the offset and size of a kernel parameter in the device-side parameter layout
//   - [Kernel.SetAttribute]: Sets information about a kernel
//   - [Kernel.SetCacheConfig]: Sets the preferred cache configuration for a device kernel.
//
// # Kernel Execution
//
//   - [Function.LaunchCooperativeKernel]: Launches a CUDA function [Function] or a CUDA kernel [Kernel] where thread blocks can cooperate and synchronize as they execute.
//   - [Stream.LaunchHostFunc]: Enqueues a host function call in a stream.
//   - [Function.LaunchKernel]: Launches a CUDA function [Function] or a CUDA kernel [Kernel].
//   - [Function.LaunchKernelEx]: Launches a CUDA function [Function] or a CUDA kernel [Kernel] with launch-time configuration.
//
// # Library Management
//
//   - [Library.EnumerateKernels]: Retrieve the kernel handles within a library.
//   - [Library.GetGlobal]: Returns a global device pointer
//   - [Library.GetKernel]: Returns a kernel handle
//   - [Library.GetKernelCount]: Returns the number of kernels within a library
//   - [Library.GetManaged]: Returns a pointer to managed memory
//   - [Library.GetModule]: Returns a module handle
//   - [Library.GetUnifiedFunction]: Returns a pointer to a unified function
//   - [NewLibraryLoadData]: Load a library with specified code and options
//   - [NewLibraryLoadFromFile]: Load a library with specified file and options
//   - [Library.Unload]: Unloads a library
//
// # Linker
//
//   - [LinkState.AddData]: Add an input to a pending linker invocation.
//   - [LinkState.AddFile]: Add a file input to a pending linker invocation.
//   - [LinkState.Complete]: Complete a pending linker invocation.
//   - [LinkCreate]: Creates a pending JIT linker invocation.
//   - [LinkState.Destroy]: Destroys state for a JIT linker invocation.
//
// # Virtual Memory Management
//
//   - [DevicePtr.MemAddressFree]: Free an address range reservation.
//   - [DevicePtr.MemAddressReserve]: Allocate an address range reservation.
//   - [MemCreate]: Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.
//   - [MemExportToShareableHandle]: Exports an allocation to a requested shareable handle type.
//   - [DevicePtr.MemGetAccess]: Get the access flags set for the given location and ptr.
//   - [MemGetAllocationGranularity]: Calculates either the minimal or recommended granularity.
//   - [MemGetAllocationPropertiesFromHandle]: Retrieve the contents of the property structure defining properties for this handle.
//   - [MemImportFromShareableHandle]: Imports an allocation from a requested shareable handle type.
//   - [DevicePtr.MemMap]: Maps an allocation handle to a reserved virtual address range.
//   - [Stream.MemMapArrayAsync]: Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
//   - [MemRelease]: Release a memory handle representing a memory allocation which was previously allocated through [MemCreate].
//   - [MemRetainAllocationHandle]: Given an address addr, returns the allocation handle of the backing memory allocation.
//   - [DevicePtr.MemSetAccess]: Set the access flags for each location specified in desc for the given virtual address range.
//   - [DevicePtr.MemUnmap]: Unmap the backing memory of a given address range.
//
// # Memory Management
//
//   - [DevicePtr.MemAdvise]: Advise about the usage of a given memory range.
//   - [MemAlloc]: Allocates device memory.
//   - [Stream.MemAllocAsync]: Allocates memory with stream ordered semantics.
//   - [MemoryPool.MemAllocFromPoolAsync]: Allocates memory from a specified pool with stream ordered semantics.
//   - [MemAllocHost]: Allocates page-locked host memory.
//   - [MemAllocManaged]: Allocates memory that will be automatically managed by the Unified Memory system.
//   - [MemAllocPitch]: Allocates pitched device memory.
//   - [Stream.MemBatchDecompressAsync]: Submit a batch of count independent decompression operations.
//   - [Stream.MemDiscardAndPrefetchBatchAsync]: Performs a batch of memory discards and prefetches asynchronously.
//   - [Stream.MemDiscardBatchAsync]: Performs a batch of memory discards asynchronously.
//   - [DevicePtr.MemFree]: Frees device memory.
//   - [DevicePtr.MemFreeAsync]: Frees memory with stream ordered semantics.
//   - [MemFreeHost]: Frees page-locked host memory.
//   - [DevicePtr.MemGetAddressRange]: Get information on memory allocations.
//   - [MemGetDefaultMemPool]: Returns the default memory pool for a given location and allocation type.
//   - [DevicePtr.MemGetHandleForAddressRange]: Retrieve handle for an address range.
//   - [MemGetInfo]: Gets free and total memory.
//   - [MemGetMemPool]: Gets the current memory pool for a memory location and of a particular allocation type.
//   - [MemHostAlloc]: Allocates page-locked host memory.
//   - [MemHostGetDevicePointer]: Passes back device pointer of mapped pinned memory.
//   - [MemHostGetFlags]: Passes back flags that were used for a pinned allocation.
//   - [MemHostRegister]: Registers an existing host memory range for use by CUDA.
//   - [MemHostUnregister]: Unregisters a memory range that was registered with [MemHostRegister].
//   - [DevicePtr.MemPrefetchAsync]: Prefetches memory to the specified destination location.
//   - [Stream.MemPrefetchBatchAsync]: Performs a batch of memory prefetches asynchronously.
//   - [DevicePtr.MemRangeGetAttribute]: Query an attribute of a given memory range.
//   - [DevicePtr.MemRangeGetAttributes]: Query attributes of a given memory range.
//   - [MemoryPool.MemSetMemPool]: Sets the current memory pool for a memory location and allocation type.
//   - [DevicePtr.Memcpy]: Copies memory.
//   - [Memcpy2D]: Copies memory for 2D arrays.
//   - [Stream.Memcpy2DAsync]: Copies memory for 2D arrays.
//   - [Memcpy2DUnaligned]: Copies memory for 2D arrays.
//   - [Memcpy3D]: Copies memory for 3D arrays.
//   - [Stream.Memcpy3DAsync]: Copies memory for 3D arrays.
//   - [Stream.Memcpy3DBatchAsync]: Performs a batch of 3D memory copies asynchronously.
//   - [Memcpy3DPeer]: Copies memory between contexts.
//   - [Stream.Memcpy3DPeerAsync]: Copies memory between contexts asynchronously.
//   - [Stream.Memcpy3DWithAttributesAsync]: Performs 3D memory copy with attributes asynchronously Performs the copy operation specified in op.
//   - [DevicePtr.MemcpyAsync]: Copies memory asynchronously.
//   - [Array.MemcpyAtoA]: Copies memory from Array to Array.
//   - [DevicePtr.MemcpyAtoD]: Copies memory from Array to Device.
//   - [Array.MemcpyAtoH]: Copies memory from Array to Host.
//   - [Array.MemcpyAtoHAsync]: Copies memory from Array to Host.
//   - [Stream.MemcpyBatchAsync]: Performs a batch of memory copies asynchronously.
//   - [Array.MemcpyDtoA]: Copies memory from Device to Array.
//   - [DevicePtr.MemcpyDtoD]: Copies memory from Device to Device.
//   - [DevicePtr.MemcpyDtoDAsync]: Copies memory from Device to Device.
//   - [DevicePtr.MemcpyDtoH]: Copies memory from Device to Host.
//   - [DevicePtr.MemcpyDtoHAsync]: Copies memory from Device to Host.
//   - [Array.MemcpyHtoA]: Copies memory from Host to Array.
//   - [Array.MemcpyHtoAAsync]: Copies memory from Host to Array.
//   - [DevicePtr.MemcpyHtoD]: Copies memory from Host to Device.
//   - [DevicePtr.MemcpyHtoDAsync]: Copies memory from Host to Device.
//   - [DevicePtr.MemcpyPeer]: Copies device memory between two contexts.
//   - [DevicePtr.MemcpyPeerAsync]: Copies device memory between two contexts asynchronously.
//   - [DevicePtr.MemcpyWithAttributesAsync]: Performs asynchronous memory copy operation with the specified attributes.
//   - [DevicePtr.MemsetD16]: Initializes device memory.
//   - [DevicePtr.MemsetD16Async]: Sets device memory.
//   - [DevicePtr.MemsetD2D16]: Initializes device memory.
//   - [DevicePtr.MemsetD2D16Async]: Sets device memory.
//   - [DevicePtr.MemsetD2D32]: Initializes device memory.
//   - [DevicePtr.MemsetD2D32Async]: Sets device memory.
//   - [DevicePtr.MemsetD2D8]: Initializes device memory.
//   - [DevicePtr.MemsetD2D8Async]: Sets device memory.
//   - [DevicePtr.MemsetD32]: Initializes device memory.
//   - [DevicePtr.MemsetD32Async]: Sets device memory.
//   - [DevicePtr.MemsetD8]: Initializes device memory.
//   - [DevicePtr.MemsetD8Async]: Sets device memory.
//   - [DevicePtr.PointerGetAttribute]: Returns information about a pointer.
//   - [DevicePtr.PointerGetAttributes]: Returns information about a pointer.
//   - [DevicePtr.PointerSetAttribute]: Set attributes on a previously allocated memory region.
//
// # Memory Pools
//
//   - [NewMemoryPool]: Creates a memory pool.
//   - [MemoryPool.Destroy]: Destroys the specified memory pool.
//   - [DevicePtr.MemPoolExportPointer]: Export data to share a memory pool allocation between processes.
//   - [MemoryPool.ExportToShareableHandle]: Exports a memory pool to the requested handle type.
//   - [MemoryPool.GetAccess]: Returns the accessibility of a pool from a device.
//   - [MemoryPool.GetAttribute]: Gets attributes of a memory pool.
//   - [NewMemoryPoolImportFromShareableHandle]: imports a memory pool from a shared handle.
//   - [MemoryPool.ImportPointer]: Import a memory pool allocation from another process.
//   - [MemoryPool.SetAccess]: Controls visibility of pools between devices.
//   - [MemoryPool.SetAttribute]: Sets attributes of a memory pool.
//   - [MemoryPool.TrimTo]: Tries to release memory back to the OS.
//
// # Texture Management
//
//   - [NewMipmappedArray]: Creates a CUDA mipmapped array.
//   - [MipmappedArray.Destroy]: Destroys a CUDA mipmapped array.
//   - [MipmappedArray.GetLevel]: Gets a mipmap level of a CUDA mipmapped array.
//   - [MipmappedArray.GetMemoryRequirements]: Returns the memory requirements of a CUDA mipmapped array.
//   - [MipmappedArray.GetSparseProperties]: Returns the layout properties of a sparse CUDA mipmapped array.
//   - [TexObjectCreate]: Creates a texture object.
//   - [TexObjectDestroy]: Destroys a texture object.
//   - [TexObjectGetResourceDesc]: Returns a texture object's resource descriptor.
//   - [TexObjectGetResourceViewDesc]: Returns a texture object's resource view descriptor.
//   - [TexObjectGetTextureDesc]: Returns a texture object's texture descriptor.
//
// # Module Management
//
//   - [Module.EnumerateFunctions]: Returns the function handles within a module.
//   - [Module.GetFunction]: Returns a function handle.
//   - [Module.GetFunctionCount]: Returns the number of functions within a module.
//   - [Module.GetGlobal]: Returns a global pointer from a module.
//   - [ModuleGetLoadingMode]: Query lazy loading mode.
//   - [NewModuleLoad]: Loads a compute module.
//   - [NewModuleLoadData]: Load a module's data.
//   - [NewModuleLoadDataEx]: Load a module's data with options.
//   - [NewModuleLoadFatBinary]: Load a module's data.
//   - [Module.Unload]: Unloads a module.
//
// # Multicast
//
//   - [Device.MulticastAdd]: Associate a device to a multicast object.
//   - [DevicePtr.MulticastBindAddr]: Bind a memory allocation represented by a virtual address to a multicast object.
//   - [Device.MulticastBindAddrV2]: Bind a memory allocation represented by a virtual address to a multicast object.
//   - [MulticastBindMem]: Bind a memory allocation represented by a handle to a multicast object.
//   - [Device.MulticastBindMemV2]: Bind a memory allocation represented by a handle to a multicast object.
//   - [MulticastCreate]: Create a generic allocation handle representing a multicast object described by the given properties.
//   - [MulticastGetGranularity]: Calculates either the minimal or recommended granularity for multicast object.
//   - [Device.MulticastUnbind]: Unbind any memory allocations bound to a multicast object at a given offset and upto a given size.
//
// # Occupancy
//
//   - [Function.AvailableDynamicSMemPerBlock]: Returns dynamic shared memory available per block when launching numBlocks blocks on SM.
//   - [Function.MaxActiveBlocksPerMultiprocessor]: Returns occupancy of a function.
//   - [Function.MaxActiveBlocksPerMultiprocessorWithFlags]: Returns occupancy of a function.
//   - [Function.MaxActiveClusters]: Given the kernel function ( func) and launch configuration ( config), return the maximum number of clusters that could co-exist on the target device in *numClusters.
//   - [Function.MaxPotentialBlockSize]: Suggest a launch configuration with reasonable occupancy.
//   - [Function.MaxPotentialBlockSizeWithFlags]: Suggest a launch configuration with reasonable occupancy.
//   - [Function.MaxPotentialClusterSize]: Given the kernel function ( func) and launch configuration ( config), return the maximum cluster size in *clusterSize.
//
// # Stream Management
//
//   - [Stream.AddCallback]: Add a callback to a compute stream.
//   - [Stream.AttachMemAsync]: Attach memory to a stream asynchronously.
//   - [Stream.BeginCapture]: Begins graph capture on a stream.
//   - [Stream.BeginCaptureToCig]: Begins capture to CIG on a stream.
//   - [Stream.BeginCaptureToGraph]: Begins graph capture on a stream to an existing graph.
//   - [Stream.CopyAttributes]: Copies attributes from source stream to destination stream.
//   - [NewStream]: Create a stream.
//   - [NewStreamCreateWithPriority]: Create a stream with the given priority.
//   - [Stream.Destroy]: Destroys a stream.
//   - [Stream.EndCapture]: Ends capture on a stream, returning the captured graph.
//   - [Stream.EndCaptureToCig]: Ends CIG capture on a stream.
//   - [Stream.GetAttribute]: Queries stream attribute.
//   - [Stream.GetCaptureInfo]: Query a stream's capture state.
//   - [Stream.GetCtx]: Query the contexts associated with a stream.
//   - [Stream.GetDevResource]: Get stream resources
//   - [Stream.GetDevice]: Returns the device handle of the stream.
//   - [Stream.GetFlags]: Query the flags of a given stream.
//   - [Stream.GetGreenCtx]: Query the green context associated with a stream
//   - [Stream.GetId]: Returns the unique Id associated with the stream handle supplied.
//   - [Stream.GetPriority]: Query the priority of a given stream.
//   - [Stream.IsCapturing]: Returns a stream's capture status.
//   - [Stream.Query]: Determine status of a compute stream.
//   - [Stream.SetAttribute]: Sets stream attribute.
//   - [Stream.Synchronize]: Wait until a stream's tasks are completed.
//   - [Stream.UpdateCaptureDependencies]: Update the set of dependencies in a capturing stream.
//   - [Stream.WaitEvent]: Make a compute stream wait on an event.
//   - [Stream.WaitValue32]: Wait on a memory location
//   - [Stream.WaitValue64]: Wait on a memory location
//   - [Stream.WriteValue32]: Write a value to memory
//   - [Stream.WriteValue64]: Write a value to memory
//
// # Stream Ordered Memory
//
//   - [Stream.BatchMemOp]: Batch operations to synchronize the stream via memory operations
//
// # Surface Management
//
//   - [SurfObjectCreate]: Creates a surface object.
//   - [SurfObjectDestroy]: Destroys a surface object.
//   - [SurfObjectGetResourceDesc]: Returns a surface object's resource descriptor.
//
// # Tensor Core
//
//   - [TensorMapEncodeIm2col]: Create a tensor map descriptor object representing im2col memory region
//   - [TensorMapEncodeIm2colWide]: Create a tensor map descriptor object representing im2col memory region, but where the elements are exclusively loaded along the W dimension.
//   - [TensorMapEncodeTiled]: Create a tensor map descriptor object representing tiled memory region
//   - [TensorMapReplaceAddress]: Modify an existing tensor map descriptor with an updated global address
//
// # User Objects
//
//   - [NewUserObject]: Create a user object.
//   - [UserObject.Release]: Release a reference to a user object.
//   - [UserObject.Retain]: Retain a reference to a user object.
//
// # Other
//
//   - [LogsCurrent]: Sets log iterator to point to the end of log buffer, where the next message would be written.
//   - [LogsDumpToFile]: Dump accumulated driver logs into a file
//   - [LogsDumpToMemory]: Dump accumulated driver logs into a buffer
//   - [LogsRegisterCallback]: Register a callback function to receive error log messages
//   - [LogsCallbackHandle.LogsUnregisterCallback]: Unregister a log message callback
//   - [ThreadExchangeStreamCaptureMode]: Swaps the stream capture interaction mode for a thread.
//
// [CUDA Driver API Documentation]: https://docs.nvidia.com/cuda/cuda-driver-api/
package driver

// Code generated by gen-nvidia from cuda.h. DO NOT EDIT.
