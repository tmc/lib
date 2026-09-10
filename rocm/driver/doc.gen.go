// Package driver provides Go bindings for the HIP runtime API.
//
// HIP (Heterogeneous-Compute Interface for Portability) is AMD's GPU programming
// interface, providing a CUDA-like API for AMD GPUs. These bindings use purego for
// dynamic loading without cgo.
//
// # Other
//
//   - [ApiName]: Returns HIP API name by ID.
//   - [NewArray3D]: Create a 3D array memory pointer on the device.
//   - [Array.Get3DDescriptor]: Gets a 3D array descriptor
//   - [NewArray]: Create an array memory pointer on the device.
//   - [Array.Destroy]: Destroy an array memory pointer on the device.
//   - [Array.GetDescriptor]: Gets a 1D or 2D array descriptor
//   - [Array.GetInfo]: Gets info about the specified array
//   - [ChooseDevice]: Device which matches [DeviceProp] is returned
//   - [Stream.ConfigureCall]: Configure a kernel launch.
//   - [NewSurfaceObject]: Create a surface object.
//   - [NewTextureObject]: Creates a texture object.
//   - [ExternalMemory.Destroy]: Destroys an external memory object.
//   - [ExternalSemaphore.Destroy]: Destroys an external semaphore object and releases any references to the underlying resource.
//   - [SurfaceObject.Destroy]: Destroy a surface object.
//   - [TextureObject.Destroy]: Destroys a texture object.
//   - [DriverGetVersion]: Returns the approximate HIP driver version.
//   - [DrvGetErrorName]: Return hip error as text string form.
//   - [DrvGetErrorString]: Return handy text string message to explain the error which occurred
//   - [Graph.AddMemFreeNode]: Creates a memory free node and adds it to a graph
//   - [Graph.AddMemcpyNode]: Creates a memcpy node and adds it to a graph.
//   - [Graph.AddMemsetNode]: Creates a memset node and adds it to a graph.
//   - [GraphExec.MemcpyNodeSetParams]: Sets the parameters for a memcpy node in the given graphExec.
//   - [GraphExec.MemsetNodeSetParams]: Sets the parameters for a memset node in the given graphExec.
//   - [GraphNode.MemcpyNodeGetParams]: Gets a memcpy node's parameters.
//   - [GraphNode.MemcpyNodeSetParams]: Sets a memcpy node's parameters.
//   - [DrvMemcpy3D]: Copies data between host and device.
//   - [Stream.Memcpy3DAsync]: Copies data between host and device asynchronously.
//   - [DevicePtr.PointerGetAttributes]: Returns information about the specified pointer.[BETA]
//   - [ExtGetLastError]: Return last error returned by any HIP runtime API call and resets the stored error code to...
//   - [ExtGetLinkTypeAndHopCount]: Returns the link type and hop count between two devices
//   - [Stream.LaunchKernel]: Launches kernel from the pointer address, with arguments and shared memory on stream.
//   - [ExtLaunchMultiKernelMultiDevice]: Launches kernels on multiple devices and guarantees all specified kernels are dispatched on...
//   - [ExtMallocWithFlags]: Allocate memory on the default accelerator
//   - [NewStreamWithCUMask]: Create an asynchronous stream with the specified CU mask.
//   - [Stream.GetCUMask]: Get CU mask associated with an asynchronous stream
//   - [ExternalMemory.GetMappedBuffer]: Maps a buffer onto an imported memory object.
//   - [ExternalMemory.GetMappedMipmappedArray]: Maps a mipmapped array onto an external memory object.
//   - [Free]: [Free] memory allocated by the hcc hip memory allocation API.
//   - [Array.Free]: Frees an array on the device.
//   - [Stream.FreeAsync]: Frees memory with stream ordered semantics
//   - [FreeHost]: Frees page-locked memory This API performs an implicit [DeviceSynchronize] call.
//   - [MipmappedArray.Free]: Frees a mipmapped array on the device.
//   - [Function.GetAttribute]: Find out a specific attribute for a given function.
//   - [FuncGetAttributes]: Find out attributes for a given function.
//   - [FuncSetAttribute]: Set attribute for a specific function
//   - [FuncSetCacheConfig]: Set Cache configuration for a specific function
//   - [FuncSetSharedMemConfig]: Set shared memory configuation for a specific function
//   - [ArrayConst.GetChannelDesc]: Gets the channel descriptor in an array.
//   - [GetDevice]: Return the default device id for the calling host thread.
//   - [GetDeviceCount]: Return number of compute-capable devices.
//   - [GetDeviceFlags]: Gets the flags set for current device
//   - [GetDeviceProperties]: Returns device properties.
//   - [Function.GetFuncBySymbol]: Gets pointer to device entry function that matches entry function symbolPtr.
//   - [GetLastError]: Return last error returned by any HIP runtime API call and resets the stored error code to...
//   - [Array.GetMipmappedArrayLevel]: Gets a mipmap level of a HIP mipmapped array.
//   - [GetProcAddress]: Gets the pointer of requested HIP driver function.
//   - [GetSymbolAddress]: Gets device pointer associated with symbol on the device.
//   - [GetSymbolSize]: Gets the size of the given symbol on the device.
//   - [TextureObject.GetTextureObjectResourceDesc]: Gets resource descriptor for the texture object.
//   - [TextureObject.GetTextureObjectResourceViewDesc]: Gets resource view descriptor for the texture object.
//   - [TextureObject.GetTextureObjectTextureDesc]: Gets texture descriptor for the texture object.
//   - [Graph.AddChildGraphNode]: Creates a child graph node and adds it to a graph.
//   - [Graph.AddDependencies]: Adds dependency edges to a graph.
//   - [Graph.AddEmptyNode]: Creates an empty node and adds it to a graph.
//   - [Graph.AddEventRecordNode]: Creates an event record node and adds it to a graph.
//   - [Graph.AddEventWaitNode]: Creates an event wait node and adds it to a graph.
//   - [Graph.AddExternalSemaphoresSignalNode]: Creates a external semaphor signal node and adds it to a graph.
//   - [Graph.AddExternalSemaphoresWaitNode]: Creates a external semaphor wait node and adds it to a graph.
//   - [Graph.AddHostNode]: Creates a host execution node and adds it to a graph.
//   - [Graph.AddKernelNode]: Creates a kernel execution node and adds it to a graph.
//   - [Graph.AddMemAllocNode]: Creates a memory allocation node and adds it to a graph
//   - [Graph.AddMemcpyNode1D]: Creates a 1D memcpy node and adds it to a graph.
//   - [Graph.AddMemcpyNodeFromSymbol]: Creates a memcpy node to copy from a symbol on the device and adds it to a graph.
//   - [Graph.AddMemcpyNodeToSymbol]: Creates a memcpy node to copy to a symbol on the device and adds it to a graph.
//   - [Graph.AddNode]: Creates a kernel execution node and adds it to a graph.
//   - [GraphNode.GetGraph]: Gets a handle to the embedded graph of a child graph node.
//   - [Graph.Clone]: Clones a graph.
//   - [NewGraph]: Creates a graph
//   - [Graph.DebugDotPrint]: Write a DOT file describing graph structure.
//   - [Graph.Destroy]: Destroys a graph
//   - [GraphNode.DestroyNode]: Remove a node from the graph.
//   - [GraphNode.RecordNodeGetEvent]: Returns the event associated with an event record node.
//   - [GraphNode.RecordNodeSetEvent]: Sets an event record node's event.
//   - [GraphNode.WaitNodeGetEvent]: Returns the event associated with an event wait node.
//   - [GraphNode.WaitNodeSetEvent]: Sets an event wait node's event.
//   - [GraphExec.ChildGraphNodeSetParams]: Updates node parameters in the child graph node in the given graphExec.
//   - [GraphExec.Destroy]: Destroys an executable graph
//   - [GraphExec.EventRecordNodeSetEvent]: Sets the event for an event record node in the given graphExec.
//   - [GraphExec.EventWaitNodeSetEvent]: Sets the event for an event record node in the given graphExec.
//   - [GraphExec.ExternalSemaphoresSignalNodeSetParams]: Updates node parameters in the external semaphore signal node in the given graphExec.
//   - [GraphExec.ExternalSemaphoresWaitNodeSetParams]: Updates node parameters in the external semaphore wait node in the given graphExec.
//   - [GraphExec.GetFlags]: Return the flags on executable graph.
//   - [GraphExec.HostNodeSetParams]: Sets the parameters for a host node in the given graphExec.
//   - [GraphExec.KernelNodeSetParams]: Sets the parameters for a kernel node in the given graphExec.
//   - [GraphExec.MemcpyNodeSetParams1D]: Sets the parameters for a memcpy node in the given graphExec to perform a 1-dimensional copy.
//   - [GraphExec.MemcpyNodeSetParamsFromSymbol]: Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the * device.
//   - [GraphExec.MemcpyNodeSetParamsToSymbol]: Sets the parameters for a memcpy node in the given graphExec to copy to a symbol on the device.
//   - [GraphExec.NodeSetParams]: Updates parameters of a created node on executable graph.
//   - [GraphExec.Update]: Check whether an executable graph can be updated with a graph and perform the update if * possible.
//   - [GraphNode.ExternalSemaphoresSignalNodeGetParams]: Returns external semaphore signal node params.
//   - [GraphNode.ExternalSemaphoresSignalNodeSetParams]: Updates node parameters in the external semaphore signal node.
//   - [GraphNode.ExternalSemaphoresWaitNodeGetParams]: Returns external semaphore wait node params.
//   - [GraphNode.ExternalSemaphoresWaitNodeSetParams]: Updates node parameters in the external semaphore wait node.
//   - [Graph.GetEdges]: Returns a graph's dependency edges.
//   - [Graph.GetNodes]: Returns graph nodes.
//   - [Graph.GetRootNodes]: Returns graph's root nodes.
//   - [GraphNode.HostNodeGetParams]: Returns a host node's parameters.
//   - [GraphNode.HostNodeSetParams]: Sets a host node's parameters.
//   - [Graph.Instantiate]: Creates an executable graph from a graph
//   - [Graph.InstantiateWithFlags]: Creates an executable graph from a graph.
//   - [Graph.InstantiateWithParams]: Creates an executable graph from a graph.
//   - [GraphNode.KernelNodeCopyAttributes]: Copies attributes from source node to destination node.
//   - [GraphNode.KernelNodeGetAttribute]: Gets a node attribute.
//   - [GraphNode.KernelNodeGetParams]: Gets kernel node's parameters.
//   - [GraphNode.KernelNodeSetAttribute]: Sets a node attribute.
//   - [GraphNode.KernelNodeSetParams]: Sets a kernel node's parameters.
//   - [GraphExec.Launch]: launches an executable graph in a stream
//   - [GraphNode.MemAllocNodeGetParams]: Returns parameters for memory allocation node
//   - [GraphNode.MemFreeNodeGetParams]: Returns parameters for memory free node
//   - [GraphNode.MemcpyNodeSetParams1D]: Sets a memcpy node's parameters to perform a 1-dimensional copy.
//   - [GraphNode.MemcpyNodeSetParamsFromSymbol]: Sets a memcpy node's parameters to copy from a symbol on the device.
//   - [GraphNode.MemcpyNodeSetParamsToSymbol]: Sets a memcpy node's parameters to copy to a symbol on the device.
//   - [GraphNode.MemsetNodeGetParams]: Gets a memset node's parameters.
//   - [GraphNode.MemsetNodeSetParams]: Sets a memset node's parameters.
//   - [GraphNode.FindInClone]: Finds a cloned version of a node.
//   - [GraphNode.GetDependencies]: Returns a node's dependencies.
//   - [GraphNode.GetDependentNodes]: Returns a node's dependent nodes.
//   - [GraphNode.GetEnabled]: Query whether a node in the given graphExec is enabled
//   - [GraphNode.GetType]: Returns a node's type.
//   - [GraphNode.SetEnabled]: Enables or disables the specified node in the given graphExec
//   - [GraphNode.SetParams]: Updates parameters of a created node.
//   - [Graph.ReleaseUserObject]: Release user object from graphs.
//   - [Graph.RemoveDependencies]: Removes dependency edges from a graph.
//   - [Graph.RetainUserObject]: Retain user object for graphs.
//   - [GraphExec.Upload]: uploads an executable graph in a stream
//   - [Stream.GraphicsMapResources]: Maps a graphics resource for access.
//   - [GraphicsResourceGetMappedPointer]: Gets device accessible address of a graphics resource.
//   - [Array.GraphicsSubResourceGetMapped]: Get an array through which to access a subresource of a mapped graphics resource.
//   - [Stream.GraphicsUnmapResources]: Unmaps graphics resources.
//   - [GraphicsUnregisterResource]: Unregisters a graphics resource.
//   - [HostAlloc]: Allocate device accessible page locked host memory
//   - [HostFree]: [Free] memory allocated by the hcc hip host memory allocation API This API performs an implicit...
//   - [HostGetDevicePointer]: Get Device pointer from Host Pointer allocated through [HostMalloc]
//   - [HostGetFlags]: Return flags associated with host pointer
//   - [HostMalloc]: Allocates device accessible page locked (pinned) host memory
//   - [HostRegister]: Register host memory so it can be accessed from the current device.
//   - [HostUnregister]: Un-register host pointer
//   - [ExternalMemory.Import]: Imports an external memory object.
//   - [ExternalSemaphore.Import]: Imports an external semaphore.
//   - [IpcCloseMemHandle]: Close memory mapped with [IpcOpenMemHandle]
//   - [Event.IpcGetEventHandle]: Gets an opaque interprocess handle for an event.
//   - [IpcGetMemHandle]: Gets an interprocess memory handle for an existing device memory allocation
//   - [NewEventIpcOpenEventHandle]: Opens an interprocess event handles.
//   - [IpcOpenMemHandle]: Opens an interprocess memory handle exported from another process and returns a device pointer...
//   - [Function.KernelNameRef]: Returns kernel name reference by function name.
//   - [Stream.KernelNameRefByPtr]: Retrives kernel for a given host pointer, unless stated otherwise.
//   - [NewMipmappedArray]: Create a mipmapped array.
//   - [MipmappedArray.Destroy]: Destroy a mipmapped array.
//   - [MipmappedArray.GetLevel]: Get a mipmapped array on a mipmapped level.
//   - [OccupancyMaxActiveBlocksPerMultiprocessor]: Returns occupancy for a device function.
//   - [OccupancyMaxActiveBlocksPerMultiprocessorWithFlags]: Returns occupancy for a device function.
//   - [OccupancyMaxPotentialBlockSize]: determine the grid and block sizes to achieves maximum occupancy for a kernel
//   - [PeekAtLastError]: Return last error returned by any HIP runtime API call.
//   - [DevicePtr.PointerGetAttribute]: Returns information about the specified pointer.[BETA]
//   - [PointerGetAttributes]: Returns attributes for the specified pointer
//   - [DevicePtr.PointerSetAttribute]: Sets information on the specified pointer.[BETA]
//   - [RuntimeGetVersion]: Returns the approximate HIP Runtime version.
//   - [SetDevice]: Set default device to be used for subsequent hip API calls from this thread.
//   - [SetDeviceFlags]: The current device behavior is changed according the flags passed.
//   - [SetValidDevices]: Set a list of devices that can be used.
//   - [SetupArgument]: Set a kernel argument.
//   - [ExternalSemaphore.SignalExternalSemaphoresAsync]: Signals a set of external semaphore objects.
//   - [NewTextureObjectTexObjectCreate]: Creates a texture object.
//   - [TextureObject.TexObjectDestroy]: Destroys a texture object.
//   - [TextureObject.TexObjectGetResourceDesc]: Gets resource descriptor of a texture object.
//   - [TextureObject.TexObjectGetResourceViewDesc]: Gets resource view descriptor of a texture object.
//   - [TextureObject.TexObjectGetTextureDesc]: Gets texture descriptor of a texture object.
//   - [ThreadExchangeStreamCaptureMode]: Swaps the stream capture mode of a thread.
//   - [NewUserObject]: Create an instance of userObject to manage lifetime of a resource.
//   - [UserObject.Release]: Release number of references to resource.
//   - [UserObject.Retain]: Retain number of references to resource.
//   - [ExternalSemaphore.WaitExternalSemaphoresAsync]: Waits on a set of external semaphore objects
//
// # Device Management
//
//   - [DeviceCanAccessPeer]: Determine if a device can access a peer's memory.
//   - [Device.ComputeCapability]: Returns the compute capability of the device
//   - [DeviceDisablePeerAccess]: Disable direct access from current device's virtual address space to memory allocations physically...
//   - [DeviceEnablePeerAccess]: Enable direct access from current device's virtual address space to memory allocations physically...
//   - [NewDevice]: Returns a handle to a compute device
//   - [DeviceGetAttribute]: Query for a specific device attribute.
//   - [DeviceGetByPCIBusId]: Returns a handle to a compute device.
//   - [DeviceGetCacheConfig]: Get Cache configuration for a specific Device
//   - [MemPool.GetDefault]: Returns the default memory pool of the specified device
//   - [DeviceGetGraphMemAttribute]: Get the mem attribute for graphs.
//   - [DeviceGetLimit]: Gets resource limits of current device
//   - [MemPool.Get]: Gets the current memory pool for the specified device
//   - [Device.GetName]: Returns an identifer string for the device.
//   - [DeviceGetP2PAttribute]: Returns a value for attribute of link between two devices
//   - [DeviceGetPCIBusId]: Returns a PCI Bus Id string for the device, overloaded to take int device ID.
//   - [DeviceGetSharedMemConfig]: Returns bank width of shared memory for current device
//   - [DeviceGetStreamPriorityRange]: Returns numerical values that correspond to the least and greatest stream priority.
//   - [Device.GetUuid]: Returns an UUID for the device.[BETA]
//   - [DeviceGraphMemTrim]: [Free] unused memory on specific device used for graph back to OS.
//   - [DeviceReset]: The state of current device is discarded and updated to a fresh state.
//   - [DeviceSetCacheConfig]: Set L1/Shared cache partition.
//   - [DeviceSetGraphMemAttribute]: Set the mem attribute for graphs.
//   - [DeviceSetLimit]: Sets resource limits of current device.
//   - [MemPool.Set]: Sets the current memory pool of a device
//   - [DeviceSetSharedMemConfig]: The bank width of shared memory on current device is set
//   - [DeviceSynchronize]: Waits on all active streams on current device
//   - [Device.TotalMem]: Returns the total amount of memory on the device.
//
// # Event Management
//
//   - [NewEventWithFlags]: Create an event with the specified flags
//   - [Event.Destroy]: Destroy the specified event.
//   - [Event.ElapsedTime]: Return the elapsed time between two events.
//   - [Event.Query]: Query event status
//   - [Event.Record]: Record an event in the specified stream.
//   - [Event.Synchronize]: Wait for an event to complete.
//
// # Kernel Launch
//
//   - [LaunchByPtr]: Launch a kernel.
//   - [Stream.LaunchCooperativeKernel]: launches kernel f with launch parameters and shared memory on stream with arguments passed to...
//   - [LaunchCooperativeKernelMultiDevice]: Launches kernels on multiple devices where thread blocks can cooperate and synchronize as they...
//   - [Stream.LaunchHostFunc]: Enqueues a host function call in a stream.
//   - [Stream.LaunchKernel]: C compliant kernel launch API
//
// # Memory Allocation
//
//   - [Malloc]: Allocate memory on the default accelerator
//   - [Malloc3D]: Create a 3D memory pointer on the device.
//   - [Array.Malloc3D]: Allocate an array on the device.
//   - [Array.Malloc]: Allocate an array on the device.
//   - [Stream.MallocAsync]: Allocates memory with stream ordered semantics
//   - [MemPool.MallocFromPoolAsync]: Allocates memory from a specified pool with stream ordered semantics.
//   - [MallocManaged]: Allocates memory that will be automatically managed by HIP.
//   - [MipmappedArray.Malloc]: Allocate a mipmapped array on the device.
//
// # Memory Operations
//
//   - [MemAddressFree]: Frees an address range reservation made via [MemAddressReserve]
//   - [MemAddressReserve]: Reserves an address range
//   - [MemAdvise]: Advise about the usage of a given memory range to HIP.
//   - [NewMemGenericAllocationHandle]: Creates a memory allocation described by the properties and size
//   - [MemGenericAllocationHandle.ExportToShareableHandle]: Exports an allocation to a requested shareable handle type.
//   - [MemGetAccess]: Get the access flags set for the given location and ptr.
//   - [DevicePtr.MemGetAddressRange]: Get information on memory allocations.
//   - [MemGetAllocationGranularity]: Calculates either the minimal or recommended granularity.
//   - [MemGenericAllocationHandle.GetAllocationPropertiesFromHandle]: Retrieve the property structure of the given handle.
//   - [MemGetInfo]: Query memory info.
//   - [MemGenericAllocationHandle.ImportFromShareableHandle]: Imports an allocation from a requested shareable handle type.
//   - [MemGenericAllocationHandle.Map]: Maps an allocation handle to a reserved virtual address range.
//   - [Stream.MemMapArrayAsync]: Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
//   - [NewMemPool]: Creates a memory pool
//   - [MemPool.Destroy]: Destroys the specified memory pool
//   - [MemPoolExportPointer]: Export data to share a memory pool allocation between processes.
//   - [MemPool.ExportToShareableHandle]: Exports a memory pool to the requested handle type.
//   - [MemPool.GetAccess]: Returns the accessibility of a pool from a device
//   - [MemPool.GetAttribute]: Gets attributes of a memory pool
//   - [MemPool.ImportFromShareableHandle]: Imports a memory pool from a shared handle.
//   - [MemPool.ImportPointer]: Import a memory pool allocation from another process.
//   - [MemPool.SetAccess]: Controls visibility of the specified pool between devices
//   - [MemPool.SetAttribute]: Sets attributes of a memory pool
//   - [MemPool.TrimTo]: Releases freed memory back to the OS
//   - [Stream.MemPrefetchAsync]: Prefetches memory to the specified destination device using HIP.
//   - [MemPtrGetInfo]: Get allocated memory size via memory pointer.
//   - [MemRangeGetAttribute]: Query an attribute of a given memory range in HIP.
//   - [MemRangeGetAttributes]: Query attributes of a given memory range in HIP.
//   - [MemGenericAllocationHandle.Release]: Release a memory handle representing a memory allocation which was previously allocated through...
//   - [NewMemGenericAllocationHandleRetainAllocationHandle]: Returns the allocation handle of the backing memory allocation given the address.
//   - [MemSetAccess]: Set the access flags for each location specified in desc for the given virtual address range.
//   - [MemUnmap]: Unmap memory allocation of a given address range.
//   - [Memcpy]: Copy data from src to dst.
//   - [Memcpy2D]: Copies data between host and device.
//   - [Array.Memcpy2DArrayTo]: Copies data between host and device.
//   - [Stream.Memcpy2DAsync]: Copies data between host and device.
//   - [ArrayConst.Memcpy2DFromArray]: Copies data between host and device.
//   - [ArrayConst.Memcpy2DFromArrayAsync]: Copies data between host and device asynchronously.
//   - [Array.Memcpy2DTo]: Copies data between host and device.
//   - [Array.Memcpy2DToArrayAsync]: Copies data between host and device.
//   - [Memcpy3D]: Copies data between host and device.
//   - [Stream.Memcpy3DAsync]: Copies data between host and device asynchronously.
//   - [Stream.MemcpyAsync]: Copy data from src to dst asynchronously.
//   - [Array.MemcpyAtoA]: Copies from one 1D array to another.
//   - [DevicePtr.MemcpyAtoD]: Copies from one 1D array to device memory.
//   - [Array.MemcpyAtoH]: Copies data between host and device.
//   - [Array.MemcpyAtoHAsync]: Copies from one 1D array to host memory.
//   - [Array.MemcpyDtoA]: Copies from device memory to a 1D array.
//   - [DevicePtr.MemcpyDtoD]: Copy data from Device to Device
//   - [DevicePtr.MemcpyDtoDAsync]: Copy data from Device to Device asynchronously
//   - [DevicePtr.MemcpyDtoH]: Copy data from Device to Host
//   - [DevicePtr.MemcpyDtoHAsync]: Copy data from Device to Host asynchronously
//   - [MemcpyFromSymbol]: Copies data from the given symbol on the device.
//   - [Stream.MemcpyFromSymbolAsync]: Copies data from the given symbol on the device asynchronously.
//   - [Array.MemcpyHtoA]: Copies data between host and device.
//   - [Array.MemcpyHtoAAsync]: Copies from host memory to a 1D array.
//   - [DevicePtr.MemcpyHtoD]: Copy data from Host to Device
//   - [DevicePtr.MemcpyHtoDAsync]: Copy data from Host to Device asynchronously
//   - [MemcpyParam2D]: Copies memory for 2D arrays.
//   - [Stream.MemcpyParam2DAsync]: Copies memory for 2D arrays.
//   - [MemcpyPeer]: Copies memory from one device to memory on another device.
//   - [Stream.MemcpyPeerAsync]: Copies memory from one device to memory on another device.
//   - [MemcpyToSymbol]: Copies data to the given symbol on the device.
//   - [Stream.MemcpyToSymbolAsync]: Copies data to the given symbol on the device asynchronously.
//   - [Stream.MemcpyWith]: Memory copy on the stream.
//   - [Memset]: Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant byte value...
//   - [Memset2D]: Fills the memory area pointed to by dst with the constant value.
//   - [Stream.Memset2DAsync]: Fills asynchronously the memory area pointed to by dst with the constant value.
//   - [Memset3D]: Fills synchronously the memory area pointed to by pitchedDevPtr with the constant value.
//   - [Stream.Memset3DAsync]: Fills asynchronously the memory area pointed to by pitchedDevPtr with the constant value.
//   - [Stream.MemsetAsync]: Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant byte value...
//   - [DevicePtr.MemsetD16]: Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant short value...
//   - [DevicePtr.MemsetD16Async]: Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant short value...
//   - [DevicePtr.MemsetD32]: Fills the memory area pointed to by dest with the constant integer value for specified number of...
//   - [DevicePtr.MemsetD32Async]: Fills the memory area pointed to by dev with the constant integer value for specified number of...
//   - [DevicePtr.MemsetD8]: Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant byte value...
//   - [DevicePtr.MemsetD8Async]: Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant byte value...
//
// # Module Management
//
//   - [Module.GetFunction]: Function with kname will be extracted if present in module
//   - [Module.GetGlobal]: Returns a global pointer from a module.
//   - [Module.GetTexRef]: returns the handle of the texture reference with the name from the module.
//   - [Function.LaunchCooperativeKernel]: launches kernel f with launch parameters and shared memory on stream with arguments passed to...
//   - [ModuleLaunchCooperativeKernelMultiDevice]: Launches kernels on multiple devices where thread blocks can cooperate and synchronize as they...
//   - [Function.LaunchKernel]: launches kernel f with launch parameters and shared memory on stream with arguments passed to...
//   - [NewModule]: Loads code object from file into a module the currrent context.
//   - [NewModuleData]: builds module from code object which resides in host memory.
//   - [NewModuleDataEx]: builds module from code object which resides in host memory.
//   - [Function.OccupancyMaxActiveBlocksPerMultiprocessor]: Returns occupancy for a device function.
//   - [Function.OccupancyMaxActiveBlocksPerMultiprocessorWithFlags]: Returns occupancy for a device function.
//   - [Module.Unload]: Frees the module
//
// # Stream Management
//
//   - [Stream.AddCallback]: Adds a callback to be called on the host after all currently enqueued items in the stream have...
//   - [Stream.AttachMemAsync]: Attach memory to a stream asynchronously in HIP.
//   - [Stream.BeginCapture]: Begins graph capture on a stream.
//   - [Stream.BeginCaptureToGraph]: Begins graph capture on a stream to an existing graph.
//   - [NewStream]: Create an asynchronous stream.
//   - [NewStreamWithFlags]: Create an asynchronous stream.
//   - [NewStreamWithPriority]: Create an asynchronous stream with the specified priority.
//   - [Stream.Destroy]: Destroys the specified stream.
//   - [Stream.EndCapture]: Ends capture on a stream, returning the captured graph.
//   - [Stream.GetCaptureInfo]: Get capture status of a stream.
//   - [Stream.GetDevice]: Get the device assocaited with the stream
//   - [Stream.GetFlags]: Return flags associated with this stream.
//   - [Stream.GetPriority]: Query the priority of a stream.
//   - [Stream.IsCapturing]: Get stream's capture state
//   - [Stream.Query]: Return #[Success] if all of the operations in the specified stream have completed, or...
//   - [Stream.Synchronize]: Wait for all commands in stream to complete.
//   - [Stream.UpdateCaptureDependencies]: Update the set of dependencies in a capturing stream
//   - [Stream.WaitEvent]: Make the specified compute stream wait for an event
//   - [Stream.WaitValue32]: Enqueues a wait command to the stream.[BETA]
//   - [Stream.WaitValue64]: Enqueues a wait command to the stream.[BETA]
//   - [Stream.WriteValue32]: Enqueues a write command to the stream.[BETA]
//   - [Stream.WriteValue64]: Enqueues a write command to the stream.[BETA]
//
// [HIP Runtime API]: https://rocm.docs.amd.com/projects/HIP/en/latest/
package driver

// Code generated by gen-rocm from hip_runtime_api.h. DO NOT EDIT.
