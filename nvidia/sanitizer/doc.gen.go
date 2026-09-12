// Package sanitizer provides low-level Go bindings for the CUDA Compute Sanitizer
// public API, the callback and patching interface NVIDIA's compute-sanitizer
// tool is built on.
//
// The API has four parts, all host-side:
//
//   - Subscriber: one process-wide subscription that delivers a callback at
//     driver, runtime, resource, launch, memory, graph, event and error
//     events ([Subscribe], [Subscriber.EnableDomain]).
//   - Patching: loading a compiled instrumentation cubin into a context and
//     attaching its device functions to instruction kinds ([AddPatches],
//     [PatchInstructions], [PatchModule]).
//   - Callback-safe memory: allocation, transfer and fill entry points that
//     are legal to call from inside a callback, where the ordinary CUDA API
//     is not ([Alloc], [StreamHandle.MemcpyDeviceToHost],
//     [StreamHandle.Memset]).
//   - Streams and barriers: converting between a CUstream and the
//     sanitizer's own stream handle, and counting a kernel's barriers
//     ([GetStreamHandle], [GetCudaBarrierCount]).
//
// # Device instrumentation is out of scope
//
// Roughly half of sanitizer_patching.h is device code and is deliberately not
// bound here. The 34 SanitizerCallback* typedefs -- SanitizerCallbackBarrier,
// SanitizerCallbackMemoryAccess and the rest -- are function pointer types
// for functions that run on the GPU, once per thread, at a patched
// instruction. The caller writes them in CUDA C, compiles them with nvcc into
// a cubin, PTX or fatbin image, loads that image with [AddPatches] or
// [AddPatchesFromFile], and attaches one to an instruction kind by naming its
// device symbol as a string to [PatchInstructions].
//
// None of them is a symbol of libsanitizer-public.so and none can be
// implemented in Go: a Go function has a host address, and installing one as
// a device instruction handler is not something the hardware can do. The
// [InstructionId] constants say which device callback signature each
// instruction kind requires, so the mapping is available from Go even though
// the functions are not.
//
// [InstructionTmaStore] is documented as requiring a
// SanitizerCallbackTensorMemoryStore, which no typedef in the pinned 13.3.27
// headers declares. That is a gap in NVIDIA's header, not in this binding.
//
// # One subscriber, and it must come first
//
// Compute Sanitizer allows exactly one subscriber per process. A second
// subscribe while one is live fails with [ErrMaxLimitReached]; this package
// returns that error rather than hiding it, and does not offer a way to
// register a second. Treat the [Subscription] as the process's single owner
// of the callback stream, and multiplex in Go if more than one consumer
// needs the events.
//
// [SubscribeFunc] is the entry point to use. It takes a Go [CallbackFunc],
// builds the single process-wide purego trampoline that dispatches to it,
// and returns a [Subscription] that owns the callback for its lifetime.
// [Subscribe] is the raw binding underneath it: it takes an already-built
// host function pointer and leaves the lifetime problems below to you.
//
// Subscribing enables nothing. Afterwards, enable the domains or individual
// callback ids you want with [Subscription.EnableDomain],
// [Subscription.EnableCallback] or [Subscription.EnableAllDomains].
//
// For complete reporting the subscription must exist before the process makes
// its first CUDA call. Events that occur before [Subscribe] returns are not
// replayed: a context created before subscribing produces no
// CONTEXT_CREATION_FINISHED callback, and a module loaded before subscribing
// produces no MODULE_LOADED callback, so anything derived from those --
// patched modules, tracked allocations -- is silently incomplete. Subscribe
// from an init function, or as the first statement of main, before touching
// any CUDA API.
//
// [Subscriber.GetCallbackState], [Subscriber.EnableCallback],
// [Subscriber.EnableDomain] and [Subscriber.EnableAllDomains] are documented
// as undefined, not merely racy, when called concurrently with each other.
// Serialize them. The methods of the same name on [Subscription] hold a
// mutex and are safe to call concurrently; the ones on the bare [Subscriber]
// handle are not. The rest of the API is documented thread-safe.
//
// # Callback lifetime and Go memory
//
// The callback data pointer a callback receives is valid only for that
// invocation. The header is explicit: to keep anything, copy it. A shallow
// copy of a [CallbackData] is not enough, because its functionParams field
// points at a parameter struct that also dies at return. The one exception
// the header names is functionName, which points at a global constant.
//
// A Go callback handed to the sanitizer must stay alive for the whole
// subscription. A callback created with purego.NewCallback is a host function
// pointer that the sanitizer stores and invokes later, from a thread this
// package did not create; if the Go closure it wraps becomes unreachable the
// program crashes at the next event, not at the point of the mistake. purego
// callbacks are also a limited process-wide resource -- a fixed pool of a
// couple of thousand that is never reclaimed -- so one created per
// subscription, or worse per enable, eventually exhausts the process.
//
// [SubscribeFunc] handles both. It creates exactly one trampoline for the
// life of the process, reuses it across subscriptions, and holds the Go
// function in a package-level slot that [Subscription.Unsubscribe] clears
// after the library has promised no further callbacks -- so a callback that
// arrives late finds an empty slot and returns instead of calling into freed
// state. Callers who use the raw [Subscribe] binding instead own all of that
// themselves: keep the callback and everything it closes over reachable from
// a package-level variable until [Subscriber.Unsubscribe] returns.
//
// The userdata pointer [SubscribeFunc] takes is stored by the sanitizer and
// handed back on every callback. It must not be a Go pointer: C storage that
// outlives the call cannot hold one, and the garbage collector is free to
// move or free what it points at. Pass a handle -- an index into a Go-side
// table, or a pointer to non-Go memory -- and look up the real state in Go.
//
// The userdata pointers [SetCallbackData],
// [LaunchHandle.SetLaunchCallbackData] and [StreamHandle.SetDeviceGraphData]
// take are a stronger version of the same rule: the
// header says these are device memory buffers, read by instrumentation code
// running on the GPU. Allocate them with [Alloc], never with Go's allocator.
//
// # Calling CUDA from inside a callback
//
// The pinned headers do not state a complete rule, and this package does not
// invent one. What they do state is that the seven entry points in
// sanitizer_memory.h are the equivalents of cudaMalloc, cudaMallocHost,
// cudaFree, cudaFreeHost, cudaMemcpyAsync, cudaMemcpy and cudaMemset "that
// can be called within a callback function" -- which is a reason for those
// functions to exist at all only if the cuda* originals cannot be. Use
// [Alloc], [AllocHost], [Free], [FreeHost],
// [StreamHandle.MemcpyHostToDeviceAsync], [StreamHandle.MemcpyDeviceToHost]
// and [StreamHandle.Memset] from inside a callback.
//
// For anything else, the headers give one tool and no rule:
// [IsInsideRecursiveCallback] reports whether the current thread is already
// executing a nested sanitizer callback, which the header describes as a way
// to tell the application's own CUDA calls apart from the ones a callback
// handler makes. The authoritative list of what is permitted is in NVIDIA's
// Compute Sanitizer API reference, not in these headers; do not infer it from
// the absence of a prohibition here.
//
// # Decoding functionParams is not provided
//
// For the driver and runtime domains, [CallbackData].FunctionParams points at
// one of roughly a thousand cu*_params or cuda*_params structs declared in
// the generated_*_meta.h headers. Those are not bound. Two reasons: they
// belong to the driver and runtime APIs rather than to this library, and they
// are in/out. At [ApiEnter] an output field of such a struct holds
// uninitialized memory; only at [ApiExit] does it hold a result. Any decoder
// must therefore be callbackSite-aware, and a generated Go struct cannot say
// so in its type. Callers who need the arguments should cast FunctionParams
// against a layout they own, having first checked CallbackSite.
//
// Similarly, the resource domain delivers a different payload struct per
// callback id, and the pinned header does not give the complete mapping: its
// documentation lists three of the nine resource payload structs and
// mistakenly names an enum as a fourth, and says nothing at all about the
// UVM, graphs, events, external-memory and error-logging domains. This
// package therefore does not dispatch on cbid. Casting the callback data on a
// guessed mapping reinterprets one struct as another of a different size,
// which is the memory-safety bug this API most invites; take the mapping from
// NVIDIA's reference manual, not from the struct names.
//
// # Struct layouts
//
// The generated struct layouts are measured from the pinned 13.3.27 headers
// by a compiler rather than inferred, and layout_test.go pins every size,
// alignment and field offset. The measurement happens wherever the generator
// runs, which is not necessarily the platform the bindings run on; the three
// targets that matter here -- linux/amd64, linux/arm64 and darwin/arm64 --
// produce byte-identical record layouts for this header set, because every
// field is a pointer, a fixed-width integer or an int-sized enum.
//
// [GraphNodeLaunchData] is the one type a Go struct cannot mirror. It
// contains an anonymous C union of [LaunchData], [ResourceMemoryData], a
// uint64 free address, [MemcpyData] and [MemsetData], discriminated by its
// NodeType field, and IsGraphUpload sits after the union. The Go type
// reserves the union's exact 144 bytes as AnonymousUnion so that
// IsGraphUpload lands at the right offset. Reading a member out of it
// requires the CUgraphNodeType values from cuda.h, which this package does
// not declare, so no typed accessor is offered: interpreting the bytes
// without first checking NodeType against the driver's own constants
// reinterprets one struct as another of a different size.
//
// # Loading
//
// libsanitizer-public.so is not on the loader's default search path. In a
// toolkit install it lives in $CUDA_HOME/compute-sanitizer/, beside the
// compute-sanitizer launcher. [Load] with an explicit path, set
// SANITIZER_LIB_PATH, or call [LoadFromToolkit], which tries every path
// [SearchPaths] reports and, on failure, returns a *NotFoundError naming
// every one it tried. A missing library or a missing
// symbol is reported as an error -- *SymbolError for the latter -- and never
// as a panic.
//
// Whether the library is usable from an ordinary process, rather than only
// under the compute-sanitizer launcher's injection, has not been verified by
// the authors of this package. The functions are bound; that a bound function
// succeeds outside the launcher is the caller's to establish.
//
// This package provides host bindings generated from the sanitizer.h header family
// using [github.com/ebitengine/purego] for cgo-free dynamic
// loading. Functions returning a C status report errors wrapping SanitizerResult with
// descriptive messages from sanitizerGetResultString.
//
// # Patching API
//
//   - [AddPatches]: Load a module containing patches that can be used by the patching API.
//   - [AddPatchesFromFile]: Load a module containing patches that can be used by the patching API.
//   - [GetCallbackPcAndSize]: Get PC and size of a device callback.
//   - [GetFunctionLoadedStatus]: Get the loading status of a function.
//   - [GetFunctionPcAndSize]: Get PC and size of a CUDA function.
//   - [PatchInstructions]: Set instrumentation points and patches to be applied in a module.
//   - [PatchModule]: Perform the actual instrumentation of a module.
//   - [SetCallbackData]: Specifies the user data pointer for callbacks.
//   - [StreamHandle.SetDeviceGraphData]: Specifies the user data pointer accessible from callbacks in the device-launched graphs launched by the specified host-launched graphExec.
//   - [LaunchHandle.SetLaunchCallbackData]: Specifies the user data pointer for callbacks.
//   - [UnpatchModule]: Remove existing instrumentation of a module.
//
// # Callback-safe Memory
//
//   - [Alloc]: Allocate memory on the device.
//   - [AllocHost]: Allocate host pinned memory.
//   - [Free]: Frees memory on the device.
//   - [FreeHost]: Frees host memory.
//   - [StreamHandle.MemcpyDeviceToHost]: Copies data from device to host.
//   - [StreamHandle.MemcpyHostToDeviceAsync]: Copies data from host to device.
//   - [StreamHandle.Memset]: Initializes or sets device memory to a value.
//
// # Subscriber API
//
//   - [Subscriber.EnableAllDomains]: Enable or disable all callbacks in all domains.
//   - [Subscriber.EnableCallback]: Enable or disable callbacks for a specific domain and callback ID
//   - [Subscriber.EnableDomain]: Enable or disable all callbacks for a specific domain.
//   - [Subscriber.GetCallbackState]: Get the current enabled/disabled state of a callback for a specific domain and function ID.
//   - [IsInsideRecursiveCallback]: Check if the current thread is inside a recursive Sanitizer callback.
//   - [Subscribe]: Initialize a callback subscriber with a callback function and user data.
//   - [Subscriber.Unsubscribe]: Unregister a callback subscriber.
//
// # Barrier API
//
//   - [GetCudaBarrierCount]: Get number of CUDA barriers used by a function.
//
// # Stream API
//
//   - [StreamHandle.GetStream]: Retrieve a [Stream] handle from a [StreamHandle] handle.
//   - [GetStreamHandle]: Retrieve a [StreamHandle] handle from a [Stream] handle
//   - [StreamHandle.Synchronize]: Synchronize a given stream.
//
// [Compute Sanitizer API Documentation]: https://docs.nvidia.com/compute-sanitizer/SanitizerApiGuide/
package sanitizer

// Code generated by gen-nvidia from sanitizer.h. DO NOT EDIT.
