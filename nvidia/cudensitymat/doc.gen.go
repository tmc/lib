// Package cudensitymat provides Go bindings for NVIDIA cuDensityMat, the cuQuantum
// library for many-body quantum operator action and expectation values on
// dense quantum states.
//
// This package loads and probes libcudensitymat independently. It shares no
// loader, handle, or symbol table with any other cuQuantum package.
//
// The bound surface is one path: build a [Handle], define a [State] and
// attach its component storage, build an [Operator] out of
// [ElementaryOperator] products grouped into an [OperatorTerm], size a
// [WorkspaceDescriptor] with PrepareAction, then accumulate the action with
// ComputeAction. [Expectation] computes the unnormalized expectation value
// of the same operator. Matrix-product states, batched entry points, matrix
// and MPO operators, gradients, time propagation, eigen-decomposition and
// the distributed interface table are not bound.
//
// ResetDistributedConfiguration imports a native communicator value. commPtr
// must point to the selected provider implementation's communicator representation;
// commSize is its byte size. The library copies that value, not the communicator's
// underlying resources. Keep the communicator valid while the context uses it.
// The NCCL provider is experimental in the pinned header. Distributed numerical
// execution requires separate runtime qualification.
//
// # Ownership
//
// Every pointer this package passes to the library is a plain address. None
// of it may be Go memory whose lifetime Go controls, and none of it is
// copied by the library unless the header says so.
//
// Component storage attached with State.AttachComponentStorage is
// caller-owned GPU memory. The library reads and writes it for the whole
// life of the state and does not free it. Its initial contents are the
// state's initial value. Free it only after State.Destroy.
//
// Workspace memory attached with WorkspaceDescriptor.SetMemory is
// caller-owned. It must stay valid and, for the scratch kind, at least as
// large as WorkspaceDescriptor.GetMemorySize reported after the matching
// Prepare call, until every operation that names the descriptor has
// completed on its stream.
//
// Elementary tensor operator storage passed to NewElementaryOperator or
// ElementaryOperator.AttachBuffer is caller-owned GPU memory that must
// outlive the operator. The header warns that two elementary operators must
// not share or overlap storage.
//
// Mode extents, mode lists and duality arrays are borrowed for the duration
// of the call only. The scaling factors, norm, trace, inner product,
// expectation value and parameter arrays are GPU-accessible addresses the
// library reads or writes on the supplied stream, so they must stay valid
// until that stream is synchronized, not merely until the call returns.
//
// Handles form a lifetime tree that this package does not enforce: an
// Expectation must not outlive its Operator, an Operator must not outlive
// the terms appended to it, and none of them may outlive the Handle. Destroy
// in reverse order of creation.
//
// # Streams
//
// Every Compute and Initialize call is asynchronous with respect to the
// host: it enqueues work on the supplied [Stream] and returns. A returned
// nil error means the work was enqueued, not that it finished. Results
// written to device memory are only readable after the caller synchronizes
// that stream. Prepare calls take a stream as well and may run device work
// while measuring the workspace.
//
// # Callbacks
//
// cuDensityMat accepts user-supplied scalar and tensor coefficient
// callbacks. This package deliberately provides no way to build one from a
// Go function. The wrapped callback structs are bound because they occupy
// argument slots in the operator-construction calls, and their Callback
// field is a uintptr, but the zero value -- [ScalarCallbackNone] and
// [TensorCallbackNone] -- is the only value this package supports.
//
// The header fixes the callback signature and says a callback may be a CPU
// or a GPU function, but it does not say which thread invokes a CPU
// callback, whether a CUDA context is current on that thread, whether the
// supplied stream may be used from inside it, or how long the library
// retains the pointer. Registering a Go function under those unknowns risks
// calling into the Go runtime from a thread the runtime does not own, so the
// contract is left unbound rather than guessed.
//
// # Verification status
//
// No call in this package has been run against libcudensitymat. It was
// generated and reviewed on a machine with no NVIDIA GPU, no CUDA driver
// and no cuQuantum installation.
//
// What has been checked, without hardware: the struct layouts against a
// clang record dump of the pinned header, the enumerator values against the
// header text, and the whole bound surface against a CPU-only shared-library
// fixture that validates every argument it receives -- built for both arm64
// and x86_64, so the by-value passing of Complex128 and the wrapped callback
// structs is exercised under both calling conventions. That establishes
// argument order, argument width and struct classification. It establishes
// nothing about what the real library computes.
//
// The tests that would establish that -- a known density matrix, a
// commutator, and a comparison against a CPU calculation including trace and
// hermiticity -- are written and have never been run. They skip unless
// CUDENSITYMAT_PHYSICAL_TEST=1 is set, and a skip is not evidence.
//
// # Library context
//
//   - [Create]: Creates and initializes the library context.
//   - [Handle.CreateElementaryOperator]: Creates an elementary tensor operator acting on a given number of quantum state modes (aka space modes).
//   - [Handle.CreateExpectation]: Creates the operator expectation value computation object.
//   - [Handle.CreateOperator]: Creates an empty operator which is going to be a collection of operator terms with some coefficients.
//   - [Handle.CreateOperatorTerm]: Creates an empty operator term which is going to be a sum of products of either elementary tensor operators or full matrix operators.
//   - [Handle.CreateState]: Defines an empty dense quantum state of a given purity and shape, or a batch of such dense quantum states.
//   - [Handle.CreateWorkspace]: Creates a workspace descriptor.
//   - [Handle.Destroy]: Destroys the library context.
//   - [ElementaryOperator.Destroy]: Destroys an elementary tensor operator (simple or batched).
//   - [Expectation.Destroy]: Destroys an expectation value computation object.
//   - [Operator.Destroy]: Destroys an operator.
//   - [OperatorTerm.Destroy]: Destroys an operator term.
//   - [State.Destroy]: Destroys the quantum state.
//   - [WorkspaceDescriptor.Destroy]: Destroys a workspace descriptor.
//   - [Handle.GetNumRanks]: Returns the total number of distributed processes associated with the given library context in its current distributed execution configuration.
//   - [Handle.GetProcRank]: Returns the rank of the current process in the distributed execution configuration associated with the given library context.
//   - [GetVersion]: Returns the semantic version number of the cuDensityMat library.
//   - [Handle.ResetDistributedConfiguration]: Resets the current distributed execution configuration associated with the given library context by importing a user-provided inter-process communicator (e.g., MPI_Comm).
//   - [Handle.ResetRandomSeed]: Resets the context-level random seed used by the random number generator inside the library context.
//
// # Operator construction
//
//   - [ElementaryOperator.AttachBuffer]: Attaches a buffer to the elementary tensor operator (either batched or non-batched).
//   - [Operator.AppendTerm]: Appends an operator term to the operator.
//   - [OperatorTerm.AppendElementaryProduct]: Appends a product of elementary tensor operators acting on quantum state modes to the operator term.
//
// # Expectation values
//
//   - [Expectation.Compute]: Computes the operator expectation value(s) with respect to the given quantum state(s).
//   - [Expectation.Prepare]: Prepares the expectation value object for computation.
//
// # Operator action
//
//   - [Operator.ComputeAction]: Computes the action of the operator on a given input quantum state, accumulating the result in the output quantum state (accumulative action).
//   - [Operator.PrepareAction]: Prepares the operator for an action on a quantum state.
//
// # Quantum state
//
//   - [State.AttachComponentStorage]: Attaches a user-owned GPU-accessible storage buffer for each component (tensor) constituting the quantum state representation (on the current process in multi-process runs).
//   - [State.ComputeAccumulation]: Computes accumulation of a quantum state(s) into another quantum state(s) of compatible shape.
//   - [State.ComputeInnerProduct]: Computes the inner product(s) between the left quantum state(s) and the right quantum state(s): < state(s)Left | state(s)Right >
//   - [State.ComputeNorm]: Computes the squared Frobenius norm(s) of the quantum state(s).
//   - [State.ComputeScaling]: Multiplies the quantum state(s) by a scalar factor(s).
//   - [State.ComputeTrace]: Computes the trace(s) of the quantum state(s).
//   - [State.GetComponentInfo]: Queries information for a locally stored component tensor which represents either the full component or its slice (on the current process in multi-process runs).
//   - [State.GetComponentNumModes]: Queries the number of modes in a local component tensor (on the current process in multi-process runs).
//   - [State.GetComponentStorageSize]: Queries the storage size (in bytes) for each component (tensor) constituting the quantum state representation (on the current process in multi-process runs).
//   - [State.GetNumComponents]: Queries the number of components (tensors) constituting the chosen quantum state representation (on the current process in multi-process runs).
//   - [State.InitializeZero]: Initializes the quantum state to zero (null state).
//
// # Workspace
//
//   - [Handle.WorkspaceGetMemory]: Retrieves a workspace buffer.
//   - [Handle.WorkspaceGetMemorySize]: Queries the required workspace buffer size.
//   - [Handle.WorkspaceSetMemory]: Attaches memory to a workspace buffer.
//
// [cuDensityMat documentation]: https://docs.nvidia.com/cuda/cuquantum/latest/densitymat/index.html
package cudensitymat

// Code generated by gen-cudensitymat from cudensitymat.h. DO NOT EDIT.
