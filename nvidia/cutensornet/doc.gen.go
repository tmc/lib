// Package cutensornet provides lazy-loaded bindings for NVIDIA cuTensorNet, the
// tensor-network contraction library shipped with cuQuantum.
//
// The contraction path is: create a [Handle]; describe the network with
// [Handle.NewNetwork]; find a contraction path with
// [Handle.CreateContractionOptimizerConfig], [Handle.CreateContractionOptimizerInfo]
// and [Handle.ContractionOptimize]; size and fund a workspace with
// [Handle.NewWorkspace]; build a plan with [Workspace.NewPlan]; and execute
// it with [Plan.Contract] on a CUDA stream.
//
// Tensors are strided, not packed. Every input and the output carries its own
// stride array in elements. A nil stride array means the library derives
// generalized column-major (Fortran) strides from the extents; passing an
// explicit array is the only way to describe any other layout, including a
// row-major one.
//
// # Ownership
//
// Mode, extent and stride arrays are borrowed for the duration of the
// descriptor call only; the library copies what it needs. Device buffers, the
// workspace, and the CUDA stream are caller-owned and must stay valid and
// unmodified until the stream work completes. A workspace buffer handed over
// with [Handle.WorkspaceSetMemory] is held by the descriptor across planning
// and every execution that uses it, so it must outlive the plan; [Workspace]
// is a Go-side owner that enforces that ordering, and [Plan] keeps both the
// workspace and the [Network] alive for as long as the library may read
// them. Cache-kind workspace additionally must not be modified between
// iterations that reuse it. Closing a workspace does not synchronize the
// stream: the caller must do that before any buffer is released. All
// handles must be destroyed in reverse order of creation.
//
// This package does not allocate device memory. Pair it with a CUDA memory
// allocator such as [github.com/tmc/lib/nvidia/driver].
//
// # Scope
//
// Only the tensor-network contraction API is bound. cutensornet.h also
// declares tensor decomposition, quantum state simulation and distributed
// reduction APIs that this package does not expose.
//
// cuTensorNet 2.13 deprecates the classic contraction entry points -- the
// ones this package's [Network], [Workspace] and [Plan] are built on -- in
// favour of the cutensornetNetwork family, bound here as
// [Handle.CreateNetwork], [Handle.NetworkAppendTensor],
// [Handle.NetworkSetOutputTensor], [Handle.NetworkPrepareContraction] and
// [Handle.NetworkContract]. Both families are present; the upstream
// Deprecated markers are carried through to the individual functions.
//
// # Verification status
//
// Nothing in this package has been run against a GPU. The struct layouts
// asserted by the generated tests were measured by the C compiler on the
// machine that ran the generator, so they are a check that the Go structs
// agree with the pinned header on that ABI, not evidence about the
// linux/x86_64 or linux/sbsa build of libcutensornet.
//
// # Execution
//
//   - [Handle.ContractSlices].
//
// # Path optimization
//
//   - [Handle.ContractionAutotune].
//   - [ContractionAutotunePreference.GetAttribute].
//   - [ContractionAutotunePreference.SetAttribute].
//   - [Handle.ContractionOptimize].
//   - [ContractionOptimizerConfig.GetAttribute].
//   - [ContractionOptimizerConfig.SetAttribute].
//   - [ContractionOptimizerInfo.GetAttribute].
//   - [ContractionOptimizerInfo.GetPackedSize].
//   - [ContractionOptimizerInfo.PackData].
//   - [ContractionOptimizerInfo.SetAttribute].
//
// # Descriptor lifecycle
//
//   - [Create].
//   - [Handle.CreateContractionAutotunePreference].
//   - [Handle.CreateContractionOptimizerConfig].
//   - [Handle.CreateContractionOptimizerInfo].
//   - [Handle.CreateContractionOptimizerInfoFromPackedData].
//   - [Handle.CreateContractionPlan].
//   - [Handle.CreateNetwork].
//   - [Handle.CreateNetworkAutotunePreference].
//   - [Handle.CreateNetworkDescriptor].
//   - [Handle.CreateSliceGroupFromIDRange].
//   - [Handle.CreateSliceGroupFromIDs].
//   - [Handle.CreateTensorDescriptor].
//   - [Handle.CreateWorkspaceDescriptor].
//   - [Handle.Destroy].
//   - [ContractionAutotunePreference.Destroy].
//   - [ContractionOptimizerConfig.Destroy].
//   - [ContractionOptimizerInfo.Destroy].
//   - [ContractionPlan.Destroy].
//   - [NetworkDescriptor.Destroy].
//   - [NetworkAutotunePreference.Destroy].
//   - [NetworkDescriptor.DestroyDescriptor].
//   - [SliceGroup.Destroy].
//   - [TensorDescriptor.Destroy].
//   - [WorkspaceDescriptor.Destroy].
//
// # Queries
//
//   - [GetCudartVersion].
//   - [Handle.GetOutputTensorDescriptor].
//   - [Handle.GetOutputTensorDetails].
//   - [Handle.GetTensorDetails].
//   - [GetVersion].
//
// # Logging
//
//   - [LoggerForceDisable].
//   - [LoggerOpenFile].
//   - [LoggerSetLevel].
//   - [LoggerSetMask].
//
// # Library context
//
//   - [Handle.NetworkAppendTensor].
//   - [Handle.NetworkAutotuneContraction].
//   - [NetworkAutotunePreference.GetAttribute].
//   - [NetworkAutotunePreference.SetAttribute].
//   - [Handle.NetworkContract].
//   - [Handle.NetworkGetAttribute].
//   - [Handle.NetworkPrepareContraction].
//   - [Handle.NetworkSetAttribute].
//   - [Handle.NetworkSetInputTensorMemory].
//   - [Handle.NetworkSetOptimizerInfo].
//   - [Handle.NetworkSetOutputTensor].
//   - [Handle.NetworkSetOutputTensorMemory].
//   - [Handle.UpdateContractionOptimizerInfoFromPackedData].
//
// # Workspace
//
//   - [Handle.WorkspaceComputeContractionSizes].
//   - [Handle.WorkspaceGetMemory].
//   - [Handle.WorkspaceGetMemorySize].
//   - [Handle.WorkspacePurgeCache].
//   - [Handle.WorkspaceSetMemory].
//
// [cuTensorNet documentation]: https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html
package cutensornet

// Code generated by gen-cutensornet from cutensornet.h. DO NOT EDIT.
