// Package cufft provides low-level Go bindings for cuFFT (CUDA Fast Fourier Transform).
//
// cuFFT computes discrete Fourier transforms (DFTs) on NVIDIA GPUs,
// supporting 1D, 2D, and 3D transforms in single and double precision
// for real-to-complex, complex-to-real, and complex-to-complex transforms.
//
// The central type is [Handle], which represents a cuFFT plan. Create plans
// with [Plan1d], [Plan2d], [Plan3d], or [PlanMany], execute with the Exec*
// methods (e.g., [Handle.ExecC2C]), and clean up with [Handle.Destroy].
//
// This package provides bindings generated from cufft.h using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libcufft.
// Functions returning a C status report errors wrapping cufftResult.
//
// # Plan Creation
//
//   - [Create]: Creates only an opaque handle, and allocates small data structures on the host.
//   - [Handle.MakePlan1d]: Following a call to [Create] makes a 1D FFT plan configuration for a specified signal size and data type.
//   - [Handle.MakePlan2d]: Following a call to [Create] makes a 2D FFT plan configuration according to specified signal sizes and data type.
//   - [Handle.MakePlan3d]: Following a call to [Create] makes a 3D FFT plan configuration according to specified signal sizes and data type.
//   - [Handle.MakePlanMany]: Following a call to [Create] makes a FFT plan configuration of dimension rank, with sizes specified in the array n.
//   - [Handle.MakePlanMany64]: Following a call to [Create] makes a FFT plan configuration of dimension rank, with sizes specified in the array n.
//   - [Plan1d]: Creates a 1D FFT plan configuration for a specified signal size and data type.
//   - [Plan2d]: Creates a 2D FFT plan configuration according to specified signal sizes and data type.
//   - [Plan3d]: Creates a 3D FFT plan configuration according to specified signal sizes and data type.
//   - [PlanMany]: Creates a FFT plan configuration of dimension rank, with sizes specified in the array n.
//
// # Plan Lifecycle
//
//   - [Handle.Destroy]: Frees all GPU resources associated with a cuFFT plan and destroys the internal plan data structure.
//
// # Execution
//
//   - [Handle.ExecC2C]: [Handle.ExecC2C] ( [Handle.ExecZ2Z]) executes a single-precision (double-precision) complex-to-complex transform plan in the transform direction as specified by direction parameter.
//   - [Handle.ExecC2R]: [Handle.ExecC2R] ( [Handle.ExecZ2D]) executes a single-precision (double-precision) complex-to-real, implicitly inverse, cuFFT transform plan.
//   - [Handle.ExecD2Z]: [Handle.ExecR2C] ( [Handle.ExecD2Z]) executes a single-precision (double-precision) real-to-complex, implicitly forward, cuFFT transform plan.
//   - [Handle.ExecR2C]: [Handle.ExecR2C] ( [Handle.ExecD2Z]) executes a single-precision (double-precision) real-to-complex, implicitly forward, cuFFT transform plan.
//   - [Handle.ExecZ2D]: [Handle.ExecC2R] ( [Handle.ExecZ2D]) executes a single-precision (double-precision) complex-to-real, implicitly inverse, cuFFT transform plan.
//   - [Handle.ExecZ2Z]: [Handle.ExecC2C] ( [Handle.ExecZ2Z]) executes a single-precision (double-precision) complex-to-complex transform plan in the transform direction as specified by direction parameter.
//
// # Version Queries
//
//   - [GetProperty]: Return in *value the number for the property described by type of the dynamically linked CUFFT library.
//   - [GetVersion]: Returns the version number of cuFFT.
//
// # Configuration
//
//   - [Handle.SetAutoAllocation]: [Handle.SetAutoAllocation] indicates that the caller intends to allocate and manage work areas for plans that have been generated.
//   - [Handle.SetStream]: Associates a CUDA stream with a cuFFT plan.
//   - [Handle.SetWorkArea]: [Handle.SetWorkArea] overrides the work area pointer associated with a plan.
//
// # Other
//
//   - [Estimate1d]: During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
//   - [Estimate2d]: During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
//   - [Estimate3d]: During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
//   - [EstimateMany]: During plan execution, cuFFT requires a work area for temporary storage of intermediate results.
//   - [Handle.GetPlanPropertyInt64]: Retrieves the property value identified by the key property associated with the cuFFT plan plan.
//   - [Handle.GetSize]: Once plan generation has been done, either with the original API or the extensible API, this call returns the actual size of the work area required to support the plan.
//   - [Handle.GetSize1d]: This call gives a more accurate estimate of the work area size required for a plan than [Estimate1d], given the specified parameters, and taking into account any plan settings that may have been made.
//   - [Handle.GetSize2d]: This call gives a more accurate estimate of the work area size required for a plan than [Estimate2d], given the specified parameters, and taking into account any plan settings that may have been made.
//   - [Handle.GetSize3d]: This call gives a more accurate estimate of the work area size required for a plan than [Estimate3d], given the specified parameters, and taking into account any plan settings that may have been made.
//   - [Handle.GetSizeMany]: This call gives a more accurate estimate of the work area size required for a plan than cufftEstimateSizeMany(), given the specified parameters, and taking into account any plan settings that may have been made.
//   - [Handle.GetSizeMany64]: This call gives a more accurate estimate of the work area size required for a plan than cufftEstimateSizeMany(), given the specified parameters, and taking into account any plan settings that may have been made.
//   - [Handle.ResetPlanProperty]: Resets the value of the property identified by the key property, associated with the cuFFT plan plan, to its default value.
//   - [Handle.SetPlanPropertyInt64]: Associates a cuFFT plan with a property identified by the key property.
//
// [cuFFT Documentation]: https://docs.nvidia.com/cuda/cufft/
package cufft

// Code generated by gen-nvidia from cufft.h. DO NOT EDIT.
