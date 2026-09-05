// Package custatevec provides lazy-loaded bindings for NVIDIA cuStateVec, the
// state-vector simulator shipped with cuQuantum.
//
// The bound path is: create a [Handle]; optionally attach a workspace with
// [Handle.SetWorkspace]; allocate a device state vector of 2^nIndexBits
// complex elements and fill it with [Handle.InitializeStateVector]; apply
// gates with [Handle.ApplyMatrix]; and read the state back out with
// [Handle.Abs2SumArray], [Handle.MeasureOnZBasis] or
// [Handle.ComputeExpectation].
//
// The declarations are generated from the custatevec.h shipped in cuQuantum
// 26.06.0.17 (CUSTATEVEC_VERSION 11400); cmd/gen-custatevec/headers/README.md
// records the source URL and header checksums. The library itself is opened
// on first use, by soname libcustatevec.so.1 or from CUSTATEVEC_LIB_PATH;
// nothing is loaded at import time.
//
// # Bit ordering
//
// Every bit position in this API is a state-vector index bit, numbered in
// little-endian order: bit 0 is the least significant bit of the state
// vector index, so amplitude sv[i] belongs to the computational basis state
// whose qubit b value is (i>>b)&1. Qubit 0 therefore selects adjacent pairs
// of amplitudes and qubit nIndexBits-1 splits the vector in half. A caller
// who numbers qubits from the other end sees plausible but wrong amplitudes,
// not an error: no call validates the intent behind a bit position. See
// [BitOrdering] for the constant that records this convention.
//
// # Complex representation
//
// A state vector is an array of [Complex64] or [Complex128], the Go
// spellings of CUDA's cuComplex (float2) and cuDoubleComplex (double2).
// These are structs of two floats, not Go's complex64 and complex128, whose
// memory layout the Go specification does not fix. cuDoubleComplex is
// declared __align__(16) in CUDA and Go cannot express a 16-byte struct
// alignment, so a Go []Complex128 has the right element size and stride but
// only 8-byte base alignment. Use [AlignedComplex128] for a host buffer that
// matches the C alignment.
//
// # Data types
//
// The state vector, the gate or observable matrix and the arithmetic are
// typed separately, and the header accepts only three combinations:
//
//   - a C64F state vector with a C64F matrix and [Compute64F]
//   - a C32F state vector with a C64F matrix and [Compute32F]
//   - a C32F state vector with a C32F matrix and [Compute32F]
//
// [ComputeTF32] is declared by the header but is documented as not
// available. [ComputeDefault] selects Compute64F for a C64F state vector
// and Compute32F for a C32F one. An unsupported combination is reported by
// the library as [ErrNotSupported], not rejected by this package.
//
// # Ownership
//
// The state vector, the extra workspace and the workspace attached to the
// handle are device memory this package neither allocates nor frees; pair it
// with a CUDA allocator such as [github.com/tmc/lib/nvidia/driver]. They
// must stay valid and unmodified until the work submitted on the handle's
// stream completes. The bit-position arrays (targets, controls, basisBits,
// bitOrdering, maskBitString, maskOrdering) and a host-resident gate matrix
// are host memory the library reads while the call runs; a Go allocation
// passed as one of them must be pinned with [runtime.Pinner] across the
// call. A [Handle] is not safe for concurrent use.
//
// Only the single-GPU minimum is bound. custatevec.h also declares a
// sampler, an accessor, batched and multi-GPU APIs, a distributed
// communicator layer and a logging layer that this package does not expose.
//
// # Observation
//
//   - [Handle.Abs2SumArray]: sums the squared absolute values of state vector elements over a set of index bits.
//   - [Handle.ComputeExpectation]: computes the expectation value of an observable matrix.
//   - [Handle.ComputeExpectationGetWorkspaceSize]: returns the extra workspace size in bytes that a matching [Handle.ComputeExpectation] call needs.
//   - [Handle.MeasureOnZBasis]: measures a state vector on the Z basis and returns the parity of the measured bits.
//
// # Gate application
//
//   - [Handle.ApplyMatrix]: applies a gate matrix to a state vector in place.
//   - [Handle.ApplyMatrixGetWorkspaceSize]: returns the extra workspace size in bytes that a matching [Handle.ApplyMatrix] call needs.
//
// # Library context
//
//   - [Create]: creates a cuStateVec library handle bound to the current CUDA device.
//   - [Handle.Destroy]: releases a handle and the resources it owns.
//   - [GetProperty]: reports one component of the linked library version.
//   - [Handle.GetStream]: returns the CUDA stream bound to the handle.
//   - [GetVersion]: returns the linked library version, which the header encodes as major*10000 + minor*100 + patch.
//   - [Handle.SetStream]: binds the CUDA stream on which subsequent calls enqueue work.
//
// # Workspace
//
//   - [Handle.GetDefaultWorkspaceSize]: returns the size in bytes of the workspace the handle allocated for itself.
//   - [Handle.SetWorkspace]: attaches caller-owned device memory as the handle's workspace.
//
// # State vector
//
//   - [Handle.InitializeStateVector]: fills a device state vector with a standard initial state.
//
// [cuStateVec documentation]: https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html
package custatevec

// Code generated by gen-custatevec from custatevec.h. DO NOT EDIT.
