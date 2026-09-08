// Package gomlx implements the GoMLX compute interfaces using TMC's PJRT
// runtime and StableHLO builder. Loading and execution do not require cgo.
//
// The backend supports static Bool, Float32, Float64, Int32, Int64, Uint32, and Uint64
// tensors on its first addressable device. [Backend.Capabilities] lists supported
// operations; operation support varies by dtype. Float64 transport and constants
// preserve exact bits, but transcendental operations currently require Float32.
// Random generation uses Philox with an explicit [3]uint64 state.
// Unsupported operations return compute.ErrNotImplemented. Full GoMLX backend
// conformance is not established.
//
// Donated inputs become invalid after execution; their native storage is not
// reused. Close releases live child resources and reports cleanup errors.
// Finalize satisfies compute.Backend but cannot return those errors.
package gomlx
