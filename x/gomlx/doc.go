// Package gomlx implements the GoMLX compute interfaces using TMC's PJRT
// runtime and StableHLO builder. Loading and execution do not require cgo.
//
// The current backend supports static F32, Int32, and Int64 tensors on its first addressable
// device. Supported operations include constants, parameters, addition,
// multiplication, broadcasting, reshape, sum reduction, and general dot
// products with default configuration. Unsupported operations return
// compute.ErrNotImplemented. Full GoMLX backend conformance is not established.
//
// Donated inputs become invalid after execution; their native storage is not
// reused. Close releases live child resources and reports cleanup errors.
// Finalize satisfies compute.Backend but cannot return those errors.
package gomlx
