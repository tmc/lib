// Package cudacollective defines collective communication plans, rank
// membership, communicator generations, multi-transport fallback resolution,
// and double-buffered pipelined transfers for distributed CUDA execution.
//
// Communicators are created with a Membership and an explicit generation
// counter. Multi-transport fallback resolves along the hierarchy:
//
//	nvlshm (NVLink / NVSwitch P2P / multicast)
//	  -> moonep (RDMA / Eager transport)
//	  -> nccltest / TCP socket fallback
//
// In environments lacking physical devices or libraries, execution fails closed
// with ErrExecutionUnavailable or ErrUnsupportedBackend rather than pretending success.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudacollective
