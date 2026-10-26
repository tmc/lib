// Package tcgen05 implements Blackwell (sm_100a, sm_103a, sm_100f, sm_103f)
// Tensor Memory (TMEM) Fifth-Generation Tensor Core (tcgen05) and Cluster
// Launch Control (CLC) kernel emitters, hardware descriptors, and GB300
// residency planners.
//
// Kernels emitted by this package route every PTX instruction through a
// self-contained device prelude so the same CUDA C++ source compiles for
// sm_100a/sm_103a with nvcc and executes on the host CPU via cudahostemu.
// Each tcgen05 kernel also produces a formal TraceEvent sequence verified by
// cudatmem.VerifyTrace.
package tcgen05
