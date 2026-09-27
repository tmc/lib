// Package cudamega provides persistent grid execution, cooperative SM task queues,
// and mega-kernel runtime dispatching.

// Instead of dispatching hundreds of short-lived individual CUDA kernels from the host CPU,
// cudamega launches a single persistent grid of thread blocks that stay resident on GPU SMs,
// popping dynamic work items from a ring buffer queue in device memory with atomic synchronization.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudamega
