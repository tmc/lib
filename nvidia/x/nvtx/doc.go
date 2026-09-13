// Package nvtx provides a pure-Go, zero-cgo implementation of NVIDIA Tools Extension (NVTX v3).
//
// NVTX v3 is a header-only instrumentation protocol. When running under profilers such
// as NVIDIA Nsight Systems (nsys) or Nsight Compute (ncu), the profiler sets the
// NVTX_INJECTION64_PATH environment variable to inject the profiling library.
//
// When NVTX_INJECTION64_PATH is unset, calls are safe no-ops without a native
// provider invocation.
// When set, the injection provider is dynamically loaded via purego, enabling high-resolution
// timeline markers, nested ranges, domains, CUDA resource naming, and thread naming for Nsight timelines.
package nvtx
