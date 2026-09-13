package main

import (
	"fmt"

	"github.com/ebitengine/purego"
)

// stopCUDAProfiler calls the CUDA driver's cuProfilerStop on the calling
// thread's current context. Under Nsight Systems that call is what flushes
// CUPTI's buffered activity records: the capture's own diagnostics say
// "Buffers holding CUDA trace data will be flushed on CudaProfilerStop()
// call". Nothing in a Go program makes that call on its own, and nothing
// makes it on the way out either, because runtime.exit on linux/arm64 issues
// exit_group directly without running libc teardown. Whatever CUPTI has
// buffered at that point is lost, which is why the tail of every capture —
// including all of decode — has been missing.
//
// The context must still be alive, so call this after the last kernel and
// before tearing the session down.
func stopCUDAProfiler() error {
	lib, err := purego.Dlopen("libcuda.so.1", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("dlopen libcuda.so.1: %w", err)
	}
	sym, err := purego.Dlsym(lib, "cuProfilerStop")
	if err != nil {
		return fmt.Errorf("dlsym cuProfilerStop: %w", err)
	}
	if r, _, _ := purego.SyscallN(sym); r != 0 {
		return fmt.Errorf("cuProfilerStop: CUresult %d", r)
	}
	return nil
}
