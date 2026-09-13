package main

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/ebitengine/purego"
)

// stopCUDAProfiler calls the CUDA driver's cuProfilerStop, which is what
// makes a profiler flush its buffered activity records. The capture's own
// diagnostics say so: "Buffers holding CUDA trace data will be flushed on
// CudaProfilerStop() call". Nothing in a Go program makes that call, and
// nothing makes it on the way out either, because runtime.exit on
// linux/arm64 issues exit_group directly without running libc teardown.
// Whatever the profiler has buffered when the process leaves is discarded,
// which is why the tail of every capture was missing and decode, the last
// 0.17 s of the run, never appeared on the device timeline at all.
//
// cuProfilerStop operates on the calling thread's current context, and a
// goroutine does not in general run on the thread that established one, so
// the thread is pinned and the device's primary context made current first.
// That is the same context the runtime API uses, so stopping on it stops the
// profiling that covered the run.
//
// Call this after the last kernel and after the last NVTX range closes.
func stopCUDAProfiler() error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	lib, err := purego.Dlopen("libcuda.so.1", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return fmt.Errorf("dlopen libcuda.so.1: %w", err)
	}
	sym := func(name string) (uintptr, error) {
		p, err := purego.Dlsym(lib, name)
		if err != nil {
			return 0, fmt.Errorf("dlsym %s: %w", name, err)
		}
		return p, nil
	}
	fnCtxGetCurrent, err := sym("cuCtxGetCurrent")
	if err != nil {
		return err
	}
	fnCtxSetCurrent, err := sym("cuCtxSetCurrent")
	if err != nil {
		return err
	}
	fnPrimaryCtxRetain, err := sym("cuDevicePrimaryCtxRetain")
	if err != nil {
		return err
	}
	fnProfilerStop, err := sym("cuProfilerStop")
	if err != nil {
		return err
	}

	var ctx uintptr
	if r, _, _ := purego.SyscallN(fnCtxGetCurrent, uintptr(unsafe.Pointer(&ctx))); r != 0 {
		return fmt.Errorf("cuCtxGetCurrent: CUresult %d", r)
	}
	if ctx == 0 {
		// Device 0: ds4-min drives a single device.
		if r, _, _ := purego.SyscallN(fnPrimaryCtxRetain, uintptr(unsafe.Pointer(&ctx)), 0); r != 0 {
			return fmt.Errorf("cuDevicePrimaryCtxRetain: CUresult %d", r)
		}
		if r, _, _ := purego.SyscallN(fnCtxSetCurrent, ctx); r != 0 {
			return fmt.Errorf("cuCtxSetCurrent: CUresult %d", r)
		}
	}
	if r, _, _ := purego.SyscallN(fnProfilerStop); r != 0 {
		return fmt.Errorf("cuProfilerStop: CUresult %d", r)
	}
	return nil
}
