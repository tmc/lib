package driver

import "sync"

// Profiler control lives in cudaProfiler.h, which the generated bindings do
// not cover, so these two symbols bind on first use instead of in
// registerFuncs.
var (
	profilerOnce    sync.Once
	profilerErr     error
	fnProfilerStart func() Result
	fnProfilerStop  func() Result
)

// bindProfiler resolves the profiler control symbols once, reporting the same
// errors as any other call in this package: the load error when the library is
// unavailable, or *SymbolError when it does not export them.
func bindProfiler() error {
	profilerOnce.Do(func() {
		if profilerErr = checkLib(); profilerErr != nil {
			return
		}
		for _, s := range []struct {
			fn  interface{}
			sym string
		}{
			{&fnProfilerStart, "cuProfilerStart"},
			{&fnProfilerStop, "cuProfilerStop"},
		} {
			addr, err := lookupLibrarySymbol(lib, s.sym)
			if err != nil || addr == 0 {
				profilerErr = &SymbolError{Symbol: s.sym}
				return
			}
			if !bindSymbol(s.fn, addr) {
				profilerErr = &SymbolError{Symbol: s.sym, Err: errArgumentSlots}
				return
			}
		}
	})
	return profilerErr
}

// ProfilerStart resumes collection by an attached external profiler after
// [ProfilerStop]. Collection starts on by default, so a program needs this
// only to trace a second region after stopping. It requires a current context.
//
// Wraps cuProfilerStart.
func ProfilerStart() error {
	if err := bindProfiler(); err != nil {
		return err
	}
	return toError(fnProfilerStart(), "cuProfilerStart")
}

// ProfilerStop flushes an attached external profiler's buffers and stops
// collection for the process. It requires a current context, and succeeds
// with nothing to do when no profiler is attached.
//
// A Go program must call this before it exits or a profiler that attaches
// through CUDA_INJECTION64_PATH, such as Nsight Systems, writes an empty
// report: the injected library flushes from a C atexit handler, and the Go
// runtime leaves through exit_group without running one. Programs that expect
// to be profiled stop next to their device teardown:
//
//	defer dev.WithContext(driver.ProfilerStop)
//
// Collection is process-wide, so work after it returns goes untraced until
// [ProfilerStart].
//
// Wraps cuProfilerStop.
func ProfilerStop() error {
	if err := bindProfiler(); err != nil {
		return err
	}
	return toError(fnProfilerStop(), "cuProfilerStop")
}
