package main

// Go-side profiling for ds4-min.
//
// The port's throughput gap was attributed to host-side cost without the
// host ever being profiled. These flags are the cheapest instrument that
// can settle that: -cpuprofile says where Go time goes, -trace says
// whether the goroutine is running or blocked in a GPU wait, and -spans
// hands the same run's phase structure to gputrace.

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
)

// A profiler holds whatever profiling a run turned on. The zero value
// records nothing and its Stop is a no-op.
type profiler struct {
	cpu     *os.File
	trace   *os.File
	memPath string
}

// startProfiling turns on the profiling cfg asks for. Profiling starts
// after the model is loaded, so the 20-to-40 second mmap residency wait
// does not dominate every profile of a 16-token generation.
func startProfiling(cfg config) (*profiler, error) {
	p := &profiler{memPath: cfg.memProfile}
	if cfg.cpuProfile != "" {
		f, err := os.Create(cfg.cpuProfile)
		if err != nil {
			return nil, fmt.Errorf("create cpu profile: %w", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			return nil, errors.Join(fmt.Errorf("start cpu profile: %w", err), f.Close())
		}
		p.cpu = f
	}
	if cfg.traceProfile != "" {
		f, err := os.Create(cfg.traceProfile)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("create trace: %w", err), p.Stop())
		}
		if err := trace.Start(f); err != nil {
			return nil, errors.Join(fmt.Errorf("start trace: %w", err), f.Close(), p.Stop())
		}
		p.trace = f
	}
	return p, nil
}

// Stop ends every active profile and writes the heap profile last, so it
// describes the run rather than the teardown.
func (p *profiler) Stop() error {
	if p == nil {
		return nil
	}
	var errs []error
	if p.trace != nil {
		trace.Stop()
		errs = append(errs, p.trace.Close())
		p.trace = nil
	}
	if p.cpu != nil {
		pprof.StopCPUProfile()
		errs = append(errs, p.cpu.Close())
		p.cpu = nil
	}
	if p.memPath != "" {
		f, err := os.Create(p.memPath)
		if err != nil {
			errs = append(errs, fmt.Errorf("create mem profile: %w", err))
		} else {
			runtime.GC()
			errs = append(errs, pprof.WriteHeapProfile(f), f.Close())
		}
		p.memPath = ""
	}
	return errors.Join(errs...)
}
