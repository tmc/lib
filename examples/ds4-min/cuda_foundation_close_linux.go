//go:build linux

package main

import (
	"os"
	"syscall"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Operations are instance-local so fault controls cannot replace another
// session's driver. Ownership transitions remain in cudaFoundation.Close.
type cudaFoundationCloseOperations struct {
	run                func(*cudaFoundation, func() error) error
	synchronize        func(*cuda.Stream) error
	synchronizeContext func() error
	unload             func(driver.Module) error
	freeResident       func(*cudaFoundation) error
	unregister         func(uintptr) error
	unmap              func([]byte) error
	destroy            func(*cuda.Stream) error
	closeDevice        func(*cuda.Device) error
	closeFile          func(*os.File) error
}

func defaultCUDAFoundationCloseOperations() cudaFoundationCloseOperations {
	return cudaFoundationCloseOperations{
		run:                func(r *cudaFoundation, fn func() error) error { return r.device.WithContext(fn) },
		synchronize:        syncStream,
		synchronizeContext: driver.CtxSynchronize,
		unload:             func(m driver.Module) error { return m.Close() },
		freeResident:       func(r *cudaFoundation) error { return r.resident.Free() },
		unregister:         driver.MemHostUnregister,
		unmap:              syscall.Munmap,
		destroy:            func(s *cuda.Stream) error { return s.Close() },
		closeDevice:        func(d *cuda.Device) error { return d.Close() },
		closeFile:          func(f *os.File) error { return f.Close() },
	}
}

func (r *cudaFoundation) closeOperations() cudaFoundationCloseOperations {
	if r.closeOps != nil {
		return *r.closeOps
	}
	return defaultCUDAFoundationCloseOperations()
}
