//go:build linux

package main

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Instance-local operations let host controls exercise the owning loader.
// Allocation owners are returned even when a context restoration fails.
type cudaResidentAllocation interface {
	Len() int
	Free() error
}

type cudaResidentOperations struct {
	allocate func(int) (cudaResidentAllocation, error)
	stage    func(int) ([]byte, func() error, error)
	run      func(func() error) error
	copy     func(cudaResidentAllocation, int, []byte) error
	drain    func() error
	drop     func(int, uint64, uint64) error
}

func (r *cudaFoundation) residentOperations() cudaResidentOperations {
	if r.residentOps != nil {
		return *r.residentOps
	}
	return cudaResidentOperations{
		allocate: func(n int) (cudaResidentAllocation, error) {
			mem, err := cudamem.Alloc[byte](r.device, n)
			return mem, err
		},
		stage: func(n int) ([]byte, func() error, error) {
			h, err := cudamem.AllocHost[byte](n)
			if err != nil {
				return nil, h.Free, err
			}
			return h.Slice(), h.Free, nil
		},
		run: r.withContext,
		copy: func(allocation cudaResidentAllocation, offset int, src []byte) error {
			mem, ok := allocation.(cudamem.Slice[byte])
			if !ok {
				return fmt.Errorf("cuda: invalid resident allocation owner")
			}
			if offset < 0 || offset > mem.Len() || len(src) > mem.Len()-offset {
				return fmt.Errorf("cuda: invalid resident copy bounds")
			}
			return mem.Slice(offset, offset+len(src)).CopyFromHost(src)
		},
		drain: func() error { return r.withContext(func() error { return r.closeOperations().synchronizeContext() }) },
		drop:  posixFadviseDontNeed,
	}
}
