//go:build linux

package main

import "github.com/tmc/lib/nvidia/x/cudamem"

type cudaRuntimeConstructorOperations struct {
	foundation func(string, cudaRuntimeOptions) (*cudaFoundation, error)
	bind       func(*cudaFoundation) (*cudaModel, error)
	moe        func(*cudaRuntime, int) error
	dense      func(*cudaRuntime, int) error
	ballast    func(*cudaRuntime) error
	allocate   func(*cudaRuntime) error
	hc         func(*cudaFoundation) (*cudaHC, error)
	reset      func(*cudaRuntime) error
	cleanup    *cudaChildCleanupOperations
}

func defaultCUDARuntimeConstructorOperations() cudaRuntimeConstructorOperations {
	return cudaRuntimeConstructorOperations{
		foundation: newCUDAFoundationWithOptions,
		bind:       func(f *cudaFoundation) (*cudaModel, error) { return f.bindModel() },
		moe:        func(r *cudaRuntime, n int) error { return r.buildMoEArtifacts(n) },
		dense:      func(r *cudaRuntime, n int) error { return r.buildDenseArtifacts(n) },
		ballast:    func(r *cudaRuntime) error { return r.allocBallast() },
		allocate:   func(r *cudaRuntime) error { r.arena = cudamem.NewArena(r.device); r.allocState(); return r.arena.Err() },
		hc:         newCUDAHC,
		reset:      func(r *cudaRuntime) error { return r.Reset() },
	}
}
