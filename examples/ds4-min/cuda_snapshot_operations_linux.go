//go:build linux

package main

import "github.com/tmc/lib/nvidia/x/cudamem"

type cudaSnapshotOperations struct {
	synchronize func(*cudaRuntime) error
	arena       func(*cudaRuntime) (*cudamem.Arena, error)
	clone       func(*cudamem.Arena, cudaSessionState) (cudaSessionState, error)
	arenaError  func(*cudamem.Arena) error
	free        func(*cudamem.Arena) error
}

func defaultCUDASnapshotOperations() cudaSnapshotOperations {
	return cudaSnapshotOperations{
		synchronize: func(r *cudaRuntime) error { return syncStream(r.stream()) },
		arena:       func(r *cudaRuntime) (*cudamem.Arena, error) { return cudamem.NewArena(r.device), nil },
		clone:       cloneCUDAState,
		arenaError:  func(a *cudamem.Arena) error { return a.Err() },
		free:        func(a *cudamem.Arena) error { return a.Free() },
	}
}
