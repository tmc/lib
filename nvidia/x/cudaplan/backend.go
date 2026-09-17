package cudaplan

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// TargetBackend specifies the execution runtime backend for a region.
type TargetBackend uint8

const (
	// BackendLaunch dispatches kernels via direct cudalaunch calls.
	BackendLaunch TargetBackend = iota + 1

	// BackendGraph replays execution via instantiated cudagraph.Exec.
	BackendGraph

	// BackendCuBLASLt executes dense matrix multiplication using cublasltx.
	BackendCuBLASLt
)

func (b TargetBackend) String() string {
	switch b {
	case BackendLaunch:
		return "Launch"
	case BackendGraph:
		return "Graph"
	case BackendCuBLASLt:
		return "CuBLASLt"
	default:
		return fmt.Sprintf("Backend(%d)", uint8(b))
	}
}

// KernelRegistry maps kernel entry points or operation signatures to compiled artifacts.
type KernelRegistry struct {
	artifacts map[string]*cudakernel.Artifact
}

// NewKernelRegistry creates an empty KernelRegistry.
func NewKernelRegistry() *KernelRegistry {
	return &KernelRegistry{
		artifacts: make(map[string]*cudakernel.Artifact),
	}
}

// Register adds a compiled kernel artifact under the specified key.
func (r *KernelRegistry) Register(key string, art *cudakernel.Artifact) error {
	if key == "" {
		return errors.New("cudaplan: empty kernel key")
	}
	if art == nil {
		return errors.New("cudaplan: nil kernel artifact")
	}
	if err := art.Validate(); err != nil {
		return fmt.Errorf("cudaplan: invalid artifact for %q: %w", key, err)
	}
	r.artifacts[key] = art
	return nil
}

// Get returns the registered kernel artifact for key, or nil if not found.
func (r *KernelRegistry) Get(key string) *cudakernel.Artifact {
	return r.artifacts[key]
}

// StepPlan describes a single executable dispatch in the plan.
type StepPlan struct {
	NodeID          int
	NodeName        string
	Backend         TargetBackend
	Artifact        *cudakernel.Artifact
	InputNames      []string
	OutputName      string
	IsOutput        bool
	WorkspaceOffset int
	WorkspaceBytes  int
	Dependencies    []int // Step indices this step depends on
}

// SelectBackend chooses the execution backend based on operation characteristics and device capabilities.
func SelectBackend(opKind string, enableGraph bool) TargetBackend {
	if opKind == "matmul" {
		return BackendCuBLASLt
	}
	if enableGraph {
		return BackendGraph
	}
	return BackendLaunch
}
