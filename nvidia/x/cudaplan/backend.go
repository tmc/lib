package cudaplan

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
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

	// BackendMega executes an admitted FP32 region in a single-CTA worker or
	// a counted cooperative worker. LowerRegion retains dependency and workspace
	// metadata; this backend does not implement a persistent task scheduler.
	BackendMega
)

func (b TargetBackend) String() string {
	switch b {
	case BackendLaunch:
		return "Launch"
	case BackendGraph:
		return "Graph"
	case BackendCuBLASLt:
		return "CuBLASLt"
	case BackendMega:
		return "Mega"
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
	if r == nil {
		return errors.New("cudaplan: nil kernel registry")
	}
	if key == "" {
		return errors.New("cudaplan: empty kernel key")
	}
	if art == nil {
		return errors.New("cudaplan: nil kernel artifact")
	}
	if err := art.Validate(); err != nil {
		return fmt.Errorf("cudaplan: invalid artifact for %q: %w", key, err)
	}
	if r.artifacts == nil {
		r.artifacts = make(map[string]*cudakernel.Artifact)
	}
	r.artifacts[key] = art
	return nil
}

// Get returns the registered kernel artifact for key, or nil if not found.
func (r *KernelRegistry) Get(key string) *cudakernel.Artifact {
	if r == nil {
		return nil
	}
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

// BackendRequest contains the facts used to select one region's backend.
// Capability flags describe available entry points, not observed performance.
type BackendRequest struct {
	FiniteMegaRequested bool
	FiniteMegaAdmitted  bool
	// MegaCTAs is zero for the finite single-CTA worker, or 2..1024 for
	// counted execution. Runtime admission also requires measured residency.
	MegaCTAs        uint32
	Kind            cudair.NodeKind
	GraphRequested  bool
	GraphAvailable  bool
	VendorAvailable bool
	RequireArtifact bool
	Artifact        *cudakernel.Artifact
}

// BackendDecision records a backend and its admission reason.
type BackendDecision struct {
	Backend TargetBackend
	Reason  string
}

// SelectBackend applies the admission policy without opening a CUDA device.
// Mega execution requires a separately admitted task-body implementation and
// is never inferred from the operation's name or a task graph alone.
func SelectBackend(r BackendRequest) (BackendDecision, error) {
	if r.Kind < cudair.KindConstant || r.Kind > cudair.KindConcat {
		return BackendDecision{}, fmt.Errorf("%w: non-executable node kind %s", ErrUnsupportedBackend, r.Kind)
	}
	if r.MegaCTAs != 0 && (!r.FiniteMegaRequested || r.MegaCTAs < 2 || r.MegaCTAs > 1024) {
		return BackendDecision{}, fmt.Errorf("%w: invalid counted worker request", ErrUnsupportedBackend)
	}
	if r.FiniteMegaRequested {
		if r.GraphRequested || !r.FiniteMegaAdmitted || r.Artifact == nil || r.Kind == cudair.KindMatMul {
			return BackendDecision{}, fmt.Errorf("%w: finite worker region not admitted", ErrUnsupportedBackend)
		}
		if err := r.Artifact.Validate(); err != nil {
			return BackendDecision{}, fmt.Errorf("cudaplan: finite worker artifact: %w", err)
		}
		if r.MegaCTAs != 0 {
			return BackendDecision{BackendMega, fmt.Sprintf("counted FP32 region requested for %d cooperative CTAs", r.MegaCTAs)}, nil
		}
		return BackendDecision{BackendMega, "finite FP32 region admitted to one CTA"}, nil
	}
	if r.Kind == cudair.KindMatMul {
		if !r.VendorAvailable {
			return BackendDecision{}, fmt.Errorf("%w: cuBLASLt unavailable", ErrUnsupportedBackend)
		}
		if r.GraphRequested && !r.GraphAvailable {
			return BackendDecision{}, fmt.Errorf("%w: matmul stream capture unavailable", ErrUnsupportedBackend)
		}
		return BackendDecision{BackendCuBLASLt, "rank-2 matmul uses cuBLASLt"}, nil
	}
	if r.RequireArtifact && r.Artifact == nil {
		return BackendDecision{}, fmt.Errorf("%w: kernel artifact missing", ErrUnsupportedBackend)
	}
	if r.Artifact != nil {
		if err := r.Artifact.Validate(); err != nil {
			return BackendDecision{}, fmt.Errorf("cudaplan: backend artifact: %w", err)
		}
	}
	if r.GraphRequested {
		if !r.GraphAvailable {
			return BackendDecision{}, fmt.Errorf("%w: graph entry points unavailable", ErrUnsupportedBackend)
		}
		return BackendDecision{BackendGraph, "admitted kernel graph replay"}, nil
	}
	return BackendDecision{BackendLaunch, "ordinary kernel dispatch"}, nil
}
