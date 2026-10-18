package cudapdl

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// LaunchPolicy defines the graph edge dependency policy applied to a producer/consumer pair.
type LaunchPolicy uint8

const (
	// PolicyProgrammatic creates a programmatic dependent launch edge (allowing early launch
	// and independent preamble overlap).
	PolicyProgrammatic LaunchPolicy = iota + 1

	// PolicyStandardFallback creates a standard full-completion dependency edge.
	PolicyStandardFallback
)

func (p LaunchPolicy) String() string {
	switch p {
	case PolicyProgrammatic:
		return "PolicyProgrammatic"
	case PolicyStandardFallback:
		return "PolicyStandardFallback"
	default:
		return fmt.Sprintf("LaunchPolicy(%d)", p)
	}
}

// PTX intrinsics for programmatic dependent launch synchronization.
const (
	PTXLaunchDependents = "griddepcontrol.launch_dependents"
	PTXWait             = "griddepcontrol.wait"
)

// KernelDesc describes a candidate kernel for PDL participation.
type KernelDesc struct {
	// Name identifies the kernel for diagnostics.
	Name string

	// Artifact is the optional compiled kernel artifact descriptor. Its target
	// must match the selected target exactly after architecture normalization.
	Artifact *cudakernel.Artifact

	// Source is optional PTX or CUDA C++ source code text for diagnostics.
	// Source and Artifact.Raw do not establish synchronization support.
	Source string

	// IsVendor indicates whether the kernel is an opaque vendor library kernel
	// (e.g. cuBLASLt, cuDNN, or vendor CUBIN).
	IsVendor bool

	// HasLaunchDependents is the kernel owner's declaration that the kernel
	// triggers dependents. CheckEligibility does not verify trigger placement
	// or execution; those require separate qualification.
	HasLaunchDependents bool

	// HasWait is the kernel owner's declaration that the kernel waits on
	// dependencies. CheckEligibility does not verify wait placement or execution.
	HasWait bool

	// HasIndependentWork records independent preamble work; CheckEligibility checks only target SM, artifact target match, IsVendor, HasLaunchDependents, and HasWait.
	HasIndependentWork bool
}

// Eligibility reports the planning policy for a declared producer/consumer
// pair. It checks the target floor, supplied artifact target identities,
// vendor opacity and owner declarations. It does not establish trigger/wait
// placement, effect/resource safety or native execution qualification.
type Eligibility struct {
	// Eligible is true if the pair passes the planning checks.
	Eligible bool

	// Policy is PolicyProgrammatic if Eligible, or PolicyStandardFallback otherwise.
	Policy LaunchPolicy

	// FallbackReason describes the exact reason why PDL was rejected, or is empty if eligible.
	FallbackReason string

	// TargetArch is the parsed target compute architecture (e.g. "sm_90").
	TargetArch string
}

// CheckEligibility evaluates the planning policy for a target and a declared
// producer/consumer pair. An empty target is inferred from a supplied artifact
// target. Every supplied artifact must declare that same target; this exact
// identity policy is conservative and does not model general CUDA binary
// compatibility. A pair without artifacts is a planning description, not a
// qualified executable.
func CheckEligibility(target string, producer, consumer KernelDesc) Eligibility {
	arch := strings.ToLower(strings.TrimSpace(target))
	if arch == "" && producer.Artifact != nil && producer.Artifact.Target != "" {
		arch = strings.ToLower(strings.TrimSpace(producer.Artifact.Target))
	}
	if arch == "" && consumer.Artifact != nil && consumer.Artifact.Target != "" {
		arch = strings.ToLower(strings.TrimSpace(consumer.Artifact.Target))
	}

	info, err := cudakernel.TargetSM(arch)
	if err != nil {
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("invalid target architecture %q: %v", target, err),
			TargetArch:     arch,
		}
	}

	// PDL hardware fencing: requires compute capability >= 9.0 (sm_90, sm_90a, sm_100, sm_100a, sm_121, sm_121a).
	if info.Major < 9 {
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("target architecture %s (compute capability %d.%d) does not support programmatic dependent launch: sm_90 or higher required", info.Arch, info.Major, info.Minor),
			TargetArch:     info.Arch,
		}
	}
	for _, kernel := range []struct {
		role     string
		artifact *cudakernel.Artifact
	}{{"producer", producer.Artifact}, {"consumer", consumer.Artifact}} {
		if err := checkArtifactTarget(kernel.artifact, info.Arch); err != nil {
			return Eligibility{
				Policy:         PolicyStandardFallback,
				FallbackReason: fmt.Sprintf("%s artifact: %v", kernel.role, err),
				TargetArch:     info.Arch,
			}
		}
	}

	// Check vendor kernel opacity: opaque vendor kernels must be rejected until proven.
	if producer.IsVendor {
		pName := producer.Name
		if pName == "" {
			pName = "producer"
		}
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("opaque vendor kernel %q is ineligible for programmatic dependent launch", pName),
			TargetArch:     info.Arch,
		}
	}
	if consumer.IsVendor {
		cName := consumer.Name
		if cName == "" {
			cName = "consumer"
		}
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("opaque vendor kernel %q is ineligible for programmatic dependent launch", cName),
			TargetArch:     info.Arch,
		}
	}

	// Producer must emit griddepcontrol.launch_dependents
	if !producer.HasLaunchDependents {
		pName := producer.Name
		if pName == "" {
			pName = "producer"
		}
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("producer kernel %q missing required trigger %s", pName, PTXLaunchDependents),
			TargetArch:     info.Arch,
		}
	}

	// Consumer must emit griddepcontrol.wait
	if !consumer.HasWait {
		cName := consumer.Name
		if cName == "" {
			cName = "consumer"
		}
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("consumer kernel %q missing required barrier %s", cName, PTXWait),
			TargetArch:     info.Arch,
		}
	}

	return Eligibility{
		Eligible:       true,
		Policy:         PolicyProgrammatic,
		FallbackReason: "",
		TargetArch:     info.Arch,
	}
}

func checkArtifactTarget(artifact *cudakernel.Artifact, target string) error {
	if artifact == nil {
		return nil
	}
	arch := strings.ToLower(strings.TrimSpace(artifact.Target))
	if arch == "" {
		return errors.New("missing target architecture")
	}
	info, err := cudakernel.TargetSM(arch)
	if err != nil {
		return fmt.Errorf("invalid target architecture %q: %w", artifact.Target, err)
	}
	if info.Major < 9 {
		return fmt.Errorf("target %s does not support PDL: sm_90 or higher required", info.Arch)
	}
	if info.Arch != target {
		return fmt.Errorf("target %s differs from selected target %s", info.Arch, target)
	}
	return nil
}

// Composition describes the applied graph dependency edge between producer and consumer nodes.
type Composition struct {
	Policy         LaunchPolicy
	FallbackReason string
	Producer       *cudagraph.Node
	Consumer       *cudagraph.Node
}

// Compose connects producer and consumer in graph g according to eligibility.
// If el.Eligible is true, g.AddProgrammaticDependency(producer, consumer) is established.
// Otherwise, Compose gracefully falls back to g.AddDependency(producer, consumer) and
// records the fallback reason, preserving execution correctness regardless of whether
// hardware overlap occurs.
func Compose(g *cudagraph.Graph, producer, consumer *cudagraph.Node, el Eligibility) (Composition, error) {
	if g == nil || g.Raw() == 0 {
		return Composition{}, errors.New("cudapdl: nil or closed graph")
	}
	if producer == nil || producer.Raw() == 0 {
		return Composition{}, errors.New("cudapdl: nil or closed producer node")
	}
	if consumer == nil || consumer.Raw() == 0 {
		return Composition{}, errors.New("cudapdl: nil or closed consumer node")
	}

	if el.Eligible {
		if err := g.AddProgrammaticDependency(producer, consumer); err != nil {
			// If driver rejects programmatic edge at runtime, gracefully fallback to standard dependency
			if fallbackErr := g.AddDependency(producer, consumer); fallbackErr != nil {
				return Composition{}, fmt.Errorf("cudapdl: add programmatic dependency failed (%w), and fallback failed: %w", err, fallbackErr)
			}
			return Composition{
				Policy:         PolicyStandardFallback,
				FallbackReason: fmt.Sprintf("driver rejected programmatic dependency: %v", err),
				Producer:       producer,
				Consumer:       consumer,
			}, nil
		}
		return Composition{
			Policy:         PolicyProgrammatic,
			FallbackReason: "",
			Producer:       producer,
			Consumer:       consumer,
		}, nil
	}

	// Standard fallback
	if err := g.AddDependency(producer, consumer); err != nil {
		return Composition{}, fmt.Errorf("cudapdl: add fallback dependency: %w", err)
	}

	return Composition{
		Policy:         PolicyStandardFallback,
		FallbackReason: el.FallbackReason,
		Producer:       producer,
		Consumer:       consumer,
	}, nil
}

// PlanAdapter provides an adapter interface for planning engines (e.g. cudaplan)
// to inspect and apply PDL composition without reciprocal import dependencies.
type PlanAdapter struct {
	TargetArch string
}

// NewPlanAdapter creates a PlanAdapter for targetArch.
func NewPlanAdapter(targetArch string) *PlanAdapter {
	return &PlanAdapter{TargetArch: targetArch}
}

// Evaluate checks eligibility for candidate producer and consumer kernel descriptors.
func (a *PlanAdapter) Evaluate(producer, consumer KernelDesc) Eligibility {
	return CheckEligibility(a.TargetArch, producer, consumer)
}

// Apply composes the dependency edge in g.
func (a *PlanAdapter) Apply(g *cudagraph.Graph, producer, consumer *cudagraph.Node, producerDesc, consumerDesc KernelDesc) (Composition, error) {
	el := a.Evaluate(producerDesc, consumerDesc)
	return Compose(g, producer, consumer, el)
}
