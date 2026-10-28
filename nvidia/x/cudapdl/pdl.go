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
	// Source and Artifact.Raw do not establish synchronization, effect, or
	// resource support.
	Source string

	// IsVendor indicates whether the kernel is an opaque vendor library kernel
	// (e.g. cuBLASLt, cuDNN, or vendor CUBIN).
	IsVendor bool

	// HasLaunchDependents is the kernel owner's declaration that the kernel
	// triggers dependents via griddepcontrol.launch_dependents.
	HasLaunchDependents bool

	// HasWait is the kernel owner's declaration that the kernel waits on
	// dependencies via griddepcontrol.wait.
	HasWait bool

	// HasIndependentWork is the kernel owner's declaration that a consumer
	// kernel executes producer-independent BeforeWait preamble work prior to
	// griddepcontrol.wait. Must agree with Effects.BeforeWait and
	// Effects.WaitPlacement.
	HasIndependentWork bool

	// Effects describes the owner-supplied phase effects and barrier placement
	// for the kernel. CheckEligibility requires bounded phase effect
	// declarations on both producer and consumer kernels.
	Effects KernelEffects

	// Resources specifies hardware resource requirements for the kernel.
	// When nil and Artifact is non-nil, CheckEligibility uses Artifact's
	// non-zero ResourceRequirements; when both are supplied, they must match.
	Resources *cudakernel.ResourceRequirements

	// ThreadsPerBlock specifies the launch CTA thread count used to validate
	// warp and register budgets against the probed target SM limits. When zero
	// and Artifact is non-nil, CheckEligibility uses
	// Artifact.JITOptions.ThreadsPerBlock; when both are non-zero, they must match.
	ThreadsPerBlock uint32
}

// Eligibility reports the planning policy for a declared producer/consumer
// pair. It checks the target floor, probed target resource limits, supplied
// artifact target identities, vendor opacity, owner synchronization
// declarations, bounded phase effects (including preamble hazard absence), and
// per-CTA/combined resource constraints.
//
// Eligibility establishes ProvenanceOwnerDeclared for admitted pairs; it does
// not establish SASS instruction placement verification, native device
// execution qualification, or runtime CTA overlap guarantees.
type Eligibility struct {
	// Eligible is true if the pair passes all planning, effect, and resource checks.
	Eligible bool

	// Policy is PolicyProgrammatic if Eligible, or PolicyStandardFallback otherwise.
	Policy LaunchPolicy

	// FallbackReason describes the exact reason why PDL was rejected, or is empty if eligible.
	FallbackReason string

	// TargetArch is the parsed target compute architecture (e.g. "sm_90").
	TargetArch string

	// Provenance records the evidence level of the eligibility decision.
	// Admitted pairs carry ProvenanceOwnerDeclared; rejected pairs carry ProvenanceNone.
	Provenance EvidenceProvenance

	// ConsumerHasIndependentPreamble is true when the admitted consumer declares
	// verified producer-independent BeforeWait preamble effects prior to
	// griddepcontrol.wait, and false for entry-synchronizing consumers.
	ConsumerHasIndependentPreamble bool

	// OverlapGuaranteed is always false for static/owner-declared planning:
	// sum-of-resources arithmetic and programmatic graph edges allow overlap when
	// hardware scheduling permits, but never guarantee runtime overlap.
	OverlapGuaranteed bool

	// NoOverlapStillCorrect is true when the admitted pair preserves semantic
	// correctness even if zero CTA overlap occurs at runtime.
	NoOverlapStillCorrect bool

	// Resources summarizes the resolved per-CTA and combined producer/consumer
	// resource footprint for admitted pairs.
	Resources ResourceSummary
}

// CheckEligibility evaluates the planning policy for a target and a declared
// producer/consumer pair. An empty target is inferred from a supplied artifact
// target. Every supplied artifact must declare that same target; this exact
// identity policy is conservative and does not model general CUDA binary
// compatibility. A pair without artifacts is a planning description, not a
// qualified executable.
func CheckEligibility(target string, producer, consumer KernelDesc) Eligibility {
	arch := strings.ToLower(strings.TrimSpace(target))
	if arch == "" {
		if pTarget := artifactTarget(producer.Artifact); pTarget != "" {
			arch = strings.ToLower(strings.TrimSpace(pTarget))
		}
	}
	if arch == "" {
		if cTarget := artifactTarget(consumer.Artifact); cTarget != "" {
			arch = strings.ToLower(strings.TrimSpace(cTarget))
		}
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
		pName := kernelRoleName(producer.Name, "producer")
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("opaque vendor kernel %q is ineligible for programmatic dependent launch", pName),
			TargetArch:     info.Arch,
		}
	}
	if consumer.IsVendor {
		cName := kernelRoleName(consumer.Name, "consumer")
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("opaque vendor kernel %q is ineligible for programmatic dependent launch", cName),
			TargetArch:     info.Arch,
		}
	}

	// Producer must emit griddepcontrol.launch_dependents.
	if !producer.HasLaunchDependents {
		pName := kernelRoleName(producer.Name, "producer")
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("producer kernel %q missing required trigger %s", pName, PTXLaunchDependents),
			TargetArch:     info.Arch,
		}
	}

	// Consumer must emit griddepcontrol.wait.
	if !consumer.HasWait {
		cName := kernelRoleName(consumer.Name, "consumer")
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: fmt.Sprintf("consumer kernel %q missing required barrier %s", cName, PTXWait),
			TargetArch:     info.Arch,
		}
	}

	summary, hasIndepPreamble, reason := validateEffectsAndResources(info, producer, consumer)
	if reason != "" {
		return Eligibility{
			Eligible:       false,
			Policy:         PolicyStandardFallback,
			FallbackReason: reason,
			TargetArch:     info.Arch,
			Resources:      summary,
		}
	}

	return Eligibility{
		Eligible:                       true,
		Policy:                         PolicyProgrammatic,
		FallbackReason:                 "",
		TargetArch:                     info.Arch,
		Provenance:                     ProvenanceOwnerDeclared,
		ConsumerHasIndependentPreamble: hasIndepPreamble,
		OverlapGuaranteed:              false,
		NoOverlapStillCorrect:          true,
		Resources:                      summary,
	}
}

func artifactTarget(artifact *cudakernel.Artifact) string {
	if artifact == nil {
		return ""
	}
	return artifact.Target()
}

func artifactResources(artifact *cudakernel.Artifact) (cudakernel.ResourceRequirements, bool) {
	if artifact == nil {
		return cudakernel.ResourceRequirements{}, false
	}
	res := artifact.Resources()
	return res, res != (cudakernel.ResourceRequirements{})
}

func artifactThreadsPerBlock(artifact *cudakernel.Artifact) uint32 {
	if artifact == nil {
		return 0
	}
	return artifact.JITOptions().ThreadsPerBlock
}

func checkArtifactTarget(artifact *cudakernel.Artifact, target string) error {
	if artifact == nil {
		return nil
	}
	rawTarget := artifactTarget(artifact)
	arch := strings.ToLower(strings.TrimSpace(rawTarget))
	if arch == "" {
		return errors.New("missing target architecture")
	}
	info, err := cudakernel.TargetSM(arch)
	if err != nil {
		return fmt.Errorf("invalid target architecture %q: %w", rawTarget, err)
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
//
// Subsequent updates on an instantiated executable graph containing producer
// or consumer nodes remain governed by cudagraph's existing node-update
// constraints (such as SetKernelNodeGrid, SetKernelNodeLaunch, UpdateKernelNode,
// and HotSwapNode parameter-layout and function-compatibility rules). Edge
// dependency policy and graph topology are fixed at graph instantiation time.
type Composition struct {
	Policy         LaunchPolicy
	FallbackReason string
	Producer       *cudagraph.Node
	Consumer       *cudagraph.Node
	Provenance     EvidenceProvenance
}

// Compose connects producer and consumer in graph g according to eligibility.
// If el.Eligible is true and carries ProvenanceOwnerDeclared from CheckEligibility,
// g.AddProgrammaticDependency(producer, consumer) is established.
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

	if el.Eligible && (el.Policy != PolicyProgrammatic || el.Provenance != ProvenanceOwnerDeclared || !el.NoOverlapStillCorrect) {
		el.Eligible = false
		el.Policy = PolicyStandardFallback
		if el.FallbackReason == "" {
			el.FallbackReason = "eligibility was not established by CheckEligibility"
		}
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
				Provenance:     ProvenanceNone,
			}, nil
		}
		return Composition{
			Policy:         PolicyProgrammatic,
			FallbackReason: "",
			Producer:       producer,
			Consumer:       consumer,
			Provenance:     el.Provenance,
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
		Provenance:     ProvenanceNone,
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
	if a == nil {
		return CheckEligibility("", producer, consumer)
	}
	return CheckEligibility(a.TargetArch, producer, consumer)
}

// Apply composes the dependency edge in g.
func (a *PlanAdapter) Apply(g *cudagraph.Graph, producer, consumer *cudagraph.Node, producerDesc, consumerDesc KernelDesc) (Composition, error) {
	el := a.Evaluate(producerDesc, consumerDesc)
	return Compose(g, producer, consumer, el)
}
