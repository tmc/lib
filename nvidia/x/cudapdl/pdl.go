package cudapdl

import (
	"bytes"
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

	// Artifact is the optional compiled kernel artifact descriptor.
	Artifact *cudakernel.Artifact

	// Source is optional PTX or CUDA C++ source code text.
	Source string

	// IsVendor indicates whether the kernel is an opaque vendor library kernel
	// (e.g. cuBLASLt, cuDNN, or vendor CUBIN).
	IsVendor bool

	// HasLaunchDependents explicitly indicates whether the kernel triggers dependents.
	// If false, Source and Artifact.Raw are inspected for "griddepcontrol.launch_dependents".
	HasLaunchDependents bool

	// HasWait explicitly indicates whether the kernel waits on dependencies.
	// If false, Source and Artifact.Raw are inspected for "griddepcontrol.wait".
	HasWait bool

	// PreambleBytes or IndependentWork indicates the presence of independent preamble work.
	HasIndependentWork bool
}

// Eligibility reports whether a producer/consumer kernel pair on a target device
// meets all hardware and software criteria for Programmatic Dependent Launch.
type Eligibility struct {
	// Eligible is true if both kernels and the target hardware support PDL.
	Eligible bool

	// Policy is PolicyProgrammatic if Eligible, or PolicyStandardFallback otherwise.
	Policy LaunchPolicy

	// FallbackReason describes the exact reason why PDL was rejected, or is empty if eligible.
	FallbackReason string

	// TargetArch is the parsed target compute architecture (e.g. "sm_90").
	TargetArch string
}

// CheckEligibility evaluates whether a target architecture and a producer/consumer pair
// qualify for Programmatic Dependent Launch.
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
	if !hasLaunchDependents(producer) {
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
	if !hasWait(consumer) {
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

func hasLaunchDependents(k KernelDesc) bool {
	if k.HasLaunchDependents {
		return true
	}
	if strings.Contains(k.Source, PTXLaunchDependents) {
		return true
	}
	if k.Artifact != nil && len(k.Artifact.Raw) > 0 {
		if bytes.Contains(k.Artifact.Raw, []byte(PTXLaunchDependents)) {
			return true
		}
	}
	return false
}

func hasWait(k KernelDesc) bool {
	if k.HasWait {
		return true
	}
	if strings.Contains(k.Source, PTXWait) {
		return true
	}
	if k.Artifact != nil && len(k.Artifact.Raw) > 0 {
		if bytes.Contains(k.Artifact.Raw, []byte(PTXWait)) {
			return true
		}
	}
	return false
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
