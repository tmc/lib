package cudapdl

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// MaxPhaseEffects is the maximum number of PhaseEffect entries permitted in a
// single execution phase slice of KernelEffects.
const MaxPhaseEffects = 64

// maxRegistersPerSM is the hardware 32-bit register file capacity per SM on
// supported sm_90, sm_100, and sm_121 architectures.
const maxRegistersPerSM = 65536

// maxTMEMBytesPerSM is the hardware Tensor Memory capacity per SM (256 KiB) on
// architectures that support TMEM (e.g. sm_100a).
const maxTMEMBytesPerSM = 256 * 1024

// maxThreadsPerCTA is the architectural thread limit per CTA on sm_90+.
const maxThreadsPerCTA = 1024

// maxRegistersPerThread is the architectural register limit per thread on sm_90+.
const maxRegistersPerThread = 255

// EvidenceProvenance identifies the evidence boundary backing a planning
// eligibility result.
type EvidenceProvenance uint8

const (
	// ProvenanceNone indicates no verified planning eligibility provenance.
	ProvenanceNone EvidenceProvenance = iota

	// ProvenanceOwnerDeclared indicates that CheckEligibility validated
	// owner-declared phase effects, synchronization placement assertions, and
	// static target/artifact resource bounds on the host. It does not establish
	// observed native execution, verified SASS instruction ordering, or runtime
	// CTA overlap.
	ProvenanceOwnerDeclared
)

func (p EvidenceProvenance) String() string {
	switch p {
	case ProvenanceNone:
		return "ProvenanceNone"
	case ProvenanceOwnerDeclared:
		return "ProvenanceOwnerDeclared"
	default:
		return fmt.Sprintf("EvidenceProvenance(%d)", p)
	}
}

// TriggerPlacement describes the owner-declared position of
// griddepcontrol.launch_dependents relative to producer memory effects.
type TriggerPlacement uint8

const (
	// TriggerPlacementUnknown indicates that producer trigger placement has not
	// been declared or verified. Pairs with unknown placement are ineligible.
	TriggerPlacementUnknown TriggerPlacement = iota

	// TriggerAfterSharedWrites declares that the producer completes and fences
	// all consumer-visible handoff writes in BeforeLaunchDependents prior to
	// executing griddepcontrol.launch_dependents.
	TriggerAfterSharedWrites

	// TriggerBeforeSharedWritesComplete declares that the producer emits
	// griddepcontrol.launch_dependents before consumer-visible handoff writes
	// have completed. Pairs with this placement are ineligible.
	TriggerBeforeSharedWritesComplete
)

func (p TriggerPlacement) String() string {
	switch p {
	case TriggerPlacementUnknown:
		return "TriggerPlacementUnknown"
	case TriggerAfterSharedWrites:
		return "TriggerAfterSharedWrites"
	case TriggerBeforeSharedWritesComplete:
		return "TriggerBeforeSharedWritesComplete"
	default:
		return fmt.Sprintf("TriggerPlacement(%d)", p)
	}
}

// WaitPlacement describes the owner-declared position of griddepcontrol.wait
// relative to consumer preamble and dependent body effects.
type WaitPlacement uint8

const (
	// WaitPlacementUnknown indicates that consumer wait placement has not been
	// declared or verified. Pairs with unknown placement are ineligible.
	WaitPlacementUnknown WaitPlacement = iota

	// WaitAfterIndependentPreamble declares that the consumer executes only
	// producer-independent BeforeWait preamble effects prior to
	// griddepcontrol.wait, and performs all producer-dependent accesses in
	// AfterWait. Requires HasIndependentWork=true and non-empty BeforeWait.
	WaitAfterIndependentPreamble

	// WaitAtEntryBeforeDependentBody declares that the consumer executes
	// griddepcontrol.wait at kernel entry with no BeforeWait preamble effects
	// (HasIndependentWork=false and empty BeforeWait) before accessing
	// producer-dependent resources in AfterWait.
	WaitAtEntryBeforeDependentBody

	// WaitAfterDependentAccess declares that the consumer accesses
	// producer-dependent resources prior to griddepcontrol.wait. Pairs with
	// this placement are ineligible.
	WaitAfterDependentAccess
)

func (p WaitPlacement) String() string {
	switch p {
	case WaitPlacementUnknown:
		return "WaitPlacementUnknown"
	case WaitAfterIndependentPreamble:
		return "WaitAfterIndependentPreamble"
	case WaitAtEntryBeforeDependentBody:
		return "WaitAtEntryBeforeDependentBody"
	case WaitAfterDependentAccess:
		return "WaitAfterDependentAccess"
	default:
		return fmt.Sprintf("WaitPlacement(%d)", p)
	}
}

// EffectKind classifies a declared execution effect within a kernel phase.
type EffectKind uint8

const (
	// EffectKindUnknown is an unset or unknown effect kind and is always refused.
	EffectKindUnknown EffectKind = iota

	// EffectMemory is a bounded device memory access to a named logical resource
	// and optional concrete byte range.
	EffectMemory

	// EffectPureCompute is register- or CTA-local computation with no global
	// memory or external side effects.
	EffectPureCompute

	// EffectVolatileMMIO is a volatile device or MMIO side effect and is refused
	// for PDL composition.
	EffectVolatileMMIO

	// EffectHostCallback is a host callback or host-synchronizing side effect
	// and is refused for PDL composition.
	EffectHostCallback

	// EffectSystemAtomic is a system-scope atomic side effect outside bounded
	// producer-consumer handoff memory and is refused for PDL composition.
	EffectSystemAtomic

	// EffectPeerMemory is a cross-device peer memory side effect and is refused
	// for PDL composition.
	EffectPeerMemory

	// EffectDynamicParallelism is a device-side child grid launch and is refused
	// for PDL composition.
	EffectDynamicParallelism

	// EffectOpaqueExternal is an unmodeled external side effect and is refused
	// for PDL composition.
	EffectOpaqueExternal
)

func (k EffectKind) String() string {
	switch k {
	case EffectKindUnknown:
		return "EffectKindUnknown"
	case EffectMemory:
		return "EffectMemory"
	case EffectPureCompute:
		return "EffectPureCompute"
	case EffectVolatileMMIO:
		return "EffectVolatileMMIO"
	case EffectHostCallback:
		return "EffectHostCallback"
	case EffectSystemAtomic:
		return "EffectSystemAtomic"
	case EffectPeerMemory:
		return "EffectPeerMemory"
	case EffectDynamicParallelism:
		return "EffectDynamicParallelism"
	case EffectOpaqueExternal:
		return "EffectOpaqueExternal"
	default:
		return fmt.Sprintf("EffectKind(%d)", k)
	}
}

// AccessMode describes how an EffectMemory phase effect accesses a resource.
type AccessMode uint8

const (
	// AccessNone indicates no memory access and is valid only with EffectPureCompute.
	AccessNone AccessMode = iota

	// AccessRead indicates read-only memory access.
	AccessRead

	// AccessWrite indicates write-only memory access.
	AccessWrite

	// AccessReadWrite indicates combined read and write memory access.
	AccessReadWrite

	// AccessAtomic indicates read-modify-write atomic memory access.
	AccessAtomic
)

func (m AccessMode) String() string {
	switch m {
	case AccessNone:
		return "AccessNone"
	case AccessRead:
		return "AccessRead"
	case AccessWrite:
		return "AccessWrite"
	case AccessReadWrite:
		return "AccessReadWrite"
	case AccessAtomic:
		return "AccessAtomic"
	default:
		return fmt.Sprintf("AccessMode(%d)", m)
	}
}

func (m AccessMode) reads() bool {
	return m == AccessRead || m == AccessReadWrite || m == AccessAtomic
}

func (m AccessMode) writes() bool {
	return m == AccessWrite || m == AccessReadWrite || m == AccessAtomic
}

func (m AccessMode) validMemoryAccess() bool {
	switch m {
	case AccessRead, AccessWrite, AccessReadWrite, AccessAtomic:
		return true
	default:
		return false
	}
}

// PhaseEffect describes one bounded effect within a producer or consumer phase.
type PhaseEffect struct {
	// Kind classifies the effect. Zero (EffectKindUnknown) is refused.
	Kind EffectKind

	// Resource names the logical buffer or resource for EffectMemory.
	// Must be empty for EffectPureCompute.
	Resource string

	// Access specifies the memory access direction for EffectMemory.
	// Must be AccessNone for EffectPureCompute.
	Access AccessMode

	// Base is an optional concrete device base address for range-aliasing checks.
	// When Base is non-zero on both effects being compared, [Base, Base+Bytes)
	// overlap is checked in addition to logical Resource name identity.
	Base uint64

	// Bytes is the positive byte extent of the memory access for EffectMemory.
	// Must be zero for EffectPureCompute.
	Bytes uint64
}

// KernelEffects describes the owner-supplied phase effects and barrier placement
// for a candidate producer or consumer kernel.
type KernelEffects struct {
	// TriggerPlacement declares where the producer emits
	// griddepcontrol.launch_dependents relative to shared handoff writes.
	TriggerPlacement TriggerPlacement

	// WaitPlacement declares where the consumer emits griddepcontrol.wait
	// relative to its preamble and dependent body.
	WaitPlacement WaitPlacement

	// BeforeLaunchDependents lists producer effects that complete and are
	// fenced before griddepcontrol.launch_dependents.
	BeforeLaunchDependents []PhaseEffect

	// AfterLaunchDependents lists optional producer trailing effects that
	// execute after griddepcontrol.launch_dependents.
	AfterLaunchDependents []PhaseEffect

	// BeforeWait lists consumer preamble effects that execute before
	// griddepcontrol.wait and may overlap with the producer.
	BeforeWait []PhaseEffect

	// AfterWait lists consumer effects that execute after griddepcontrol.wait.
	AfterWait []PhaseEffect

	// RequireSingleSMCoResidency requires that the combined producer and
	// consumer CTA resource footprint fit simultaneously on a single SM.
	// Even when satisfied, single-SM co-residency capability does not guarantee
	// runtime overlap or performance improvement.
	RequireSingleSMCoResidency bool
}

// ResourceSummary records the resolved per-CTA and combined producer/consumer
// hardware resource footprint evaluated against a probed target architecture.
type ResourceSummary struct {
	ProducerCTAWarps           uint32
	ConsumerCTAWarps           uint32
	CombinedCTAWarps           uint32
	ProducerSharedBytes        uint32
	ConsumerSharedBytes        uint32
	CombinedSharedBytes        uint32
	ProducerCTARegisters       uint32
	ConsumerCTARegisters       uint32
	CombinedCTARegisters       uint32
	ProducerTMEMBytes          uint32
	ConsumerTMEMBytes          uint32
	CombinedTMEMBytes          uint32
	SingleSMCoResidentPossible bool
}

type validatedMemoryEffect struct {
	role     string
	kernel   string
	phase    string
	resource string
	access   AccessMode
	base     uint64
	bytes    uint64
}

func kernelRoleName(name, fallback string) string {
	if s := strings.TrimSpace(name); s != "" {
		return s
	}
	return fallback
}

func validateEffectsAndResources(info cudakernel.TargetInfo, producer, consumer KernelDesc) (ResourceSummary, bool, string) {
	probedInfo, err := cudakernel.ProbedTargetSM(info.Arch)
	if err != nil || info.Estimated {
		return ResourceSummary{}, false, fmt.Sprintf("target architecture %s has unprobed estimated resource limits: %v", info.Arch, err)
	}

	pName := kernelRoleName(producer.Name, "producer")
	cName := kernelRoleName(consumer.Name, "consumer")

	switch producer.Effects.TriggerPlacement {
	case TriggerPlacementUnknown:
		return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q trigger placement is unknown or unverified", pName)
	case TriggerBeforeSharedWritesComplete:
		return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q triggers dependents before shared handoff writes complete", pName)
	case TriggerAfterSharedWrites:
	default:
		return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q has unsupported trigger placement %s", pName, producer.Effects.TriggerPlacement)
	}

	if len(producer.Effects.BeforeWait) > 0 || len(producer.Effects.AfterWait) > 0 || producer.Effects.WaitPlacement != WaitPlacementUnknown {
		return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q declares consumer wait phases on producer role", pName)
	}
	if len(consumer.Effects.BeforeLaunchDependents) > 0 || len(consumer.Effects.AfterLaunchDependents) > 0 || consumer.Effects.TriggerPlacement != TriggerPlacementUnknown {
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q declares producer launch_dependents phases on consumer role", cName)
	}

	if len(producer.Effects.BeforeLaunchDependents) == 0 {
		return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q declares no before_launch_dependents phase effects", pName)
	}

	switch consumer.Effects.WaitPlacement {
	case WaitPlacementUnknown:
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q wait placement is unknown or unverified", cName)
	case WaitAfterDependentAccess:
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q wait placement occurs after dependent resource access", cName)
	case WaitAfterIndependentPreamble, WaitAtEntryBeforeDependentBody:
	default:
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q has unsupported wait placement %s", cName, consumer.Effects.WaitPlacement)
	}

	if consumer.HasIndependentWork && len(consumer.Effects.BeforeWait) == 0 {
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q claims HasIndependentWork without declared before_wait preamble effects", cName)
	}
	if !consumer.HasIndependentWork && len(consumer.Effects.BeforeWait) > 0 {
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q declares before_wait preamble effects with HasIndependentWork=false", cName)
	}
	if consumer.Effects.WaitPlacement == WaitAfterIndependentPreamble && (!consumer.HasIndependentWork || len(consumer.Effects.BeforeWait) == 0) {
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q declares WaitAfterIndependentPreamble without independent before_wait preamble work", cName)
	}
	if consumer.Effects.WaitPlacement == WaitAtEntryBeforeDependentBody && (consumer.HasIndependentWork || len(consumer.Effects.BeforeWait) > 0) {
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q declares WaitAtEntryBeforeDependentBody with non-empty before_wait preamble", cName)
	}
	if len(consumer.Effects.AfterWait) == 0 {
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q declares no after_wait phase effects", cName)
	}

	prodBefore, reason := validatePhaseSlice("producer", pName, "before_launch_dependents", producer.Effects.BeforeLaunchDependents)
	if reason != "" {
		return ResourceSummary{}, false, reason
	}
	prodAfter, reason := validatePhaseSlice("producer", pName, "after_launch_dependents", producer.Effects.AfterLaunchDependents)
	if reason != "" {
		return ResourceSummary{}, false, reason
	}
	consBefore, reason := validatePhaseSlice("consumer", cName, "before_wait", consumer.Effects.BeforeWait)
	if reason != "" {
		return ResourceSummary{}, false, reason
	}
	consAfter, reason := validatePhaseSlice("consumer", cName, "after_wait", consumer.Effects.AfterWait)
	if reason != "" {
		return ResourceSummary{}, false, reason
	}

	allMem := make([]validatedMemoryEffect, 0, len(prodBefore)+len(prodAfter)+len(consBefore)+len(consAfter))
	allMem = append(allMem, prodBefore...)
	allMem = append(allMem, prodAfter...)
	allMem = append(allMem, consBefore...)
	allMem = append(allMem, consAfter...)
	if reason := checkAddressRangeConsistency(allMem); reason != "" {
		return ResourceSummary{}, false, reason
	}

	hasProducerWrite := false
	for _, eff := range prodBefore {
		if eff.access.writes() {
			hasProducerWrite = true
			break
		}
	}
	if !hasProducerWrite {
		return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q before_launch_dependents phase writes no handoff memory resource", pName)
	}

	// Hazard Rule 1: Consumer before_wait preamble must not conflict with any
	// producer phase (neither reading producer handoff outputs before wait nor
	// racing with producer before/after launch_dependents).
	for _, cEff := range consBefore {
		for _, pEff := range prodBefore {
			if memoryEffectsAlias(pEff, cEff) {
				if pEff.access.writes() {
					return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q before_wait preamble accesses producer handoff resource %q before %s", cName, cEff.resource, PTXWait)
				}
				if cEff.access.writes() {
					return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q before_wait preamble mutates shared producer resource %q before %s", cName, cEff.resource, PTXWait)
				}
			}
		}
		for _, pEff := range prodAfter {
			if memoryEffectsAlias(pEff, cEff) && (pEff.access.writes() || cEff.access.writes()) {
				return ResourceSummary{}, false, fmt.Sprintf("concurrent shared-resource hazard between producer kernel %q after_launch_dependents and consumer kernel %q before_wait on resource %q", pName, cName, cEff.resource)
			}
		}
	}

	// Consumer after_wait must read at least one handoff resource written by
	// producer before_launch_dependents.
	hasHandoffRead := false
	for _, cEff := range consAfter {
		if !cEff.access.reads() {
			continue
		}
		for _, pEff := range prodBefore {
			if pEff.access.writes() && memoryEffectsAlias(pEff, cEff) {
				hasHandoffRead = true
				break
			}
		}
		if hasHandoffRead {
			break
		}
	}
	if !hasHandoffRead {
		return ResourceSummary{}, false, fmt.Sprintf("consumer kernel %q after_wait phase does not read any handoff resource written by producer kernel %q before_launch_dependents", cName, pName)
	}

	// Hazard Rule 2: Producer after_launch_dependents trailing work must not
	// mutate handoff resources written in before_launch_dependents or conflict
	// with resources accessed by consumer after_wait.
	for _, pAfter := range prodAfter {
		for _, pBefore := range prodBefore {
			if memoryEffectsAlias(pAfter, pBefore) && pBefore.access.writes() && pAfter.access.writes() {
				return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q after_launch_dependents trailing phase mutates handoff resource %q after trigger", pName, pAfter.resource)
			}
		}
		for _, cAfter := range consAfter {
			if memoryEffectsAlias(pAfter, cAfter) && (pAfter.access.writes() || cAfter.access.writes()) {
				return ResourceSummary{}, false, fmt.Sprintf("producer kernel %q after_launch_dependents trailing phase accesses handoff/consumer resource %q after trigger", pName, pAfter.resource)
			}
		}
	}

	prodWarps, prodShared, prodRegs, prodTMEM, reason := validateKernelResources("producer", pName, producer, probedInfo)
	if reason != "" {
		return ResourceSummary{}, false, reason
	}
	consWarps, consShared, consRegs, consTMEM, reason := validateKernelResources("consumer", cName, consumer, probedInfo)
	if reason != "" {
		return ResourceSummary{}, false, reason
	}

	combinedTMEM64 := uint64(prodTMEM) + uint64(consTMEM)
	if combinedTMEM64 > maxTMEMBytesPerSM {
		return ResourceSummary{}, false, fmt.Sprintf("combined producer/consumer TMEM (%d B) exceeds %d B per-SM TMEM limit on %s", combinedTMEM64, maxTMEMBytesPerSM, probedInfo.Arch)
	}

	combinedWarps := prodWarps + consWarps
	combinedShared64 := uint64(prodShared) + uint64(consShared)
	var combinedShared uint32
	if combinedShared64 > math.MaxUint32 {
		combinedShared = math.MaxUint32
	} else {
		combinedShared = uint32(combinedShared64)
	}
	combinedRegs := prodRegs + consRegs

	coResidentPossible := probedInfo.MaxBlocksPerSM >= 2 &&
		combinedWarps <= probedInfo.WarpsPerSM &&
		combinedShared64 <= uint64(probedInfo.MaxSharedMemoryBytes) &&
		uint64(combinedRegs) <= maxRegistersPerSM &&
		combinedTMEM64 <= maxTMEMBytesPerSM

	summary := ResourceSummary{
		ProducerCTAWarps:           prodWarps,
		ConsumerCTAWarps:           consWarps,
		CombinedCTAWarps:           combinedWarps,
		ProducerSharedBytes:        prodShared,
		ConsumerSharedBytes:        consShared,
		CombinedSharedBytes:        combinedShared,
		ProducerCTARegisters:       prodRegs,
		ConsumerCTARegisters:       consRegs,
		CombinedCTARegisters:       combinedRegs,
		ProducerTMEMBytes:          prodTMEM,
		ConsumerTMEMBytes:          consTMEM,
		CombinedTMEMBytes:          uint32(combinedTMEM64),
		SingleSMCoResidentPossible: coResidentPossible,
	}

	if (producer.Effects.RequireSingleSMCoResidency || consumer.Effects.RequireSingleSMCoResidency) && !coResidentPossible {
		return summary, false, fmt.Sprintf(
			"combined producer/consumer CTA resources (warps=%d/%d, shared=%d/%d B, regs=%d/%d) exceed single-SM co-residency limits on %s",
			combinedWarps, probedInfo.WarpsPerSM,
			combinedShared64, probedInfo.MaxSharedMemoryBytes,
			combinedRegs, maxRegistersPerSM,
			probedInfo.Arch,
		)
	}

	return summary, len(consBefore) > 0, ""
}

func validatePhaseSlice(role, kName, phase string, effects []PhaseEffect) ([]validatedMemoryEffect, string) {
	if len(effects) > MaxPhaseEffects {
		return nil, fmt.Sprintf("%s kernel %q phase %s declares %d effects, exceeding maximum %d", role, kName, phase, len(effects), MaxPhaseEffects)
	}
	var memEffects []validatedMemoryEffect
	for _, eff := range effects {
		switch eff.Kind {
		case EffectKindUnknown:
			return nil, fmt.Sprintf("%s kernel %q phase %s has unknown or unspecified effect kind", role, kName, phase)
		case EffectVolatileMMIO, EffectHostCallback, EffectSystemAtomic, EffectPeerMemory, EffectDynamicParallelism, EffectOpaqueExternal:
			return nil, fmt.Sprintf("%s kernel %q phase %s uses unsafe or unsupported effect %s", role, kName, phase, eff.Kind)
		case EffectPureCompute:
			if strings.TrimSpace(eff.Resource) != "" || eff.Access != AccessNone || eff.Base != 0 || eff.Bytes != 0 {
				return nil, fmt.Sprintf("%s kernel %q phase %s EffectPureCompute must not specify a memory resource, access mode, or address range", role, kName, phase)
			}
		case EffectMemory:
			resName := strings.TrimSpace(eff.Resource)
			if resName == "" || strings.ContainsRune(resName, 0) {
				return nil, fmt.Sprintf("%s kernel %q phase %s EffectMemory requires a valid non-empty resource name", role, kName, phase)
			}
			if !eff.Access.validMemoryAccess() {
				return nil, fmt.Sprintf("%s kernel %q phase %s resource %q has invalid memory access mode %s", role, kName, phase, resName, eff.Access)
			}
			if eff.Bytes == 0 {
				return nil, fmt.Sprintf("%s kernel %q phase %s resource %q requires positive byte count", role, kName, phase, resName)
			}
			if eff.Base != 0 && eff.Bytes > math.MaxUint64-eff.Base {
				return nil, fmt.Sprintf("%s kernel %q phase %s resource %q address range [0x%x, +0x%x) overflows uint64", role, kName, phase, resName, eff.Base, eff.Bytes)
			}
			cand := validatedMemoryEffect{
				role:     role,
				kernel:   kName,
				phase:    phase,
				resource: resName,
				access:   eff.Access,
				base:     eff.Base,
				bytes:    eff.Bytes,
			}
			for _, prev := range memEffects {
				if memoryEffectsAlias(prev, cand) {
					return nil, fmt.Sprintf("%s kernel %q phase %s has duplicate or overlapping memory effects on resource %q and %q", role, kName, phase, prev.resource, cand.resource)
				}
			}
			memEffects = append(memEffects, cand)
		default:
			return nil, fmt.Sprintf("%s kernel %q phase %s has unrecognized effect kind %s", role, kName, phase, eff.Kind)
		}
	}
	return memEffects, ""
}

func checkAddressRangeConsistency(effects []validatedMemoryEffect) string {
	for i := 0; i < len(effects); i++ {
		a := effects[i]
		for j := i + 1; j < len(effects); j++ {
			b := effects[j]
			if a.resource != b.resource {
				continue
			}
			if (a.base != 0 || b.base != 0) && (a.base != b.base || a.bytes != b.bytes) {
				return fmt.Sprintf("inconsistent address range for resource %q: [0x%x, +0x%x) vs [0x%x, +0x%x)", a.resource, a.base, a.bytes, b.base, b.bytes)
			}
		}
	}
	return ""
}

func memoryEffectsAlias(a, b validatedMemoryEffect) bool {
	if a.resource == b.resource {
		return true
	}
	if a.base != 0 && b.base != 0 {
		endA := a.base + a.bytes
		endB := b.base + b.bytes
		return a.base < endB && b.base < endA
	}
	return false
}

func validateKernelResources(role, kName string, desc KernelDesc, info cudakernel.TargetInfo) (warps, sharedBytes, ctaRegs, tmemBytes uint32, reason string) {
	artRes, hasArtRes := artifactResources(desc.Artifact)
	if desc.Resources == nil && !hasArtRes {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q missing required resource requirements", role, kName)
	}
	if desc.Resources != nil && hasArtRes && *desc.Resources != artRes {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q resource requirements conflict with artifact resources", role, kName)
	}
	res := artRes
	if desc.Resources != nil {
		res = *desc.Resources
	}

	artTPB := artifactThreadsPerBlock(desc.Artifact)
	if desc.ThreadsPerBlock != 0 && artTPB != 0 && desc.ThreadsPerBlock != artTPB {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q ThreadsPerBlock (%d) conflicts with artifact JITOptions.ThreadsPerBlock (%d)", role, kName, desc.ThreadsPerBlock, artTPB)
	}
	tpb := desc.ThreadsPerBlock
	if tpb == 0 {
		tpb = artTPB
	}
	if tpb == 0 {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q missing required ThreadsPerBlock", role, kName)
	}
	if tpb > maxThreadsPerCTA {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q ThreadsPerBlock (%d) exceeds architectural limit %d", role, kName, tpb, maxThreadsPerCTA)
	}

	if res.AllocatedRegisters == 0 {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q missing AllocatedRegisters in resource requirements", role, kName)
	}
	if res.AllocatedRegisters > maxRegistersPerThread {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q AllocatedRegisters (%d) exceeds per-thread limit %d", role, kName, res.AllocatedRegisters, maxRegistersPerThread)
	}
	if res.PeakLiveGPR > res.AllocatedRegisters {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q PeakLiveGPR (%d) exceeds AllocatedRegisters (%d)", role, kName, res.PeakLiveGPR, res.AllocatedRegisters)
	}
	if res.RegisterQuantum != 0 && res.RegisterQuantum != info.RegisterQuantum {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q RegisterQuantum (%d) mismatches target %s quantum (%d)", role, kName, res.RegisterQuantum, info.Arch, info.RegisterQuantum)
	}

	warps = (tpb + 31) / 32
	if warps > info.WarpsPerSM {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q CTA warps (%d) exceed target %s WarpsPerSM (%d)", role, kName, warps, info.Arch, info.WarpsPerSM)
	}

	rawRegs := uint64(warps) * 32 * uint64(res.AllocatedRegisters)
	q := uint64(info.RegisterQuantum)
	if q == 0 {
		q = 256
	}
	roundedRegs := ((rawRegs + q - 1) / q) * q
	if roundedRegs > maxRegistersPerSM {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q CTA registers (%d) exceed per-SM register file limit %d on %s", role, kName, roundedRegs, maxRegistersPerSM, info.Arch)
	}
	ctaRegs = uint32(roundedRegs)

	totalShared64 := uint64(res.StaticSharedMemory) + uint64(res.DynamicSharedMemory)
	if totalShared64 > math.MaxUint32 {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q static (%d B) + dynamic (%d B) shared memory overflows uint32", role, kName, res.StaticSharedMemory, res.DynamicSharedMemory)
	}
	if res.DynamicSharedMemory > info.MaxOptInSharedBytes {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q dynamic shared memory (%d B) exceeds target %s opt-in limit (%d B)", role, kName, res.DynamicSharedMemory, info.Arch, info.MaxOptInSharedBytes)
	}
	if totalShared64 > uint64(info.MaxSharedMemoryBytes) {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q total shared memory (%d B) exceeds target %s SM limit (%d B)", role, kName, totalShared64, info.Arch, info.MaxSharedMemoryBytes)
	}
	sharedBytes = uint32(totalShared64)

	if res.TMEMBytes > 0 && !info.SupportsTMEM {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q requests %d B TMEM, but target %s does not support TMEM", role, kName, res.TMEMBytes, info.Arch)
	}
	if res.TMEMBytes > maxTMEMBytesPerSM {
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q TMEMBytes (%d B) exceeds %d B per-SM TMEM limit on %s", role, kName, res.TMEMBytes, maxTMEMBytesPerSM, info.Arch)
	}
	tmemBytes = res.TMEMBytes

	switch res.ExecutionGroup {
	case "", "thread", "warp", "cta", "cluster", "grid":
	case "wgmma":
		if !info.SupportsWGMMA {
			return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q ExecutionGroup %q is not supported on target %s", role, kName, res.ExecutionGroup, info.Arch)
		}
	case "tcgen05":
		if !info.SupportsTCGen05 {
			return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q ExecutionGroup %q is not supported on target %s", role, kName, res.ExecutionGroup, info.Arch)
		}
	default:
		return 0, 0, 0, 0, fmt.Sprintf("%s kernel %q has unrecognized ExecutionGroup %q", role, kName, res.ExecutionGroup)
	}

	return warps, sharedBytes, ctaRegs, tmemBytes, ""
}
