package cudapipeline

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// Supported GPU architecture identifiers.
const (
	ArchSM90   = "sm_90"
	ArchSM90a  = "sm_90a"
	ArchSM100  = "sm_100"
	ArchSM100a = "sm_100a"
	ArchSM121a = "sm_121a"
)

// Feature identifiers for PTX / GPU instructions and hardware capabilities.
const (
	FeatureTMA      = "tma"
	FeatureMBarrier = "mbarrier"
	FeatureMultimem = "multimem"
	FeatureWGMMA    = "wgmma"
	FeatureTCGen05  = "tcgen05"
	FeatureTMEM     = "tmem"
)

// Architectural limits and dimensions for Blackwell Tensor Memory (TMEM).
const (
	BlackwellTMEMCapacityBytes = 256 * 1024 // 256 KiB capacity per SM (512 columns * 512 bytes)
	BlackwellTMEMColumns       = 512        // Total addressable TMEM columns per SM
	BlackwellTMEMBytesPerCol   = 512        // 512 bytes per column (128 rows * 4 bytes)
	BlackwellTMEMAllocUnitCols = 32         // Column allocation granularity (32 columns = 16 KiB)
	BlackwellTMEMMinAlignBytes = 16         // 16-byte minimum alignment required for bulk async copy
)

// ArchProfile describes the architectural capabilities and physical constraints
// of a target NVIDIA GPU architecture.
//
// The Supports fields are derived at startup from the probed feature table in
// [cudadevicetable] rather than written here. Hand-written copies of these
// flags existed in two packages and disagreed: this one had sm_90 supporting
// wgmma, which ptxas rejects because wgmma requires the sm_90a target.
type ArchProfile struct {
	Name string

	// SupportsTMA and the fields below it report whether ptxas assembles the
	// construct for this target, not how it performs on silicon.
	SupportsTMA      bool
	SupportsMBarrier bool
	SupportsMultimem bool
	SupportsWGMMA    bool
	SupportsTCGen05  bool
	SupportsTMEM     bool

	// FeatureSource names the assembler the Supports fields were read with.
	// Empty means this target is not in the probed table and every Supports
	// field is an unmeasured false.
	FeatureSource string

	// Estimated reports that this profile was synthesized for a target the
	// tables do not describe. Its alignment fields are the architectural
	// constants every target here shares; its Supports fields are unmeasured
	// and its TMEM dimensions are zero because nothing said otherwise, not
	// because the hardware lacks them.
	Estimated bool

	MinBufferAlignBytes  int
	MinBarrierAlignBytes int
	MaxBoxWidthBytes     int
	MaxTMEMCapacityBytes int
	MinTMEMAlignBytes    int
	TMEMBytesPerColumn   int
	TMEMColumnsPerSM     int
	TMEMAllocColUnit     int
}

// knownProfiles maps canonical architecture names to their hardware capabilities.
var knownProfiles = map[string]ArchProfile{
	ArchSM90: {
		Name:                 ArchSM90,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
		MaxTMEMCapacityBytes: 0,
		MinTMEMAlignBytes:    0,
		TMEMBytesPerColumn:   0,
		TMEMColumnsPerSM:     0,
		TMEMAllocColUnit:     0,
	},
	// sm_90a is a separate profile because the suffix selects an instruction
	// set: wgmma is accepted at sm_90a and rejected at sm_90, on one piece of
	// silicon. Without this entry a caller that wants wgmma has no target to
	// ask for, which is why the sm_90 row used to claim it.
	ArchSM90a: {
		Name:                 ArchSM90a,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
	},
	ArchSM100: {
		Name:                 ArchSM100,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
		MaxTMEMCapacityBytes: BlackwellTMEMCapacityBytes,
		MinTMEMAlignBytes:    BlackwellTMEMMinAlignBytes,
		TMEMBytesPerColumn:   BlackwellTMEMBytesPerCol,
		TMEMColumnsPerSM:     BlackwellTMEMColumns,
		TMEMAllocColUnit:     BlackwellTMEMAllocUnitCols,
	},
	ArchSM100a: {
		Name:                 ArchSM100a,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
		MaxTMEMCapacityBytes: BlackwellTMEMCapacityBytes,
		MinTMEMAlignBytes:    BlackwellTMEMMinAlignBytes,
		TMEMBytesPerColumn:   BlackwellTMEMBytesPerCol,
		TMEMColumnsPerSM:     BlackwellTMEMColumns,
		TMEMAllocColUnit:     BlackwellTMEMAllocUnitCols,
	},
	ArchSM121a: {
		Name:                 ArchSM121a,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
		MaxTMEMCapacityBytes: 0,
		MinTMEMAlignBytes:    0,
		TMEMBytesPerColumn:   0,
		TMEMColumnsPerSM:     0,
		TMEMAllocColUnit:     0,
	},
}

// featureSetters maps a probed feature name to the profile field it fills.
// TMEM follows tcgen05: Tensor Memory is reachable only through those
// instructions, so one probe answers both.
var featureSetters = map[string]func(*ArchProfile, bool){
	"tma":      func(p *ArchProfile, v bool) { p.SupportsTMA = v },
	"mbarrier": func(p *ArchProfile, v bool) { p.SupportsMBarrier = v },
	"multimem": func(p *ArchProfile, v bool) { p.SupportsMultimem = v },
	"wgmma":    func(p *ArchProfile, v bool) { p.SupportsWGMMA = v },
	"tcgen05":  func(p *ArchProfile, v bool) { p.SupportsTCGen05, p.SupportsTMEM = v, v },
}

// init fills the capability fields from the probed table, and adds a profile
// for every probed target this map does not list.
//
// The second half is what keeps the hand-written map from shadowing the
// evidence. It listed five targets while the table records eleven, so sm_103a
// and sm_110a were probed, recorded, and then unreachable through this
// package -- the measurement existed and the lookup said "unknown
// architecture". Seeding from the table means probing a new target publishes
// it here with one regeneration and no edit.
func init() {
	for _, name := range cudadevicetable.Targets() {
		if _, ok := knownProfiles[name]; !ok {
			knownProfiles[name] = baseProfile(name)
		}
	}
	for arch, profile := range knownProfiles {
		tgt, ok := cudadevicetable.LookupTarget(arch)
		if !ok {
			continue
		}
		measured := false
		for feature, set := range featureSetters {
			v, err := tgt.Supports(feature)
			if err != nil {
				continue
			}
			set(&profile, v)
			measured = true
		}
		if measured && tgt.Probe != nil {
			profile.FeatureSource = fmt.Sprintf("%s %s (%s, %s)", tgt.Probe.Tool, tgt.Probe.ToolVersion, tgt.Probe.Host, tgt.Probe.ProbedAt)
		}
		// TMEM dimensions describe hardware, not instruction acceptance. A
		// target whose tcgen05 support was measured gets the Blackwell
		// dimensions; one measured without it gets zeros. The probe settles
		// reachability, not size.
		if profile.SupportsTMEM && profile.MaxTMEMCapacityBytes == 0 {
			profile.MaxTMEMCapacityBytes = BlackwellTMEMCapacityBytes
			profile.MinTMEMAlignBytes = BlackwellTMEMMinAlignBytes
			profile.TMEMBytesPerColumn = BlackwellTMEMBytesPerCol
			profile.TMEMColumnsPerSM = BlackwellTMEMColumns
			profile.TMEMAllocColUnit = BlackwellTMEMAllocUnitCols
		}
		if !profile.SupportsTMEM {
			profile.MaxTMEMCapacityBytes, profile.MinTMEMAlignBytes = 0, 0
			profile.TMEMBytesPerColumn, profile.TMEMColumnsPerSM, profile.TMEMAllocColUnit = 0, 0, 0
		}
		knownProfiles[arch] = profile
	}
}

// CanonicalArch normalizes an architecture string like "sm90", "SM_121A", or "sm_121a"
// into canonical form.
func CanonicalArch(arch string) string {
	s := strings.ToLower(strings.TrimSpace(arch))
	if strings.HasPrefix(s, "sm") && len(s) > 2 && s[2] != '_' {
		s = "sm_" + s[2:]
	}
	return s
}

// baseProfile is the profile for a target with no recorded geometry: the TMA
// alignment rules, which are the same for every target this package has ever
// described, and nothing else. Its Supports fields are filled by init when the
// target turns out to be in the feature table.
func baseProfile(name string) ArchProfile {
	return ArchProfile{
		Name:                 name,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
	}
}

// ArchProfileFor returns the architecture profile for the specified
// architecture name.
//
// A target neither this package nor the feature table describes gets a profile
// marked Estimated rather than an error. Refusing would make a GPU released
// after this code unusable for the work that does not depend on the missing
// facts, which is most of it; the marker and [ProbedArchProfileFor] are for
// the work that does. The argument must still look like a target: "sm_130a"
// is a plausible future part, "hopper" is a typo, and inventing a profile for
// a typo helps nobody.
func ArchProfileFor(arch string) (ArchProfile, error) {
	norm := CanonicalArch(arch)
	if p, ok := knownProfiles[norm]; ok {
		return p, nil
	}
	if !archPattern.MatchString(norm) {
		return ArchProfile{}, fmt.Errorf("unknown or unsupported architecture: %q", arch)
	}
	p := baseProfile(norm)
	p.Estimated = true
	return p, nil
}

// ProbedArchProfileFor is [ArchProfileFor] restricted to targets something has
// measured. Use it where an estimate would be reported as a fact.
func ProbedArchProfileFor(arch string) (ArchProfile, error) {
	p, err := ArchProfileFor(arch)
	if err != nil {
		return ArchProfile{}, err
	}
	if p.Estimated || p.FeatureSource == "" {
		return ArchProfile{}, fmt.Errorf("cudapipeline: %s: no probed feature record; run cmd/cudadevicetable -ptxas to add one", p.Name)
	}
	return p, nil
}

// archPattern matches an exact PTX target such as "sm_90a".
var archPattern = regexp.MustCompile(`^sm_[0-9]+[a-z]*$`)

// SupportsFeature reports whether the architecture supports the given feature.
func (p ArchProfile) SupportsFeature(feature string) bool {
	switch strings.ToLower(feature) {
	case FeatureTMA:
		return p.SupportsTMA
	case FeatureMBarrier:
		return p.SupportsMBarrier
	case FeatureMultimem:
		return p.SupportsMultimem
	case FeatureWGMMA:
		return p.SupportsWGMMA
	case FeatureTCGen05:
		return p.SupportsTCGen05
	case FeatureTMEM:
		return p.SupportsTMEM
	default:
		return false
	}
}

// ValidateFeature returns an error if the specified feature is not supported
// on this architecture.
//
// The explanation is assembled from the probed table rather than written here.
// The messages this replaced named the targets a feature requires -- "tcgen05
// requires sm_100 or sm_100a", "wgmma is not supported on Blackwell sm_121a"
// -- and ptxas disagrees with the first of those: tcgen05 needs the sm_100a
// suffix and is rejected at sm_100. A sentence that restates a capability
// table drifts from it silently, because nothing reads the sentence.
func (p ArchProfile) ValidateFeature(feature string) error {
	feat := strings.ToLower(feature)
	if p.SupportsFeature(feat) {
		return nil
	}
	// TMEM has no fixture of its own: it is reachable only through tcgen05,
	// so that is the rejection to report.
	probed := feat
	if probed == FeatureTMEM {
		probed = FeatureTCGen05
	}
	// An unmeasured target is a different answer from a rejected one, and
	// saying "does not support" for both is how a table's silence becomes a
	// hardware claim. Report the silence as silence and name the way out.
	if a := cudadevicetable.Ask(p.Name, probed); !a.Measured {
		return fmt.Errorf("target %s: cannot say whether %s is supported: %s; probe it with cmd/cudadevicetable -ptxas -targets=%s", p.Name, feat, a.Why(), p.Name)
	}
	msg := fmt.Sprintf("target %s does not support %s", p.Name, feat)
	if feat == FeatureTMEM {
		msg += ": Tensor Memory is addressable only via tcgen05"
	}
	if tgt, ok := cudadevicetable.LookupTarget(p.Name); ok {
		if d := tgt.Diagnostic(probed); d != "" {
			msg += ": " + d
		}
	}
	if where := cudadevicetable.TargetsSupporting(probed); len(where) > 0 {
		msg += fmt.Sprintf(" (accepted at %s)", strings.Join(where, ", "))
	}
	return errors.New(msg)
}
