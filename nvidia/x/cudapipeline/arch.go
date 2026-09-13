package cudapipeline

import (
	"fmt"
	"strings"
)

// Supported GPU architecture identifiers.
const (
	ArchSM90   = "sm_90"
	ArchSM100  = "sm_100"
	ArchSM121a = "sm_121a"
)

// Feature identifiers for PTX / GPU instructions and hardware capabilities.
const (
	FeatureTMA      = "tma"
	FeatureMBarrier = "mbarrier"
	FeatureMultimem = "multimem"
	FeatureWGMMA    = "wgmma"
	FeatureTCGen05  = "tcgen05"
)

// ArchProfile describes the architectural capabilities and physical constraints
// of a target NVIDIA GPU architecture.
type ArchProfile struct {
	Name                 string
	SupportsTMA          bool
	SupportsMBarrier     bool
	SupportsMultimem     bool
	SupportsWGMMA        bool
	SupportsTCGen05      bool
	MinBufferAlignBytes  int
	MinBarrierAlignBytes int
	MaxBoxWidthBytes     int
}

// knownProfiles maps canonical architecture names to their hardware capabilities.
var knownProfiles = map[string]ArchProfile{
	ArchSM90: {
		Name:                 ArchSM90,
		SupportsTMA:          true,
		SupportsMBarrier:     true,
		SupportsMultimem:     false,
		SupportsWGMMA:        true,
		SupportsTCGen05:      false,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
	},
	ArchSM100: {
		Name:                 ArchSM100,
		SupportsTMA:          true,
		SupportsMBarrier:     true,
		SupportsMultimem:     true,
		SupportsWGMMA:        false,
		SupportsTCGen05:      true,
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
	},
	ArchSM121a: {
		Name:                 ArchSM121a,
		SupportsTMA:          true,
		SupportsMBarrier:     true,
		SupportsMultimem:     true,
		SupportsWGMMA:        false, // Blackwell sm_121a rejects wgmma
		SupportsTCGen05:      false, // Blackwell sm_121a rejects tcgen05
		MinBufferAlignBytes:  16,
		MinBarrierAlignBytes: 8,
		MaxBoxWidthBytes:     128,
	},
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

// ArchProfileFor returns the architecture profile for the specified architecture name.
// If the architecture is unknown, an error is returned.
func ArchProfileFor(arch string) (ArchProfile, error) {
	norm := CanonicalArch(arch)
	if p, ok := knownProfiles[norm]; ok {
		return p, nil
	}
	return ArchProfile{}, fmt.Errorf("unknown or unsupported architecture: %q", arch)
}

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
	default:
		return false
	}
}

// ValidateFeature returns an error if the specified feature is not supported
// on this architecture, including architecture-specific explanations.
func (p ArchProfile) ValidateFeature(feature string) error {
	feat := strings.ToLower(feature)
	if p.SupportsFeature(feat) {
		return nil
	}
	switch feat {
	case FeatureWGMMA:
		if p.Name == ArchSM121a {
			return fmt.Errorf("target %s rejects wgmma: Hopper warp-group MMA instructions are not supported on Blackwell sm_121a", p.Name)
		}
		return fmt.Errorf("target %s does not support wgmma", p.Name)
	case FeatureTCGen05:
		if p.Name == ArchSM121a {
			return fmt.Errorf("target %s rejects tcgen05: Blackwell tensor-core generation 5 instructions are rejected on sm_121a", p.Name)
		}
		return fmt.Errorf("target %s does not support tcgen05", p.Name)
	case FeatureMultimem:
		return fmt.Errorf("target %s does not support multimem: multimem requires sm_100 or sm_121a", p.Name)
	default:
		return fmt.Errorf("target %s does not support feature %q", p.Name, feature)
	}
}
