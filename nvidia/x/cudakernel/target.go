package cudakernel

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// TargetInfo defines hardware architecture limits, features, and capabilities.
//
// The Supports fields are not written here. They are derived at startup from
// the probed feature table in [cudadevicetable], which records what ptxas
// accepts for each exact target. They were hand-written once, and the two
// copies of them in this module drifted: this table claimed sm_100a supports
// wgmma and sm_121a does not support multicast, and ptxas contradicts both.
// Deriving them means a disagreement of that kind cannot be written down.
type TargetInfo struct {
	Arch                 string
	Major                int
	Minor                int
	Suffix               string
	MaxSharedMemoryBytes uint32
	MaxOptInSharedBytes  uint32
	WarpsPerSM           uint32
	MaxBlocksPerSM       uint32
	RegisterQuantum      uint32

	// SupportsTCGen05 and the fields below it report whether ptxas assembles
	// the construct for this target. They say nothing about how fast it runs,
	// and nothing about whether a device here has executed it.
	SupportsTCGen05   bool
	SupportsWGMMA     bool
	SupportsTMEM      bool
	SupportsMulticast bool

	// FeatureSource names where the Supports fields came from, so a caller
	// can tell a probed answer from a target nobody has probed. It is empty
	// only in the latter case, where every Supports field is false because
	// nothing was measured -- not because the answer is no.
	FeatureSource string

	// Estimated reports that the geometry fields above -- shared memory,
	// warps, blocks, register quantum -- are defaults for an architecture
	// this table does not describe, not facts about it.
	//
	// The defaults exist because refusing outright would make a new GPU
	// unusable for everything, including the work that does not depend on
	// geometry. They are dangerous for exactly the work that does: an
	// occupancy or roofline figure computed from an invented 48 KB is wrong
	// in the way that reads as right. Callers that consume the geometry must
	// check this field, or use [ProbedTargetSM], which refuses instead.
	Estimated bool
}

// knownTargets holds the targets whose geometry is recorded, keyed by
// target name. It is filled at startup from [cudadevicetable].
var knownTargets = map[string]TargetInfo{}

// recordedTargets lists the targets knownTargets describes.
var recordedTargets = []string{"sm_90", "sm_90a", "sm_100", "sm_100a", "sm_121", "sm_121a"}

// profileTarget returns the geometry of arch from the cudadevicetable
// profile of its architecture. A feature suffix selects an instruction set,
// not different hardware, so sm_90 and sm_90a share sm_90a's profile.
func profileTarget(arch string) (TargetInfo, error) {
	major, minor, suffix, err := ParseArch(arch)
	if err != nil {
		return TargetInfo{}, err
	}
	p, err := cudadevicetable.LookupProfile(fmt.Sprintf("sm_%d%da", major, minor))
	if err != nil {
		return TargetInfo{}, err
	}
	return TargetInfo{
		Arch:                 arch,
		Major:                major,
		Minor:                minor,
		Suffix:               suffix,
		MaxSharedMemoryBytes: uint32(p.SharedPerSM.Value),
		MaxOptInSharedBytes:  uint32(p.SharedPerBlockOptin.Value),
		WarpsPerSM:           uint32(p.ThreadsPerSM.Value / 32),
		MaxBlocksPerSM:       uint32(p.BlocksPerSM.Value),
		RegisterQuantum:      256,
	}, nil
}

// featureNames maps a TargetInfo capability to the probed feature that answers
// it. tmem has no entry: Tensor Memory is addressable only through tcgen05, so
// probing a separate fixture for it would measure the same instruction twice.
var featureNames = map[string]func(*TargetInfo, bool){
	"tcgen05":      func(t *TargetInfo, v bool) { t.SupportsTCGen05, t.SupportsTMEM = v, v },
	"wgmma":        func(t *TargetInfo, v bool) { t.SupportsWGMMA = v },
	"tmamulticast": func(t *TargetInfo, v bool) { t.SupportsMulticast = v },
}

// init fills knownTargets from the device table: geometry from the
// profiles, capability fields from the probed feature table. A target the
// feature table does not cover keeps zero capability fields and an empty
// FeatureSource, which is the honest record of "not measured" rather than
// a row of confident falses.
func init() {
	for _, arch := range recordedTargets {
		info, err := profileTarget(arch)
		if err != nil {
			panic("cudakernel: " + err.Error())
		}
		knownTargets[arch] = info
	}
	for arch, info := range knownTargets {
		tgt, ok := cudadevicetable.LookupTarget(arch)
		if !ok {
			continue
		}
		measured := false
		for feature, set := range featureNames {
			v, err := tgt.Supports(feature)
			if err != nil {
				continue
			}
			set(&info, v)
			measured = true
		}
		if measured && tgt.Probe != nil {
			info.FeatureSource = fmt.Sprintf("%s %s (%s, %s)", tgt.Probe.Tool, tgt.Probe.ToolVersion, tgt.Probe.Host, tgt.Probe.ProbedAt)
		}
		knownTargets[arch] = info
	}
}

// ParseArch parses an architecture string (e.g. "sm_90", "sm_121a", "compute_100")
// into its major, minor, and optional suffix components.
func ParseArch(arch string) (major int, minor int, suffix string, err error) {
	s := strings.TrimPrefix(arch, "sm_")
	s = strings.TrimPrefix(s, "compute_")
	if s == arch {
		return 0, 0, "", fmt.Errorf("invalid arch format: %q (expected sm_XX or compute_XX)", arch)
	}

	// Suffix is any non-digit tail, e.g. "a", "f".
	var numStr string
	for i, r := range s {
		if r < '0' || r > '9' {
			numStr = s[:i]
			suffix = s[i:]
			break
		}
	}
	if numStr == "" {
		numStr = s
	}

	n, err := strconv.Atoi(numStr)
	if err != nil {
		return 0, 0, "", fmt.Errorf("invalid numeric arch in %q: %w", arch, err)
	}

	if n >= 100 {
		major = n / 10
		minor = n % 10
	} else if n >= 10 {
		major = n / 10
		minor = n % 10
	} else {
		return 0, 0, "", fmt.Errorf("unrecognized architecture number: %d", n)
	}
	return major, minor, suffix, nil
}

// ValidateTargetArch validates that target is a recognized architecture string.
func ValidateTargetArch(target string) error {
	_, _, _, err := ParseArch(target)
	return err
}

// TargetSM retrieves the architectural specification and limits for the given target.
func TargetSM(target string) (TargetInfo, error) {
	canonical := strings.ToLower(target)
	if strings.HasPrefix(canonical, "compute_") {
		canonical = "sm_" + strings.TrimPrefix(canonical, "compute_")
	}
	if info, ok := knownTargets[canonical]; ok {
		return info, nil
	}

	// A target this table does not describe. The architecture number parses,
	// so the identity fields are real; every geometry field below is a
	// default standing in for a measurement nobody took, and Estimated says
	// so. Returning these unmarked was this package's own version of the
	// hallucination the feature table was built to remove: a GPU released
	// after this code silently acquired 48 KB of shared memory and 16 blocks
	// per SM, and nothing downstream could tell.
	major, minor, suffix, err := ParseArch(target)
	if err != nil {
		return TargetInfo{}, err
	}
	return TargetInfo{
		Arch:                 canonical,
		Major:                major,
		Minor:                minor,
		Suffix:               suffix,
		MaxSharedMemoryBytes: 49152,
		MaxOptInSharedBytes:  49152,
		WarpsPerSM:           32,
		MaxBlocksPerSM:       16,
		RegisterQuantum:      128,
		Estimated:            true,
	}, nil
}

// ProbedTargetSM is [TargetSM] without the estimated fallback: it returns an
// error for a target whose geometry this table does not record.
//
// Use it wherever the geometry feeds a number a reader will believe --
// occupancy, register budgeting, a roofline placement. The rule is the one
// [cudadevicetable.AssumeAbsent] states: guess where the assembler will catch
// the mistake, refuse where it will not.
func ProbedTargetSM(target string) (TargetInfo, error) {
	info, err := TargetSM(target)
	if err != nil {
		return TargetInfo{}, err
	}
	if info.Estimated {
		return TargetInfo{}, fmt.Errorf("cudakernel: %s: no recorded geometry; %s", info.Arch, estimatedFrom(info))
	}
	return info, nil
}

// estimatedFrom names what a caller can do about a missing record.
func estimatedFrom(info TargetInfo) string {
	return fmt.Sprintf("probe a device with cmd/cudadevicetable and add sm_%d%d to the table", info.Major, info.Minor)
}

// ValidateTargetCompatibility checks whether an artifact compiled for artifactTarget
// can legally execute on a device targeting deviceArch.
func ValidateTargetCompatibility(artifactTarget, deviceArch string) error {
	artInfo, err := TargetSM(artifactTarget)
	if err != nil {
		return fmt.Errorf("artifact target %q: %w", artifactTarget, err)
	}
	devInfo, err := TargetSM(deviceArch)
	if err != nil {
		return fmt.Errorf("device arch %q: %w", deviceArch, err)
	}

	// Check hardware feature fencing:
	// SM100 requires tcgen05, wgmma, and TMEM which SM121 (GB10) lacks.
	if artInfo.SupportsTCGen05 && !devInfo.SupportsTCGen05 {
		return fmt.Errorf("cudakernel: target feature mismatch: artifact compiled for %s requires tcgen05 which %s strictly lacks",
			artifactTarget, deviceArch)
	}
	if artInfo.SupportsTMEM && !devInfo.SupportsTMEM {
		return fmt.Errorf("cudakernel: target feature mismatch: artifact compiled for %s requires TMEM which %s strictly lacks",
			artifactTarget, deviceArch)
	}
	if artInfo.SupportsWGMMA && !devInfo.SupportsWGMMA {
		return fmt.Errorf("cudakernel: target feature mismatch: artifact compiled for %s requires wgmma which %s strictly lacks",
			artifactTarget, deviceArch)
	}

	// Check major compute capability:
	// A binary compiled for a higher major compute capability cannot run on lower major hardware.
	if artInfo.Major > devInfo.Major {
		return fmt.Errorf("cudakernel: target compute capability mismatch: artifact requires major %d, device has %d",
			artInfo.Major, devInfo.Major)
	}

	return nil
}
