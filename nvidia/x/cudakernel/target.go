package cudakernel

import (
	"fmt"
	"strconv"
	"strings"
)

// TargetInfo defines hardware architecture limits, features, and capabilities.
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
	SupportsTCGen05      bool
	SupportsWGMMA        bool
	SupportsTMEM         bool
	SupportsMulticast    bool
}

// Known target specifications.
var knownTargets = map[string]TargetInfo{
	// Hopper SM90
	"sm_90": {
		Arch:                 "sm_90",
		Major:                9,
		Minor:                0,
		MaxSharedMemoryBytes: 233472,
		MaxOptInSharedBytes:  232448,
		WarpsPerSM:           64,
		MaxBlocksPerSM:       32,
		RegisterQuantum:      256,
		SupportsTCGen05:      false,
		SupportsWGMMA:        true,
		SupportsTMEM:         false,
		SupportsMulticast:    true,
	},
	"sm_90a": {
		Arch:                 "sm_90a",
		Major:                9,
		Minor:                0,
		Suffix:               "a",
		MaxSharedMemoryBytes: 233472,
		MaxOptInSharedBytes:  232448,
		WarpsPerSM:           64,
		MaxBlocksPerSM:       32,
		RegisterQuantum:      256,
		SupportsTCGen05:      false,
		SupportsWGMMA:        true,
		SupportsTMEM:         false,
		SupportsMulticast:    true,
	},

	// Blackwell SM100 (DGX GB200 / B100)
	"sm_100": {
		Arch:                 "sm_100",
		Major:                10,
		Minor:                0,
		MaxSharedMemoryBytes: 233472,
		MaxOptInSharedBytes:  232448,
		WarpsPerSM:           64,
		MaxBlocksPerSM:       32,
		RegisterQuantum:      256,
		SupportsTCGen05:      true,
		SupportsWGMMA:        true,
		SupportsTMEM:         true,
		SupportsMulticast:    true,
	},
	"sm_100a": {
		Arch:                 "sm_100a",
		Major:                10,
		Minor:                0,
		Suffix:               "a",
		MaxSharedMemoryBytes: 233472,
		MaxOptInSharedBytes:  232448,
		WarpsPerSM:           64,
		MaxBlocksPerSM:       32,
		RegisterQuantum:      256,
		SupportsTCGen05:      true,
		SupportsWGMMA:        true,
		SupportsTMEM:         true,
		SupportsMulticast:    true,
	},

	// Blackwell SM121 (DGX GB10)
	"sm_121": {
		Arch:                 "sm_121",
		Major:                12,
		Minor:                1,
		MaxSharedMemoryBytes: 102400,
		MaxOptInSharedBytes:  101376,
		WarpsPerSM:           48,
		MaxBlocksPerSM:       24,
		RegisterQuantum:      256,
		SupportsTCGen05:      false, // Strictly lacks 5th-gen Tensor Core tcgen05
		SupportsWGMMA:        false, // Strictly lacks wgmma
		SupportsTMEM:         false, // Strictly lacks TMEM
		SupportsMulticast:    false,
	},
	"sm_121a": {
		Arch:                 "sm_121a",
		Major:                12,
		Minor:                1,
		Suffix:               "a",
		MaxSharedMemoryBytes: 102400,
		MaxOptInSharedBytes:  101376,
		WarpsPerSM:           48,
		MaxBlocksPerSM:       24,
		RegisterQuantum:      256,
		SupportsTCGen05:      false, // Strictly lacks 5th-gen Tensor Core tcgen05
		SupportsWGMMA:        false, // Strictly lacks wgmma
		SupportsTMEM:         false, // Strictly lacks TMEM
		SupportsMulticast:    false,
	},
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

	// Fallback to parsed numbers with default values.
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
	}, nil
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
