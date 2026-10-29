package cudaoccupancy

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// DeviceProperties defines hardware execution limits and allocation granularities
// for a GPU Streaming Multiprocessor (SM).
type DeviceProperties struct {
	Architecture            string // Architecture code (e.g. "sm_90", "sm_121")
	MaxThreadsPerBlock      int    // Maximum threads per thread block (CTA)
	WarpSize                int    // Warp size in threads (always 32)
	MaxWarpsPerSM           int    // Maximum concurrent warps per SM
	MaxBlocksPerSM          int    // Maximum concurrent thread blocks per SM
	MaxRegistersPerSM       int    // Total 32-bit registers available per SM
	MaxSharedMemoryPerSM    int    // Maximum shared memory per SM in bytes
	MaxSharedMemoryPerBlock int    // Default maximum shared memory per thread block in bytes
	MaxSharedMemoryOptin    int    // Opt-in maximum shared memory per block in bytes
	RegAllocationUnit       int    // Register allocation granularity in registers per warp
	SmemAllocationUnit      int    // Shared memory allocation granularity in bytes

	// TMEMColumnsPerSM is the Tensor Memory an SM holds, in 32-bit columns
	// of 128 lanes: 512 where tcgen05 exists, 0 elsewhere. A CTA allocates
	// a power of two from 32 to 512 columns with tcgen05.alloc, so TMEM
	// caps residency at TMEMColumnsPerSM / columns blocks per SM however
	// few registers and how little shared memory the kernel uses.
	TMEMColumnsPerSM int

	// MaxClusterSize is the largest thread block cluster a kernel may
	// launch with when it opts in to non-portable sizes; 0 where clusters
	// do not exist. The portable maximum is 8 everywhere clusters exist.
	MaxClusterSize int

	// Source names the device these values were read from, or is empty when
	// they were hand-entered from a datasheet and no device has confirmed
	// them. Every field above renders identically either way, which is the
	// problem: an occupancy figure computed from an unconfirmed capacity is
	// wrong in the way that reads as right. [LookupProbedDevice] refuses the
	// unconfirmed rows for callers that cannot afford one.
	//
	// It is filled at startup from cudadevicetable, so it cannot be set by
	// editing this file.
	Source string
}

// Predefined device specifications for modern NVIDIA GPU architectures.
//
// Every architecture but sm_70 takes its per-SM limits -- warps, blocks,
// shared memory, Tensor Memory and the cluster limit -- from the
// [cudadevicetable] profile of its "a" target, which holds a probed value
// where a device was probed and a cited datasheet value elsewhere. What is
// written here is what that table does not hold: registers and allocation
// granularities.
var (
	// Volta V100 (SM 7.0). cudadevicetable has no sm_70 profile.
	SM70 = DeviceProperties{
		Architecture:            "sm_70",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           64,
		MaxBlocksPerSM:          32,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    98304,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    98304,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
	}

	SM75  = profileLimits("sm_75", "sm_75")    // Turing T4
	SM80  = profileLimits("sm_80", "sm_80")    // Ampere A100
	SM86  = profileLimits("sm_86", "sm_86")    // Ampere A10, RTX 30xx
	SM87  = profileLimits("sm_87", "sm_87")    // Jetson Orin
	SM89  = profileLimits("sm_89", "sm_89")    // Ada L40S, L4, RTX 40xx
	SM90  = profileLimits("sm_90", "sm_90a")   // Hopper H100, H200
	SM100 = profileLimits("sm_100", "sm_100a") // Blackwell B200, GB200
	SM103 = profileLimits("sm_103", "sm_103a") // Blackwell Ultra B300, GB300
	SM110 = profileLimits("sm_110", "sm_110a") // Jetson Thor
	SM120 = profileLimits("sm_120", "sm_120a") // Blackwell RTX PRO 6000, RTX 5090
	SM121 = profileLimits("sm_121", "sm_121a") // Blackwell GB10
)

// profileLimits returns the properties of arch with its per-SM limits read
// from the cudadevicetable profile of target. Profiles of one target share
// those limits, so the first listed product serves.
func profileLimits(arch, target string) DeviceProperties {
	p, err := cudadevicetable.LookupProfile(target)
	if err != nil {
		panic("cudaoccupancy: " + err.Error())
	}
	const warpSize, tmemColumnBytes = 32, 128 * 4
	return DeviceProperties{
		Architecture:            arch,
		MaxThreadsPerBlock:      1024,
		WarpSize:                warpSize,
		MaxWarpsPerSM:           int(p.ThreadsPerSM.Value) / warpSize,
		MaxBlocksPerSM:          int(p.BlocksPerSM.Value),
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    int(p.SharedPerSM.Value),
		MaxSharedMemoryPerBlock: int(p.SharedPerBlock.Value),
		MaxSharedMemoryOptin:    int(p.SharedPerBlockOptin.Value),
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
		TMEMColumnsPerSM:        int(p.TMEMPerSM.Value) / tmemColumnBytes,
		MaxClusterSize:          int(p.ClusterSizeOptin.Value),
	}
}

// knownArchitectures maps an architecture string to its properties. A key may
// only name the architecture its value describes: "sm_90a" reaches SM90
// because a feature suffix does not change SM geometry, but no key reaches a
// different part: sm_120 was once aliased to SM121, and now has its own
// entry. Add an entry from a measurement or a cited NVIDIA table, not because
// a caller asks for it. TestEveryAliasNamesItsOwnArchitecture enforces this.
var knownArchitectures = map[string]DeviceProperties{
	"sm_70":   SM70,
	"sm70":    SM70,
	"sm_75":   SM75,
	"sm75":    SM75,
	"sm_80":   SM80,
	"sm80":    SM80,
	"sm_86":   SM86,
	"sm86":    SM86,
	"sm_87":   SM87,
	"sm87":    SM87,
	"sm_89":   SM89,
	"sm89":    SM89,
	"sm_90":   SM90,
	"sm90":    SM90,
	"sm_90a":  SM90,
	"sm90a":   SM90,
	"sm_100":  SM100,
	"sm100":   SM100,
	"sm_100a": SM100,
	"sm100a":  SM100,
	"gb200":   SM100,
	"sm_103":  SM103,
	"sm103":   SM103,
	"sm_103a": SM103,
	"sm103a":  SM103,
	"gb300":   SM103,
	"sm_110":  SM110,
	"sm110":   SM110,
	"sm_110a": SM110,
	"sm110a":  SM110,
	"sm_120":  SM120,
	"sm120":   SM120,
	"sm_120a": SM120,
	"sm120a":  SM120,
	"sm_121":  SM121,
	"sm121":   SM121,
	"sm_121a": SM121,
	"sm121a":  SM121,
	"gb10":    SM121,
}

// init records which architectures have a probed device behind them.
// TestHandTableAgreesWithProbedDevices checks that the values themselves
// match; this records that somebody looked.
func init() {
	for arch, dev := range knownArchitectures {
		rec, ok := cudadevicetable.Lookup(arch)
		if !ok || rec.Probe == nil {
			continue
		}
		dev.Source = fmt.Sprintf("%s (%s, %s)", rec.Probe.Device, rec.Probe.Host, rec.Probe.ProbedAt)
		knownArchitectures[arch] = dev
	}
}

// LookupProbedDevice returns device properties only for an architecture a real
// device has been probed for. Use it where a wrong value would flow onward
// silently -- an occupancy ceiling, a launch geometry, a headline ratio --
// because it turns a plausible datasheet number into an error.
//
// [LookupDevice] remains the right call for advisory output, which is honest
// about being advisory.
func LookupProbedDevice(arch string) (DeviceProperties, error) {
	dev, err := LookupDevice(arch)
	if err != nil {
		return DeviceProperties{}, err
	}
	if dev.Source == "" {
		return DeviceProperties{}, fmt.Errorf("cudaoccupancy: %s has not been probed: its properties are hand-entered and no device has confirmed them", arch)
	}
	return dev, nil
}

// LookupDevice returns device properties for the requested architecture string.
// It accepts formats like "sm_90", "sm90", "sm_121a", "SM_80", etc.
func LookupDevice(arch string) (DeviceProperties, error) {
	norm := strings.ToLower(strings.TrimSpace(arch))
	if dev, ok := knownArchitectures[norm]; ok {
		return dev, nil
	}
	return DeviceProperties{}, fmt.Errorf("cudaoccupancy: unknown architecture %q", arch)
}
