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
var (
	// Volta V100 (SM 7.0)
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

	// Turing T4 / RTX 2080 (SM 7.5)
	SM75 = DeviceProperties{
		Architecture:            "sm_75",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           32,
		MaxBlocksPerSM:          16,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    65536,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    65536,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
	}

	// Ampere A100 (SM 8.0)
	SM80 = DeviceProperties{
		Architecture:            "sm_80",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           64,
		MaxBlocksPerSM:          32,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    167936,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    166912,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
	}

	// Ampere RTX 3080/3090 / A10 (SM 8.6)
	SM86 = DeviceProperties{
		Architecture:            "sm_86",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           48,
		MaxBlocksPerSM:          16,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    102400,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    101376,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
	}

	// Ada Lovelace RTX 4090 / L40 (SM 8.9)
	SM89 = DeviceProperties{
		Architecture:            "sm_89",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           48,
		MaxBlocksPerSM:          24,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    102400,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    101376,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
	}

	// Ampere Jetson Orin (SM 8.7). CUDA Programming Guide 13.x, Compute
	// Capabilities, Tables 30 and 31: 16 blocks, 48 warps, 164 KB per SM,
	// 163 KB per block.
	SM87 = DeviceProperties{
		Architecture:            "sm_87",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           48,
		MaxBlocksPerSM:          16,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    167936,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    166912,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
	}

	// Hopper H100 (SM 9.0)
	SM90 = DeviceProperties{
		Architecture:            "sm_90",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           64,
		MaxBlocksPerSM:          32,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    233472,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    232448,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
		MaxClusterSize:          16,
	}

	// Blackwell GB200 (SM 10.0 / SM 10.0a)
	SM100 = DeviceProperties{
		Architecture:            "sm_100",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           64,
		MaxBlocksPerSM:          32,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    233472,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    232448,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
		TMEMColumnsPerSM:        512,
		MaxClusterSize:          16,
	}

	// Blackwell Ultra GB300 / B300 (SM 10.3 / SM 10.3a). CUDA Programming
	// Guide 13.x, Tables 30 and 31 give 10.3 the limits of 10.0; PTX ISA,
	// Tensor Memory, gives 512 columns. The non-portable cluster size of 16
	// is documented for B200 and assumed unchanged.
	SM103 = DeviceProperties{
		Architecture:            "sm_103",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           64,
		MaxBlocksPerSM:          32,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    233472,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    232448,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
		TMEMColumnsPerSM:        512,
		MaxClusterSize:          16,
	}

	// Jetson Thor (SM 11.0, formerly SM 10.1). CUDA Programming Guide
	// 13.x, Tables 30 and 31: 24 blocks, 48 warps, 228 KB per SM, 227 KB
	// per block. sm_110a assembles tcgen05 (cudadevicetable feature probe),
	// so TMEM is assumed to be the 512 columns of 10.x; the cluster limit
	// is the portable 8.
	SM110 = DeviceProperties{
		Architecture:            "sm_110",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           48,
		MaxBlocksPerSM:          24,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    233472,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    232448,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
		TMEMColumnsPerSM:        512,
		MaxClusterSize:          8,
	}

	// Blackwell RTX PRO 6000 / RTX 5090 (SM 12.0). CUDA Programming Guide
	// 13.x, Tables 30 and 31 give 12.x one set of limits; these are the
	// probed GB10 values below, which agree with them.
	SM120 = DeviceProperties{
		Architecture:            "sm_120",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           48,
		MaxBlocksPerSM:          24,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    102400,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    101376,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
		MaxClusterSize:          8,
	}

	// Blackwell GB10 (SM 12.1)
	// Provenance: measured on physical NVIDIA GB10 hardware via cudaGetDeviceProperties /
	// cudaDeviceGetAttribute (CUDA 13.0.85, driver 580.173.02):
	// SMs=48, maxThreadsPerSM=1536 (48 warps/SM), regsPerSM=65536, smemPerSM=102400,
	// smemPerBlock=49152, smemOptin=101376 (102400 - 1024 B), maxBlocksPerSM=24.
	// Note: GB200 is a different architecture (sm_100a, 64 warps/SM, tcgen05).
	SM121 = DeviceProperties{
		Architecture:            "sm_121",
		MaxThreadsPerBlock:      1024,
		WarpSize:                32,
		MaxWarpsPerSM:           48,
		MaxBlocksPerSM:          24,
		MaxRegistersPerSM:       65536,
		MaxSharedMemoryPerSM:    102400,
		MaxSharedMemoryPerBlock: 49152,
		MaxSharedMemoryOptin:    101376,
		RegAllocationUnit:       256,
		SmemAllocationUnit:      128,
		MaxClusterSize:          8,
	}
)

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
