package cudaoccupancy

import (
	"fmt"
	"strings"
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
	}
)

// knownArchitectures maps an architecture string to its properties. A key may
// only name the architecture its value describes: "sm_90a" reaches SM90
// because a feature suffix does not change SM geometry, but no key reaches a
// different part. sm_120 (GB20x) was once aliased to SM121; LookupDevice now reports an
// error for that part rather than returning a neighbour's geometry.
// SM100 has its own explicit properties. Add an entry when the part has been measured, not when a caller asks
// for it. TestEveryAliasNamesItsOwnArchitecture enforces this.
var knownArchitectures = map[string]DeviceProperties{
	"sm_70":   SM70,
	"sm70":    SM70,
	"sm_75":   SM75,
	"sm75":    SM75,
	"sm_80":   SM80,
	"sm80":    SM80,
	"sm_86":   SM86,
	"sm86":    SM86,
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
	"sm_121":  SM121,
	"sm121":   SM121,
	"sm_121a": SM121,
	"sm121a":  SM121,
	"gb10":    SM121,
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
