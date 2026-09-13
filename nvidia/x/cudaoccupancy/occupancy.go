package cudaoccupancy

import "fmt"

// LimitingFactor constants identify which hardware constraint determines theoretical occupancy.
const (
	LimitRegisters    = "registers"
	LimitSharedMemory = "shared_memory"
	LimitBlockSize    = "block_size"
	LimitBlocks       = "blocks"
	LimitNone         = "none"
)

// Report records theoretical occupancy and hardware bottleneck analysis for a kernel configuration.
type Report struct {
	Architecture     string  // Target GPU architecture (e.g. "sm_90", "sm_121")
	TheoreticalWarps int     // Theoretical active warps per SM
	MaxWarps         int     // Architecture maximum warps per SM
	Occupancy        float64 // Theoretical occupancy ratio [0.0, 1.0]
	ActiveBlocks     int     // Resident thread blocks (CTAs) per SM
	LimitingFactor   string  // Primary bottleneck: "registers", "shared_memory", "block_size", "blocks", "none"
	BlockLimit       int     // Ceiling imposed by max blocks per SM
	WarpLimit        int     // Ceiling imposed by max warps per SM
	RegLimit         int     // Ceiling imposed by register file capacity
	SmemLimit        int     // Ceiling imposed by shared memory capacity
}

// Calculate determines theoretical occupancy on the named GPU architecture.
func (k KernelResource) Calculate(arch string, threadsPerBlock, dynamicSmemBytes int) (Report, error) {
	dev, err := LookupDevice(arch)
	if err != nil {
		return Report{}, err
	}
	return k.CalculateDevice(dev, threadsPerBlock, dynamicSmemBytes), nil
}

// CalculateDevice determines theoretical occupancy on a specific device specification.
func (k KernelResource) CalculateDevice(dev DeviceProperties, threadsPerBlock, dynamicSmemBytes int) Report {
	rep := Report{
		Architecture: dev.Architecture,
		MaxWarps:     dev.MaxWarpsPerSM,
		BlockLimit:   dev.MaxBlocksPerSM,
	}

	if threadsPerBlock <= 0 || threadsPerBlock > dev.MaxThreadsPerBlock {
		rep.LimitingFactor = LimitBlockSize
		return rep
	}

	warpsPerBlock := (threadsPerBlock + dev.WarpSize - 1) / dev.WarpSize
	rep.WarpLimit = dev.MaxWarpsPerSM / warpsPerBlock

	// Register allocation ceiling
	rep.RegLimit = dev.MaxBlocksPerSM
	if k.Registers > 0 {
		regsPerWarp := roundUp(k.Registers*dev.WarpSize, dev.RegAllocationUnit)
		regsPerBlock := warpsPerBlock * regsPerWarp
		if regsPerBlock > 0 {
			rep.RegLimit = dev.MaxRegistersPerSM / regsPerBlock
		}
	}

	// Shared memory allocation ceiling
	rep.SmemLimit = dev.MaxBlocksPerSM
	totalSmem := k.StaticSharedMem + dynamicSmemBytes
	if totalSmem > 0 {
		alignedSmem := roundUp(totalSmem, dev.SmemAllocationUnit)
		maxBlockSmem := dev.MaxSharedMemoryOptin
		if maxBlockSmem == 0 {
			maxBlockSmem = dev.MaxSharedMemoryPerBlock
		}
		if alignedSmem > maxBlockSmem {
			rep.SmemLimit = 0
			rep.LimitingFactor = LimitSharedMemory
			return rep
		}
		rep.SmemLimit = dev.MaxSharedMemoryPerSM / alignedSmem
	}

	activeBlocks := min(rep.BlockLimit, min(rep.WarpLimit, min(rep.RegLimit, rep.SmemLimit)))
	rep.ActiveBlocks = activeBlocks
	rep.TheoreticalWarps = activeBlocks * warpsPerBlock
	if dev.MaxWarpsPerSM > 0 {
		rep.Occupancy = float64(rep.TheoreticalWarps) / float64(dev.MaxWarpsPerSM)
	}

	// Determine primary limiting factor
	switch {
	case rep.TheoreticalWarps == dev.MaxWarpsPerSM && dev.MaxWarpsPerSM > 0:
		rep.LimitingFactor = LimitNone
	case activeBlocks == rep.SmemLimit && rep.SmemLimit < rep.WarpLimit && rep.SmemLimit < rep.RegLimit && rep.SmemLimit < rep.BlockLimit:
		rep.LimitingFactor = LimitSharedMemory
	case activeBlocks == rep.RegLimit && rep.RegLimit < rep.WarpLimit && rep.RegLimit < rep.BlockLimit:
		rep.LimitingFactor = LimitRegisters
	case activeBlocks == rep.WarpLimit && rep.WarpLimit < rep.BlockLimit:
		rep.LimitingFactor = LimitBlockSize
	case activeBlocks == rep.BlockLimit && rep.TheoreticalWarps < dev.MaxWarpsPerSM:
		rep.LimitingFactor = LimitBlocks
	default:
		if rep.WarpLimit <= rep.BlockLimit {
			rep.LimitingFactor = LimitBlockSize
		} else {
			rep.LimitingFactor = LimitBlocks
		}
	}

	return rep
}

// String returns a concise human-readable summary of the occupancy report.
func (r Report) String() string {
	return fmt.Sprintf("%s: %d/%d warps (%.1f%%, %d blocks/SM) limited by %s",
		r.Architecture, r.TheoreticalWarps, r.MaxWarps, r.Occupancy*100, r.ActiveBlocks, r.LimitingFactor)
}

func roundUp(x, unit int) int {
	if unit <= 0 {
		return x
	}
	return ((x + unit - 1) / unit) * unit
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
