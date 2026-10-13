package cudapipeline

import (
	"fmt"
)

// SM121 (GB10) Hardware Limits:
// - Max opt-in shared memory per block: 101,376 bytes (102,400 bytes total per SM)
// - Max warps per SM: 48
// - Max blocks per SM: 24
// - Register allocation quantum: 256 registers per warp (65,536 registers per SM)
const (
	SM121MaxOptinSharedMemBytes = 101376 // 99 KB max opt-in
	SM121TotalSharedMemBytes    = 102400 // 100 KB total SM smem
	SM121MaxWarpsPerSM          = 48
	SM121MaxBlocksPerSM         = 24
	SM121RegAllocQuantumPerWarp = 256
	SM121TotalRegsPerSM         = 65536
	ThreadsPerWarp              = 32
)

// ResourceSummary provides a factual diagnostic breakdown of physical resource
// consumption for search and validation without performance claims.
// A derived quantity that cannot be represented as int is left at zero and
// reported as unavailable in the validation errors; it is not a zero allocation.
// Negative inputs leave all derived quantities unavailable at zero.
type ResourceSummary struct {
	SharedMemBytes      int
	WarpsPerBlock       int
	BlocksPerSM         int
	RegistersPerThread  int
	RegistersPerWarp    int
	TotalWarpsPerSM     int
	TotalSharedMemPerSM int
	TotalRegsPerSM      int
	WithinLimits        bool
	Diagnostic          string
}

// ValidateSM121Resources validates nonnegative resource quantities against
// strict SM121 (GB10) architectural limits. Zero quantities are allowed; this
// checks resource demand, not whether an occupancy configuration can be launched.
// Unrepresentable products or quantum rounding are rejected, not saturated.
// Limits:
//   - Shared memory per block must not exceed 101,376 B opt-in limit (and total SM smem <= 102,400 B).
//   - Warps per SM must not exceed 48 across active blocks.
//   - Blocks per SM must not exceed 24.
//   - Register allocation quantum must allocate in multiples of 256 registers per warp.
func ValidateSM121Resources(smemBytesPerBlock int, warpsPerBlock int, blocksPerSM int, registersPerThread int) (ResourceSummary, []ValidationError) {
	summary := ResourceSummary{
		SharedMemBytes:     smemBytesPerBlock,
		WarpsPerBlock:      warpsPerBlock,
		BlocksPerSM:        blocksPerSM,
		RegistersPerThread: registersPerThread,
	}

	var violations []ValidationError
	for _, input := range []struct {
		name      string
		value     int
		violation string
	}{
		{"shared memory per block", smemBytesPerBlock, ErrExceededSMEMLimit},
		{"warps per block", warpsPerBlock, ErrExceededWarpsPerSM},
		{"blocks per SM", blocksPerSM, ErrExceededBlocksPerSM},
		{"registers per thread", registersPerThread, ErrInvalidRegisterQuantum},
	} {
		if input.value < 0 {
			violations = append(violations, ValidationError{
				Violation: input.violation,
				Expected:  0,
				Actual:    int64(input.value),
				Message:   fmt.Sprintf("%s must be nonnegative, got %d; derived quantities unavailable", input.name, input.value),
			})
		}
	}
	if len(violations) != 0 {
		summary.Diagnostic = fmt.Sprintf("SM121 infeasible (%d violations): %s", len(violations), violations[0].Message)
		return summary, violations
	}

	maxInt := int(^uint(0) >> 1)
	overflow := func(violation, message string) {
		violations = append(violations, ValidationError{
			Violation: violation,
			Expected:  int64(maxInt),
			Message:   message + "; derived quantity unavailable (cannot be represented as int)",
		})
	}

	// Shared memory validation
	if smemBytesPerBlock > SM121MaxOptinSharedMemBytes {
		violations = append(violations, ValidationError{
			Violation: ErrExceededSMEMLimit,
			Expected:  int64(SM121MaxOptinSharedMemBytes),
			Actual:    int64(smemBytesPerBlock),
			Message:   fmt.Sprintf("requested shared memory %d bytes exceeds SM121 max opt-in limit of %d bytes", smemBytesPerBlock, SM121MaxOptinSharedMemBytes),
		})
	}
	totalSmem, smemOK := resourceProduct(smemBytesPerBlock, blocksPerSM)
	summary.TotalSharedMemPerSM = totalSmem
	if !smemOK {
		overflow(ErrExceededSMEMLimit, fmt.Sprintf("total shared memory product %d * %d overflows", smemBytesPerBlock, blocksPerSM))
	} else if totalSmem > SM121TotalSharedMemBytes {
		violations = append(violations, ValidationError{
			Violation: ErrExceededSMEMLimit,
			Expected:  int64(SM121TotalSharedMemBytes),
			Actual:    int64(totalSmem),
			Message:   fmt.Sprintf("total shared memory across %d blocks (%d bytes) exceeds SM total capacity of %d bytes", blocksPerSM, totalSmem, SM121TotalSharedMemBytes),
		})
	}

	// Block count validation
	if blocksPerSM > SM121MaxBlocksPerSM {
		violations = append(violations, ValidationError{
			Violation: ErrExceededBlocksPerSM,
			Expected:  int64(SM121MaxBlocksPerSM),
			Actual:    int64(blocksPerSM),
			Message:   fmt.Sprintf("blocks per SM %d exceeds SM121 limit of %d", blocksPerSM, SM121MaxBlocksPerSM),
		})
	}

	// Warp count validation
	totalWarps, warpsOK := resourceProduct(warpsPerBlock, blocksPerSM)
	summary.TotalWarpsPerSM = totalWarps
	if !warpsOK {
		overflow(ErrExceededWarpsPerSM, fmt.Sprintf("total warps product %d * %d overflows", warpsPerBlock, blocksPerSM))
	} else if totalWarps > SM121MaxWarpsPerSM {
		violations = append(violations, ValidationError{
			Violation: ErrExceededWarpsPerSM,
			Expected:  int64(SM121MaxWarpsPerSM),
			Actual:    int64(totalWarps),
			Message:   fmt.Sprintf("total warps per SM %d exceeds SM121 limit of %d", totalWarps, SM121MaxWarpsPerSM),
		})
	}

	// Register allocation quantum: 256 registers per warp
	// Each warp has 32 threads. Raw registers needed = registersPerThread * 32.
	// Hardware rounds up to 256-register quantum.
	rawRegsPerWarp, regsOK := resourceProduct(registersPerThread, ThreadsPerWarp)
	allocatedRegsPerWarp := rawRegsPerWarp
	if !regsOK {
		overflow(ErrInvalidRegisterQuantum, fmt.Sprintf("registers per warp product %d * %d overflows", registersPerThread, ThreadsPerWarp))
	} else if rem := rawRegsPerWarp % SM121RegAllocQuantumPerWarp; rem != 0 {
		padding := SM121RegAllocQuantumPerWarp - rem
		if rawRegsPerWarp > maxInt-padding {
			regsOK = false
			allocatedRegsPerWarp = 0
			overflow(ErrInvalidRegisterQuantum, fmt.Sprintf("registers per warp quantum rounding of %d to %d overflows", rawRegsPerWarp, SM121RegAllocQuantumPerWarp))
		} else {
			allocatedRegsPerWarp += padding
		}
	}
	summary.RegistersPerWarp = allocatedRegsPerWarp
	if regsOK && warpsOK {
		totalRegs, totalRegsOK := resourceProduct(allocatedRegsPerWarp, totalWarps)
		summary.TotalRegsPerSM = totalRegs
		if !totalRegsOK {
			overflow(ErrInvalidRegisterQuantum, fmt.Sprintf("total register allocation product %d * %d overflows", allocatedRegsPerWarp, totalWarps))
		} else if totalRegs > SM121TotalRegsPerSM {
			violations = append(violations, ValidationError{
				Violation: ErrInvalidRegisterQuantum,
				Expected:  int64(SM121TotalRegsPerSM),
				Actual:    int64(totalRegs),
				Message:   fmt.Sprintf("total register allocation %d registers (quantum %d/warp) exceeds SM121 limit of %d", totalRegs, allocatedRegsPerWarp, SM121TotalRegsPerSM),
			})
		}
	} else {
		violations = append(violations, ValidationError{
			Violation: ErrInvalidRegisterQuantum,
			Message:   "total register allocation unavailable because a required warp quantity cannot be represented as int",
		})
	}

	summary.WithinLimits = len(violations) == 0
	if summary.WithinLimits {
		summary.Diagnostic = fmt.Sprintf("SM121 feasible: %d B smem/block, %d warps/SM, %d blocks/SM, %d regs/warp",
			smemBytesPerBlock, totalWarps, blocksPerSM, allocatedRegsPerWarp)
	} else {
		summary.Diagnostic = fmt.Sprintf("SM121 infeasible (%d violations): %s",
			len(violations), violations[0].Message)
	}

	return summary, violations
}

// resourceProduct multiplies nonnegative resource quantities without wrapping.
func resourceProduct(a, b int) (int, bool) {
	maxInt := int(^uint(0) >> 1)
	if a < 0 || b < 0 || (b != 0 && a > maxInt/b) {
		return 0, false
	}
	return a * b, true
}
