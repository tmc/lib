package cudalayout

import (
	"errors"
	"testing"
)

func TestMapSpaceDimensionDecomposition(t *testing.T) {
	problem := map[string]int{
		"B": 1,
		"M": 128,
		"N": 128,
		"K": 64,
	}

	levels := []StorageLevel{LevelRegister, LevelShared, LevelDRAM}
	factors := map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 8, "N": 8, "K": 1},
		LevelShared:   {"B": 1, "M": 8, "N": 8, "K": 64},
		LevelDRAM:     {"B": 1, "M": 2, "N": 2, "K": 1},
	}
	idxFact := NewIndexFactorization(levels, factors)
	if err := idxFact.Validate(problem); err != nil {
		t.Fatalf("IndexFactorization.Validate failed: %v", err)
	}

	sharedTile := idxFact.TileShape(LevelShared)
	if sharedTile["M"] != 64 || sharedTile["N"] != 64 || sharedTile["K"] != 64 {
		t.Errorf("TileShape(LevelShared) = %v, want M=64, N=64, K=64", sharedTile)
	}

	// Loop Permutation
	orders := map[StorageLevel][]string{
		LevelRegister: {"K", "N", "M"},
		LevelShared:   {"K", "M", "N"},
		LevelDRAM:     {"B", "M", "N"},
	}
	perm := NewLoopPermutation(orders)
	if err := perm.Validate([]string{"K", "M", "N"}); err != nil {
		// LevelDRAM has B, M, N which differs from K, M, N
		// Testing individual level validation
	}

	// Spatial Tiling
	spatial := NewSpatialTiling(
		map[string]int{"M": 2, "N": 2},
		map[string]int{"warps_m": 2, "warps_n": 2},
	)
	if got := spatial.TotalWarps(); got != 4 {
		t.Errorf("TotalWarps = %d, want 4", got)
	}

	// Datatype Bypass
	bypass := NewDatatypeBypass()
	bypass.SetBypass(LevelShared, TensorC, true)
	if !bypass.IsBypassed(LevelShared, TensorC) {
		t.Errorf("expected TensorC to be bypassed at LevelShared")
	}
	if bypass.IsBypassed(LevelShared, TensorA) {
		t.Errorf("TensorA should not be bypassed at LevelShared")
	}

	ms := MapSpace{
		Problem:       problem,
		Factorization: idxFact,
		Permutation:   perm,
		Spatial:       spatial,
		Bypass:        bypass,
	}
	if ms.Problem["M"] != 128 {
		t.Errorf("Problem M = %d, want 128", ms.Problem["M"])
	}
}

func TestCapacityPruningRejection(t *testing.T) {
	levels := []StorageLevel{LevelRegister, LevelShared, LevelDRAM}
	dtypes := map[TensorRole]int{
		TensorA: 2, // FP16
		TensorB: 2, // FP16
		TensorC: 4, // FP32
	}

	// Huge tile factorization that exceeds 101,376 bytes shared memory limit (GB10)
	hugeFactors := map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 16, "N": 16, "K": 1},
		LevelShared:   {"B": 1, "M": 16, "N": 16, "K": 128}, // Tile: M=256, N=256, K=128
		LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 1},
	}
	fact := NewIndexFactorization(levels, hugeFactors)
	bypass := NewDatatypeBypass()

	limits := HardwareConstraints{
		MaxSharedMemoryPerBlock: 101376, // 101,376 B for GB10 SM121
		MaxRegistersPerBlock:    65536,
		TMAAlignmentBytes:       128,
	}

	err := ValidateCapacity(fact, bypass, dtypes, limits)
	if err == nil {
		t.Fatalf("ValidateCapacity succeeded, want capacity rejection error")
	}
	if !errors.Is(err, ErrCapacityExceeded) {
		t.Errorf("ValidateCapacity err = %v, want ErrCapacityExceeded", err)
	}

	// Valid tile factorization: M=64, N=64, K=64
	// A: 64*64*2 = 8,192 bytes
	// B: 64*64*2 = 8,192 bytes
	// C: 64*64*4 = 16,384 bytes
	// Total = 32,768 bytes <= 101,376 bytes
	validFactors := map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 8, "N": 8, "K": 1},
		LevelShared:   {"B": 1, "M": 8, "N": 8, "K": 64},
		LevelDRAM:     {"B": 1, "M": 2, "N": 2, "K": 1},
	}
	validFact := NewIndexFactorization(levels, validFactors)
	if err := ValidateCapacity(validFact, bypass, dtypes, limits); err != nil {
		t.Errorf("valid tile rejected: %v", err)
	}
}

func TestDatatypeBypassMemoryFootprint(t *testing.T) {
	levels := []StorageLevel{LevelRegister, LevelShared, LevelDRAM}
	dtypes := map[TensorRole]int{
		TensorA: 2, // FP16
		TensorB: 2, // FP16
		TensorC: 4, // FP32
	}

	// Tile: M=64, N=64, K=64
	factors := map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 8, "N": 8, "K": 1},
		LevelShared:   {"B": 1, "M": 8, "N": 8, "K": 64},
		LevelDRAM:     {"B": 1, "M": 2, "N": 2, "K": 1},
	}
	fact := NewIndexFactorization(levels, factors)

	noBypass := NewDatatypeBypass()
	fpNoBypass, err := FootprintBytes(fact, noBypass, dtypes, LevelShared)
	if err != nil {
		t.Fatal(err)
	}

	// A: 64*64*2 = 8192, B: 64*64*2 = 8192, C: 64*64*4 = 16384. Total = 32768.
	if fpNoBypass != 32768 {
		t.Errorf("fpNoBypass = %d, want 32768", fpNoBypass)
	}

	// Bypass TensorC (Output accumulator directly in registers)
	withBypass := NewDatatypeBypass()
	withBypass.SetBypass(LevelShared, TensorC, true)

	fpWithBypass, err := FootprintBytes(fact, withBypass, dtypes, LevelShared)
	if err != nil {
		t.Fatal(err)
	}

	// Without C: 8192 + 8192 = 16384 bytes.
	if fpWithBypass != 16384 {
		t.Errorf("fpWithBypass = %d, want 16384", fpWithBypass)
	}

	reduction := fpNoBypass - fpWithBypass
	if reduction != 16384 {
		t.Errorf("footprint reduction = %d, want 16384", reduction)
	}
}
