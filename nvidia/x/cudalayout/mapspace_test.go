package cudalayout

import (
	"errors"
	"strings"
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
		LevelRegister: {"B", "K", "N", "M"},
		LevelShared:   {"K", "B", "M", "N"},
		LevelDRAM:     {"B", "M", "N", "K"},
	}
	perm := NewLoopPermutation(orders)
	if err := perm.Validate([]string{"B", "K", "M", "N"}); err != nil {
		t.Fatalf("LoopPermutation.Validate failed: %v", err)
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
		MaxThreadsPerBlock:      128,
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

	registerFact := NewIndexFactorization(levels, map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 16, "N": 16, "K": 1},
		LevelShared:   {"B": 1, "M": 1, "N": 1, "K": 1},
		LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 1},
	})
	registerLimits := HardwareConstraints{MaxSharedMemoryPerBlock: 8192, MaxRegistersPerBlock: 1000, MaxThreadsPerBlock: 128}
	if err := ValidateCapacity(registerFact, bypass, dtypes, registerLimits); !errors.Is(err, ErrCapacityExceeded) || !strings.Contains(err.Error(), "register") {
		t.Fatalf("register capacity error = %v, want ErrCapacityExceeded naming registers", err)
	}

	// This tile uses 16 bytes per thread: 4 32-bit registers, or 512 per block.
	// Counting bytes as registers would incorrectly reject it at a 1024-register limit.
	unitFact := NewIndexFactorization(levels, map[StorageLevel]map[string]int{
		LevelRegister: {"B": 2, "M": 1, "N": 1, "K": 1},
		LevelShared:   {"B": 1, "M": 1, "N": 1, "K": 1},
		LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 1},
	})
	unitLimits := HardwareConstraints{MaxRegistersPerBlock: 1024, MaxThreadsPerBlock: 128}
	if err := ValidateCapacity(unitFact, bypass, dtypes, unitLimits); err != nil {
		t.Fatalf("16 bytes/thread should use 512 registers/block: %v", err)
	}
	unitLimits.MaxRegistersPerBlock = 511
	if err := ValidateCapacity(unitFact, bypass, dtypes, unitLimits); !errors.Is(err, ErrCapacityExceeded) {
		t.Fatalf("512 registers should exceed limit 511: %v", err)
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

func TestIndexFactorizationValidate(t *testing.T) {
	levels := []StorageLevel{LevelRegister, LevelShared, LevelDRAM}
	valid := map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 2, "N": 2, "K": 1},
		LevelShared:   {"B": 1, "M": 2, "N": 2, "K": 2},
		LevelDRAM:     {"B": 1, "M": 2, "N": 2, "K": 1},
	}
	problem := map[string]int{"B": 1, "M": 8, "N": 8, "K": 2}
	tests := []struct {
		name    string
		factors map[StorageLevel]map[string]int
		problem map[string]int
		wantErr bool
	}{
		{name: "valid", factors: valid, problem: problem},
		{name: "wrong product", factors: map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 2, "N": 8, "K": 2}}, problem: problem, wantErr: true},
		{name: "non-positive", factors: map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 0, "N": 8, "K": 2}}, problem: problem, wantErr: true},
		{name: "missing dimension", factors: map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 8, "N": 8}}, problem: problem, wantErr: true},
		{name: "unknown dimension", factors: map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 8, "N": 8, "K": 2, "P": 1}}, problem: problem, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewIndexFactorization(levels, tt.factors).Validate(tt.problem)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate error = %v, wantErr %t", err, tt.wantErr)
			}
		})
	}
}

func TestLoopPermutationValidate(t *testing.T) {
	dimensions := []string{"B", "M", "N", "K"}
	tests := []struct {
		name    string
		order   []string
		wantErr bool
	}{
		{name: "valid", order: []string{"B", "M", "N", "K"}},
		{name: "duplicate", order: []string{"B", "M", "N", "N"}, wantErr: true},
		{name: "unknown", order: []string{"B", "M", "N", "P"}, wantErr: true},
		{name: "missing", order: []string{"B", "M", "N"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := NewLoopPermutation(map[StorageLevel][]string{LevelShared: tt.order}).Validate(dimensions)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Validate error = %v, wantErr %t", err, tt.wantErr)
			}
		})
	}
}

func TestCapacityPruningRejectsTMAMisalignment(t *testing.T) {
	levels := []StorageLevel{LevelRegister, LevelShared, LevelDRAM}
	fact := NewIndexFactorization(levels, map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 1, "N": 1, "K": 1},
		LevelShared:   {"B": 1, "M": 64, "N": 64, "K": 63},
		LevelDRAM:     {"B": 1, "M": 1, "N": 1, "K": 1},
	})
	dtypes := map[TensorRole]int{TensorA: 2, TensorB: 2, TensorC: 4}
	limits := HardwareConstraints{TMAAlignmentBytes: 128}
	if err := ValidateCapacity(fact, NewDatatypeBypass(), dtypes, limits); !errors.Is(err, ErrAlignmentFailed) {
		t.Fatalf("unaligned A row error = %v, want ErrAlignmentFailed", err)
	}
	bypass := NewDatatypeBypass()
	bypass.SetBypass(LevelShared, TensorA, true)
	if err := ValidateCapacity(fact, bypass, dtypes, limits); err != nil {
		t.Fatalf("bypassed A should be exempt from shared TMA alignment: %v", err)
	}
}

func TestFootprintBytesRejectsUnknownInputs(t *testing.T) {
	levels := []StorageLevel{LevelRegister, LevelShared}
	dtypes := map[TensorRole]int{TensorA: 2, TensorB: 2, TensorC: 4}
	tests := []struct {
		name    string
		factors IndexFactorization
		dtypes  map[TensorRole]int
		level   StorageLevel
	}{
		{name: "unknown dimension", factors: NewIndexFactorization(levels, map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 2, "N": 2, "K": 2, "P": 4096}, LevelShared: {}}), dtypes: dtypes, level: LevelShared},
		{name: "missing extent", factors: NewIndexFactorization(levels, map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 2, "N": 2}, LevelShared: {}}), dtypes: dtypes, level: LevelShared},
		{name: "missing dtype", factors: NewIndexFactorization(levels, map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 2, "N": 2, "K": 2}, LevelShared: {}}), dtypes: map[TensorRole]int{TensorA: 2, TensorB: 2}, level: LevelShared},
		{name: "absent level", factors: NewIndexFactorization([]StorageLevel{LevelRegister}, map[StorageLevel]map[string]int{LevelRegister: {"B": 1, "M": 2, "N": 2, "K": 2}}), dtypes: dtypes, level: LevelShared},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := FootprintBytes(tt.factors, NewDatatypeBypass(), tt.dtypes, tt.level); err == nil {
				t.Fatal("FootprintBytes succeeded on invalid inputs")
			}
		})
	}
}

func TestTileShapeAbsentLevel(t *testing.T) {
	factors := NewIndexFactorization([]StorageLevel{LevelRegister}, map[StorageLevel]map[string]int{
		LevelRegister: {"B": 1, "M": 2, "N": 2, "K": 2},
	})
	if tile := factors.TileShape(LevelShared); tile != nil {
		t.Fatalf("TileShape(LevelShared) = %v, want nil for absent level", tile)
	}
}
