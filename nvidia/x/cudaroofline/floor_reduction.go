package cudaroofline

import (
	"fmt"
	"maps"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

// GEMMReductionRound describes one adjacent-pair reduction of all output cells.
// Every input is read once and every output written once, including an odd
// unpaired input copied without addition. The final round writes final C.
// Bytes are explicit direct DRAM requests by one serial logical merge owner;
// no intermediate retained buffer or reduction network is assumed.
type GEMMReductionRound struct {
	InputPartials, OutputPartials uint64
	ReadBytes, WriteBytes         uint64
	Additions                     uint64
}

// GEMMSplitKMovement describes exact traffic/capacity for the selected recipe,
// not a universal minimum, hardware behavior or a produced native executor.
// GEMM contains the original owner-major phase: its C cells are tagged by K
// owner and reside in scratch. Its per-instance/union quantities cover only
// that phase. DRAM includes both phases; its intensity uses useful 2*MACs.
// ExecutedFLOPs additionally includes the declared tree additions.
//
// Merge uses one logical scalar owner and two C-width registers, overwriting
// the left operand with the result. It bypasses all intermediate storage,
// independently of GEMM's bypass mapping. Rounds iterate output cells in
// B/M/N lexical order, then adjacent partial pairs, with a barrier per round.
// The tree's reassociation is not a bitwise floating-point guarantee.
//
// WorkspaceBytes excludes final C (OutputBytes) and original A/B. For more
// than two partials it reserves two disjoint arrays, S and ceil(S/2) outputs
// per cell; they alternate after barriers and remain allocated through the
// last round. Two partials need only the input scratch array. Partial cells
// use q-major offsets q*(B*M*N)+p. Stale nodes after a barrier are not live.
// PeakLivePartialBytes counts tagged partial cells, excluding final C.
//
// With one partial, GEMM equals AnalyticalGEMMSpatial and writes final C;
// Rounds, workspace, live partials, merge registers and extra additions are zero.
// Inputs are borrowed during analysis; returned maps/slices are caller-owned.
type GEMMSplitKMovement struct {
	GEMM                 SpatialHierarchyMovement
	Rounds               []GEMMReductionRound
	DRAM                 CompoundDataMovement
	Partials             uint64
	OutputBytes          uint64
	WorkspaceBytes       uint64
	PeakLivePartialBytes uint64
	MergeRegisterBytes   uint64
	ReductionAdditions   uint64
	ExecutedFLOPs        uint64
}

// AnalyticalGEMMSplitK analyzes a fixed two-phase recipe without enumerating
// MACs, outputs or owners. The original CTA/warp/lane digit partition defines
// q=((ctaK*warpKCount)+warpK)*laneKCount+laneK. Each C buffer retains one full
// projected tile tagged with q; first visits initialize zero, revisits reload.
// DRAM retains all S partial tensors. A/B follow the original retention rules.
// After the GEMM flush/barrier, adjacent partials are added left then right;
// odd inputs copy unchanged. The last round writes final C exactly once.
//
// This computes exact selected-recipe logical traffic/capacity, not universal
// minimum traffic, theoretical hardware evidence, native timing or device
// memory admission. Byte sizes declare the partial/output storage width;
// numerical qualification and actual register allocation remain unobserved.
// Multi-partial backing sizes must fit the host int; no backing buffer is
// allocated. Counts/offsets use checked arithmetic and rounds are word-bounded.
func AnalyticalGEMMSplitK(mapping cudalayout.MapSpace, dtypes map[cudalayout.TensorRole]int, profile GEMMSpatialProfile) (GEMMSplitKMovement, error) {
	gemm, err := analyticalGEMMSpatial(mapping, dtypes, profile, true)
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	partials, err := spatialKParts(mapping, profile)
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	outputs, err := checkedProduct(uint64(mapping.Problem["B"]), uint64(mapping.Problem["M"]), uint64(mapping.Problem["N"]))
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	outputBytes, err := checkedProduct(outputs, uint64(dtypes[cudalayout.TensorC]))
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	flops, err := checkedProduct(2, outputs, uint64(mapping.Problem["K"]))
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	result := GEMMSplitKMovement{GEMM: gemm, Partials: partials, OutputBytes: outputBytes, ExecutedFLOPs: flops, DRAM: gemm.Movement.DRAM}
	result.DRAM.TensorTransfers = maps.Clone(result.DRAM.TensorTransfers)
	if partials == 1 {
		return result, nil
	}
	slots := partials
	if partials > 2 {
		slots, err = checkedSum(slots, partials/2+partials%2)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
	}
	result.WorkspaceBytes, err = checkedProduct(slots, outputBytes)
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	backing, err := checkedSum(result.WorkspaceBytes, outputBytes)
	if err != nil || backing > uint64(^uint(0)>>1) {
		return GEMMSplitKMovement{}, fmt.Errorf("split-K backing does not fit int: %w", cudalayout.ErrOverflow)
	}
	result.PeakLivePartialBytes = result.WorkspaceBytes
	result.MergeRegisterBytes, err = checkedProduct(2, uint64(dtypes[cudalayout.TensorC]))
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	for n := partials; n > 1; {
		next := n/2 + n%2
		round := GEMMReductionRound{InputPartials: n, OutputPartials: next}
		round.ReadBytes, err = checkedProduct(n, outputBytes)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
		round.WriteBytes, err = checkedProduct(next, outputBytes)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
		round.Additions, err = checkedProduct(n/2, outputs)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
		result.ReductionAdditions, err = checkedSum(result.ReductionAdditions, round.Additions)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
		result.DRAM.ReadBytes, err = checkedSum(result.DRAM.ReadBytes, round.ReadBytes)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
		result.DRAM.WriteBytes, err = checkedSum(result.DRAM.WriteBytes, round.WriteBytes)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
		c := result.DRAM.TensorTransfers["C"]
		c, err = checkedSum(c, round.ReadBytes, round.WriteBytes)
		if err != nil {
			return GEMMSplitKMovement{}, err
		}
		result.DRAM.TensorTransfers["C"] = c
		result.Rounds = append(result.Rounds, round)
		n = next
	}
	result.DRAM.TotalBytes, err = checkedSum(result.DRAM.ReadBytes, result.DRAM.WriteBytes)
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	result.ExecutedFLOPs, err = checkedSum(flops, result.ReductionAdditions)
	if err != nil {
		return GEMMSplitKMovement{}, err
	}
	result.DRAM.OperationalIntensity = float64(flops) / float64(result.DRAM.TotalBytes)
	return result, nil
}

// Call only after spatialGEMMLoops has validated positive mesh dimensions.
func spatialKParts(mapping cudalayout.MapSpace, profile GEMMSpatialProfile) (uint64, error) {
	parts := uint64(1)
	for _, mesh := range []map[string]int{mapping.Spatial.CTAMesh, mapping.Spatial.WarpMesh, profile.LaneMesh} {
		if n, ok := mesh["K"]; ok {
			var err error
			parts, err = checkedProduct(parts, uint64(n))
			if err != nil {
				return 0, err
			}
		}
	}
	return parts, nil
}
