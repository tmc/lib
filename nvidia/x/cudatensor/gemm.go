package cudatensor

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// GEMMWorkload contains only the dimension names and tensor roles accepted by
// cudaroofline.AnalyticalDataMovement.
type GEMMWorkload struct {
	Problem      map[string]int
	RankNames    map[cudashape.DimToken]string
	OperandRoles []string
	OutputRole   string
}

// MapCanonicalGEMM maps the supported canonical matrix multiplication
// equations to B/M/N/K and A/B/C. It refuses equations or ranks outside the
// GEMM contract instead of forwarding unknown labels to cudaroofline.
func MapCanonicalGEMM(equation string, extents map[cudashape.DimToken]int) (GEMMWorkload, error) {
	inputs, output, err := cudashape.ParseEinsumEquation(equation)
	if err != nil {
		return GEMMWorkload{}, fmt.Errorf("cudatensor: GEMM equation %q: %w", equation, err)
	}
	if len(inputs) != 2 {
		return GEMMWorkload{}, unsupportedGEMM(equation, "expected two input operands")
	}
	roles := make(map[cudashape.DimToken]string)
	var b, m, n, k cudashape.DimToken
	switch {
	case len(inputs[0]) == 2 && len(inputs[1]) == 2 && len(output) == 2:
		if inputs[0][1] != inputs[1][0] || output[0] != inputs[0][0] || output[1] != inputs[1][1] {
			return GEMMWorkload{}, unsupportedGEMM(equation, "ranks do not form MxK by KxN to MxN")
		}
		m, k, n = inputs[0][0], inputs[0][1], inputs[1][1]
	case len(inputs[0]) == 3 && len(inputs[1]) == 3 && len(output) == 3:
		if inputs[0][0] != inputs[1][0] || inputs[0][2] != inputs[1][1] || output[0] != inputs[0][0] || output[1] != inputs[0][1] || output[2] != inputs[1][2] {
			return GEMMWorkload{}, unsupportedGEMM(equation, "ranks do not form batched BMK by BKN to BMN")
		}
		b, m, k, n = inputs[0][0], inputs[0][1], inputs[0][2], inputs[1][2]
	default:
		return GEMMWorkload{}, unsupportedGEMM(equation, "only rank-2 and batched rank-3 GEMM are supported")
	}
	roles[m], roles[k], roles[n] = "M", "K", "N"
	problem := map[string]int{}
	if b != "" {
		roles[b] = "B"
	}
	for _, rank := range sortedDimTokens(extents) {
		if _, ok := roles[rank]; !ok {
			return GEMMWorkload{}, fmt.Errorf("cudatensor: GEMM equation %q has unmapped dimension %q", equation, rank)
		}
	}
	for _, rank := range []cudashape.DimToken{b, m, n, k} {
		if rank == "" {
			continue
		}
		name := roles[rank]
		extent, ok := extents[rank]
		if !ok {
			return GEMMWorkload{}, fmt.Errorf("cudatensor: GEMM equation %q has no extent for rank %q", equation, rank)
		}
		if extent <= 0 {
			return GEMMWorkload{}, fmt.Errorf("cudatensor: GEMM equation %q has non-positive extent %q=%d", equation, rank, extent)
		}
		problem[name] = extent
	}
	if b == "" {
		problem["B"] = 1
	}
	return GEMMWorkload{Problem: problem, RankNames: roles, OperandRoles: []string{"A", "B"}, OutputRole: "C"}, nil
}

func sortedDimTokens(in map[cudashape.DimToken]int) []cudashape.DimToken {
	out := make([]cudashape.DimToken, 0, len(in))
	for rank := range in {
		out = append(out, rank)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// CanonicalGEMMDataMovement maps a canonical GEMM equation to lane 43's
// accepted labels and invokes its analytical byte model. Mapping failures are
// returned with the original equation for diagnosis.
func CanonicalGEMMDataMovement(equation string, extents map[cudashape.DimToken]int, tileFactors, dtypes map[string]int, bypassed map[string]bool) (cudaroofline.CompoundDataMovement, error) {
	workload, err := MapCanonicalGEMM(equation, extents)
	if err != nil {
		return cudaroofline.CompoundDataMovement{}, err
	}
	movement, err := cudaroofline.AnalyticalDataMovement(workload.Problem, tileFactors, dtypes, bypassed)
	if err != nil {
		return cudaroofline.CompoundDataMovement{}, fmt.Errorf("cudatensor: GEMM equation %q: %w", equation, err)
	}
	return movement, nil
}

func unsupportedGEMM(equation, reason string) error {
	return fmt.Errorf("cudatensor: unsupported GEMM equation %q: %s", equation, reason)
}
