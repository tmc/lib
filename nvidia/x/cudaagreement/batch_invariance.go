// SPDX-License-Identifier: Apache-2.0

package cudaagreement

import (
	"fmt"
	"math"
)

// BatchInvarianceCase compares the solo (batch=1) output of a sequence against
// its output when evaluated at RowIndex within a batch of size BatchSize.
type BatchInvarianceCase struct {
	SeqID         string
	Input         []float32
	SoloOutput    []float32
	BatchedOutput []float32
	BatchSize     int
	RowIndex      int
}

// BatchInvarianceMismatch records the first element where a sequence's batched
// output diverged bitwise from its solo output.
type BatchInvarianceMismatch struct {
	SeqID       string
	BatchSize   int
	RowIndex    int
	ElemIndex   int
	SoloBits    uint32
	BatchedBits uint32
	SoloVal     float32
	BatchedVal  float32
	AbsDiff     float64
}

// BatchInvarianceReport summarizes a batch-invariance check across one or more
// [BatchInvarianceCase] evaluations.
type BatchInvarianceReport struct {
	TotalCases        int
	BitIdenticalCases int
	MaxAbsDiff        float64
	Mismatches        []BatchInvarianceMismatch
}

// Passed reports whether every checked case was bitwise identical.
func (r BatchInvarianceReport) Passed() bool {
	return len(r.Mismatches) == 0
}

// CheckBatchInvariance compares each case's SoloOutput against BatchedOutput
// bit-for-bit using [math.Float32bits]. Exact IEEE-754 bitwise identity is
// required; a single-ULP difference or a signed-zero flip (-0.0 vs +0.0) is
// recorded as a mismatch. For each mismatched case, the first differing element
// is recorded in Mismatches and MaxAbsDiff tracks the maximum absolute drift
// across all elements of all cases.
func CheckBatchInvariance(cases []BatchInvarianceCase) (BatchInvarianceReport, error) {
	if len(cases) == 0 {
		return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: no batch invariance cases")
	}

	report := BatchInvarianceReport{
		TotalCases: len(cases),
	}

	for i, c := range cases {
		if c.BatchSize <= 0 {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: case %d (%q): invalid batch size %d", i, c.SeqID, c.BatchSize)
		}
		if c.RowIndex < 0 || c.RowIndex >= c.BatchSize {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: case %d (%q): row index %d out of bounds for batch size %d",
				i, c.SeqID, c.RowIndex, c.BatchSize)
		}
		if len(c.SoloOutput) == 0 {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: case %d (%q): empty solo output", i, c.SeqID)
		}
		if len(c.SoloOutput) != len(c.BatchedOutput) {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: case %d (%q): solo output length %d != batched output length %d",
				i, c.SeqID, len(c.SoloOutput), len(c.BatchedOutput))
		}

		caseIdentical := true
		for elemIdx := range c.SoloOutput {
			soloVal := c.SoloOutput[elemIdx]
			batchedVal := c.BatchedOutput[elemIdx]
			soloBits := math.Float32bits(soloVal)
			batchedBits := math.Float32bits(batchedVal)

			diff := math.Abs(float64(soloVal) - float64(batchedVal))
			if math.IsNaN(diff) || diff > report.MaxAbsDiff {
				report.MaxAbsDiff = diff
			}

			if soloBits != batchedBits && caseIdentical {
				caseIdentical = false
				report.Mismatches = append(report.Mismatches, BatchInvarianceMismatch{
					SeqID:       c.SeqID,
					BatchSize:   c.BatchSize,
					RowIndex:    c.RowIndex,
					ElemIndex:   elemIdx,
					SoloBits:    soloBits,
					BatchedBits: batchedBits,
					SoloVal:     soloVal,
					BatchedVal:  batchedVal,
					AbsDiff:     diff,
				})
			}
		}
		if caseIdentical {
			report.BitIdenticalCases++
		}
	}

	return report, nil
}

// VerifyBatchInvariance evaluates each input solo (batch=1) via fn, then embeds
// each input at front (0), middle (B/2), and back (B-1) row positions and across
// forward and reversed filler permutations for every target batch size in
// batchSizes. It returns the resulting [BatchInvarianceReport] from
// [CheckBatchInvariance].
func VerifyBatchInvariance(
	inputs [][]float32,
	batchSizes []int,
	fn func(batch [][]float32) ([][]float32, error),
) (BatchInvarianceReport, error) {
	if len(inputs) == 0 {
		return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: empty inputs")
	}
	if len(batchSizes) == 0 {
		return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: empty batchSizes")
	}
	if fn == nil {
		return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: nil batch function")
	}
	for i, in := range inputs {
		if len(in) == 0 {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: input %d is empty", i)
		}
	}
	for _, bs := range batchSizes {
		if bs <= 0 {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: invalid batch size %d", bs)
		}
	}

	soloOutputs := make([][]float32, len(inputs))
	for i, in := range inputs {
		soloBatch := [][]float32{cloneFloat32Slice(in)}
		out, err := fn(soloBatch)
		if err != nil {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: solo run for seq-%d: %w", i, err)
		}
		if len(out) != 1 || len(out[0]) == 0 {
			return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: solo run for seq-%d returned %d rows", i, len(out))
		}
		soloOutputs[i] = cloneFloat32Slice(out[0])
	}

	var cases []BatchInvarianceCase
	for _, bs := range batchSizes {
		positions := placementPositions(bs)
		numPerms := 1
		if bs > 1 && len(inputs) > 1 {
			numPerms = 2
		}
		for seqIdx, in := range inputs {
			seqID := fmt.Sprintf("seq-%d", seqIdx)
			for _, rowIdx := range positions {
				for perm := 0; perm < numPerms; perm++ {
					batch := make([][]float32, bs)
					for r := 0; r < bs; r++ {
						if r == rowIdx {
							batch[r] = cloneFloat32Slice(in)
							continue
						}
						var fillerIdx int
						if perm == 0 {
							fillerIdx = (seqIdx + 1 + r) % len(inputs)
						} else {
							fillerIdx = (seqIdx + len(inputs) - 1 - (r % len(inputs))) % len(inputs)
						}
						batch[r] = cloneFloat32Slice(inputs[fillerIdx])
					}
					batchedOut, err := fn(batch)
					if err != nil {
						return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: batched run (batch=%d, row=%d) for %s: %w",
							bs, rowIdx, seqID, err)
					}
					if len(batchedOut) != bs {
						return BatchInvarianceReport{}, fmt.Errorf("cudaagreement: batched run (batch=%d) returned %d rows",
							bs, len(batchedOut))
					}
					cases = append(cases, BatchInvarianceCase{
						SeqID:         seqID,
						Input:         cloneFloat32Slice(in),
						SoloOutput:    soloOutputs[seqIdx],
						BatchedOutput: cloneFloat32Slice(batchedOut[rowIdx]),
						BatchSize:     bs,
						RowIndex:      rowIdx,
					})
				}
			}
		}
	}

	return CheckBatchInvariance(cases)
}

func placementPositions(bs int) []int {
	if bs <= 1 {
		return []int{0}
	}
	if bs == 2 {
		return []int{0, 1}
	}
	mid := bs / 2
	return []int{0, mid, bs - 1}
}

func cloneFloat32Slice(src []float32) []float32 {
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}

// FixedSplitKReductionCPU computes a deterministic fixed-tile tree reduction
// over row: it partitions row into tiles of size tileSize, pads each tile to
// the next power of two with +0.0, performs a pairwise binary tree reduction
// in float32 within each tile, and sums the resulting tile partials in strict
// index order 0..numTiles-1. Because the reduction tree depends only on
// tileSize and len(row)—never on the outer batch size—evaluating it across
// arbitrary batch sizes passes [VerifyBatchInvariance].
func FixedSplitKReductionCPU(row []float32, tileSize int) float32 {
	if len(row) == 0 {
		return 0
	}
	if tileSize <= 0 {
		tileSize = 16
	}
	paddedTile := 1
	for paddedTile < tileSize {
		paddedTile <<= 1
	}

	numTiles := (len(row) + tileSize - 1) / tileSize
	buf := make([]float32, paddedTile)
	var total float32

	for t := 0; t < numTiles; t++ {
		start := t * tileSize
		end := start + tileSize
		if end > len(row) {
			end = len(row)
		}
		for i := range buf {
			buf[i] = 0
		}
		copy(buf, row[start:end])

		for stride := paddedTile / 2; stride > 0; stride >>= 1 {
			for i := 0; i < stride; i++ {
				buf[i] = buf[i] + buf[i+stride]
			}
		}
		total = total + buf[0]
	}
	return total
}
