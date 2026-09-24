// SPDX-License-Identifier: Apache-2.0

package cudaattn

import (
	"errors"
	"fmt"
	"math"
)

// ErrTreeAttentionShape indicates invalid dimensions for tree attention.
var ErrTreeAttentionShape = errors.New("cudaattn: invalid tree attention shape")

// TreeAttentionReference evaluates tree attention on the CPU in float64 using
// bitmasks, where bit j in bitmasks[i] indicates whether node i attends to node j.
// Completely masked rows output exact zero vectors without NaN/Inf.
//
// Shapes:
// q: [numNodes, numHeads, headDim]
// k: [numNodes, numHeads, headDim]
// v: [numNodes, numHeads, headDim]
// bitmasks: [numNodes]uint64
//
// Returns out: [numNodes, numHeads, headDim]
func TreeAttentionReference(q, k, v []float32, bitmasks []uint64, numNodes, numHeads, headDim int, scale float32) ([]float32, error) {
	if numNodes < 0 || numHeads <= 0 || headDim <= 0 || scale <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return nil, ErrTreeAttentionShape
	}
	expectedLen := numNodes * numHeads * headDim
	if len(q) != expectedLen || len(k) != expectedLen || len(v) != expectedLen {
		return nil, fmt.Errorf("%w: tensor length mismatch (got q=%d k=%d v=%d, want %d)", ErrTreeAttentionShape, len(q), len(k), len(v), expectedLen)
	}
	if len(bitmasks) < numNodes {
		return nil, fmt.Errorf("%w: bitmasks length %d < numNodes %d", ErrTreeAttentionShape, len(bitmasks), numNodes)
	}
	if numNodes == 0 {
		return []float32{}, nil
	}

	out := make([]float32, expectedLen)
	nLimit := numNodes
	if nLimit > TreeAttentionMaxNodes {
		nLimit = TreeAttentionMaxNodes
	}

	for nodeIdx := 0; nodeIdx < numNodes; nodeIdx++ {
		mask := bitmasks[nodeIdx]
		for headIdx := 0; headIdx < numHeads; headIdx++ {
			qOff := (nodeIdx*numHeads + headIdx) * headDim
			qVec := q[qOff : qOff+headDim]

			// Step 1: Compute scores for attended nodes
			scores := make([]float64, nLimit)
			hasValid := false
			maxScore := -math.MaxFloat64

			for t := 0; t < nLimit; t++ {
				if (mask>>t)&1 == 1 {
					kOff := (t*numHeads + headIdx) * headDim
					kVec := k[kOff : kOff+headDim]
					var dot float64
					for d := 0; d < headDim; d++ {
						dot += float64(qVec[d]) * float64(kVec[d])
					}
					dot *= float64(scale)
					scores[t] = dot
					if !hasValid || dot > maxScore {
						maxScore = dot
						hasValid = true
					}
				} else {
					scores[t] = -math.MaxFloat64
				}
			}

			outOff := (nodeIdx*numHeads + headIdx) * headDim
			if !hasValid {
				// Fully masked row: policy is exact zeros
				for d := 0; d < headDim; d++ {
					out[outOff+d] = 0.0
				}
				continue
			}

			// Step 2: Softmax normalizer
			var sumExp float64
			p := make([]float64, nLimit)
			for t := 0; t < nLimit; t++ {
				if (mask>>t)&1 == 1 {
					expVal := math.Exp(scores[t] - maxScore)
					p[t] = expVal
					sumExp += expVal
				}
			}

			if sumExp == 0 {
				for d := 0; d < headDim; d++ {
					out[outOff+d] = 0.0
				}
				continue
			}

			// Step 3: Weighted sum over V
			for d := 0; d < headDim; d++ {
				var acc float64
				for t := 0; t < nLimit; t++ {
					if (mask>>t)&1 == 1 {
						vOff := (t*numHeads + headIdx) * headDim
						acc += p[t] * float64(v[vOff+d])
					}
				}
				out[outOff+d] = float32(acc / sumExp)
			}
		}
	}

	return out, nil
}

// TreeDenseReferenceMask computes the full dense boolean mask matrix [numNodes, numNodes]
// from bitmasks.
func TreeDenseReferenceMask(bitmasks []uint64, numNodes int) [][]bool {
	nLimit := numNodes
	if nLimit > TreeAttentionMaxNodes {
		nLimit = TreeAttentionMaxNodes
	}
	dense := make([][]bool, numNodes)
	for i := 0; i < numNodes; i++ {
		dense[i] = make([]bool, numNodes)
		if i < len(bitmasks) {
			mask := bitmasks[i]
			for j := 0; j < nLimit; j++ {
				if (mask>>j)&1 == 1 {
					dense[i][j] = true
				}
			}
		}
	}
	return dense
}
