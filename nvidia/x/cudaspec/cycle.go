// SPDX-License-Identifier: Apache-2.0

package cudaspec

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudaattn"
)

// CycleConfig configures an end-to-end speculative decoding cycle.
type CycleConfig struct {
	NumHeads      int
	HeadDim       int
	Scale         float32
	VocabSize     int
	NumDraftHeads int
	ModelDigest   [32]byte
	NumericalMode string
}

// CycleOutcome reports acceptance statistics and emitted tokens from a speculative cycle.
type CycleOutcome struct {
	DraftTokens    []int32
	AcceptedTokens []int32
	BonusToken     int32
	TotalEmitted   int
	DiscardedCount int
	AcceptedRatio  float64
	Committed      bool
}

// ExecuteSpeculativeCycle runs a deterministic propose, attention-reference-verify,
// and commit-or-rollback cycle using explicit branch state and KV release tracking.
func ExecuteSpeculativeCycle(
	cfg CycleConfig,
	bt *BranchTree,
	parentBranchID BranchID,
	draftTokens []int32,
	allocatedKVBlocks []int,
	releaser KVBlockReleaser,
	targetOracle func(contextTokens []int32) int32,
) (*CycleOutcome, error) {
	if len(draftTokens) == 0 {
		return nil, fmt.Errorf("cudaspec: cycle requires non-empty draft tokens")
	}

	// 1. Propose candidate branch
	branch, err := bt.ProposeBranch(parentBranchID, draftTokens, allocatedKVBlocks)
	if err != nil {
		return nil, fmt.Errorf("cudaspec: propose branch: %w", err)
	}

	// 2. Query target oracle for target predictions at each draft position + bonus position
	// Build full prefix path up to parent branch
	prefix, err := bt.PathTokens(parentBranchID)
	if err != nil {
		return nil, err
	}

	targetPredictions := make([]int32, len(draftTokens)+1)
	runningContext := append([]int32(nil), prefix...)

	for i := 0; i < len(draftTokens); i++ {
		pred := targetOracle(runningContext)
		targetPredictions[i] = pred
		runningContext = append(runningContext, pred)
	}
	// Bonus position prediction
	targetPredictions[len(draftTokens)] = targetOracle(runningContext)

	// 3. Verify greedy
	verifyRes := VerifyGreedy(targetPredictions, draftTokens)

	// 4. Decision: commit or rollback
	discarded := len(draftTokens) - verifyRes.AcceptedCount
	var committed bool

	if verifyRes.AcceptedCount == len(draftTokens) {
		// All draft tokens accepted! Commit branch
		if err := bt.CommitBranch(branch.ID, verifyRes.AcceptedTokens, releaser); err != nil {
			return nil, fmt.Errorf("cudaspec: commit branch: %w", err)
		}
		committed = true
	} else {
		// Partial or complete reject: roll back branch and release KV
		if err := bt.RollbackBranch(branch.ID, releaser); err != nil {
			return nil, fmt.Errorf("cudaspec: rollback branch: %w", err)
		}
		committed = false
	}

	ratio := float64(0.0)
	if len(draftTokens) > 0 {
		ratio = float64(verifyRes.AcceptedCount) / float64(len(draftTokens))
	}

	return &CycleOutcome{
		DraftTokens:    draftTokens,
		AcceptedTokens: verifyRes.AcceptedTokens,
		BonusToken:     verifyRes.BonusToken,
		TotalEmitted:   verifyRes.TotalEmitted,
		DiscardedCount: discarded,
		AcceptedRatio:  ratio,
		Committed:      committed,
	}, nil
}

// TreeMaskToBitmasks converts a 2D boolean tree attention mask [N, N] to uint64 row bitmasks.
func TreeMaskToBitmasks(mask [][]bool) ([]uint64, error) {
	n := len(mask)
	if n > cudaattn.TreeAttentionMaxNodes {
		return nil, fmt.Errorf("cudaspec: tree size %d exceeds max nodes %d", n, cudaattn.TreeAttentionMaxNodes)
	}
	bitmasks := make([]uint64, n)
	for i := 0; i < n; i++ {
		var row uint64
		if len(mask[i]) < n {
			return nil, fmt.Errorf("cudaspec: mask row %d length %d < %d", i, len(mask[i]), n)
		}
		for j := 0; j < n; j++ {
			if mask[i][j] {
				row |= (1 << j)
			}
		}
		bitmasks[i] = row
	}
	return bitmasks, nil
}

// VerifyTreeAttentionOutputs runs dense reference attention and compares against sparse tree reference attention.
func VerifyTreeAttentionOutputs(q, k, v []float32, mask [][]bool, numNodes, numHeads, headDim int, scale float32) error {
	bitmasks, err := TreeMaskToBitmasks(mask)
	if err != nil {
		return err
	}
	sparseOut, err := cudaattn.TreeAttentionReference(q, k, v, bitmasks, numNodes, numHeads, headDim, scale)
	if err != nil {
		return fmt.Errorf("cudaspec: sparse tree attention: %w", err)
	}

	// Compare with dense reference row by row
	for nodeIdx := 0; nodeIdx < numNodes; nodeIdx++ {
		for headIdx := 0; headIdx < numHeads; headIdx++ {
			// Extract single row
			qOff := (nodeIdx*numHeads + headIdx) * headDim
			qVec := q[qOff : qOff+headDim]

			// Compute scores
			scores := make([]float32, numNodes)
			valMatrix := make([]float32, numNodes*headDim)
			for j := 0; j < numNodes; j++ {
				kOff := (j*numHeads + headIdx) * headDim
				vOff := (j*numHeads + headIdx) * headDim
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qVec[d] * k[kOff+d]
					valMatrix[j*headDim+d] = v[vOff+d]
				}
				scores[j] = dot * scale
			}

			// Invert mask boolean for egraph: false = unmasked, true = masked
			egraphMask := make([]bool, numNodes)
			for j := 0; j < numNodes; j++ {
				egraphMask[j] = !mask[nodeIdx][j]
			}

			// Dense softmax reference
			allMasked := true
			for j := 0; j < numNodes; j++ {
				if !egraphMask[j] {
					allMasked = false
					break
				}
			}

			denseOut := make([]float32, headDim)
			if !allMasked {
				var maxSc float64 = -math.MaxFloat64
				for j := 0; j < numNodes; j++ {
					if !egraphMask[j] && float64(scores[j]) > maxSc {
						maxSc = float64(scores[j])
					}
				}
				var sumExp float64
				p := make([]float64, numNodes)
				for j := 0; j < numNodes; j++ {
					if !egraphMask[j] {
						e := math.Exp(float64(scores[j]) - maxSc)
						p[j] = e
						sumExp += e
					}
				}
				if sumExp > 0 {
					for d := 0; d < headDim; d++ {
						var acc float64
						for j := 0; j < numNodes; j++ {
							if !egraphMask[j] {
								acc += p[j] * float64(valMatrix[j*headDim+d])
							}
						}
						denseOut[d] = float32(acc / sumExp)
					}
				}
			}

			// Check match with sparseOut
			outOff := (nodeIdx*numHeads + headIdx) * headDim
			for d := 0; d < headDim; d++ {
				diff := math.Abs(float64(sparseOut[outOff+d] - denseOut[d]))
				if diff > 1e-4 {
					return fmt.Errorf("cudaspec: output mismatch at node %d head %d dim %d: got %f, want %f (diff %f)",
						nodeIdx, headIdx, d, sparseOut[outOff+d], denseOut[d], diff)
				}
			}
		}
	}
	return nil
}
