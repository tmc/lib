package cudaspec

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamtp"
)

// BatchCycleRequest describes one bounded deterministic MTP speculation cycle.
// Key and ExpectedKey bind the admitted model, prefix, branch, position,
// numerical mode, and process/model generation. These are caller declarations,
// not independently measured model or device identity.
type BatchCycleRequest struct {
	// DraftDigest pins exact draft projection and head weights, including shape.
	DraftDigest                       [32]byte
	Key, ExpectedKey                  SpeculationCacheKey
	Prefix                            []int32
	PrefixSequence, Sequence          int64
	Hidden, DraftWeights, HeadWeights []float32
	MTP                               cudamtp.MTPConfig
}

// BatchVerifier verifies target tokens through the existing batch row contract.
// Wait must establish that all work for the step is quiescent, including when
// ExecuteForward returns cancellation or failure. Wait receives no canceled
// context: cancellation alone cannot authorize releasing borrowed device KV.
// It must have its own bounded native drain policy. A failed Wait without independent Done proof retains KV
// in the returned result for explicit owner recovery, rather than guessing.
type BatchVerifier struct {
	// ReadOnlyPrefix requires verification that never writes borrowed prefix KV.
	// Writes to private branch blocks are permitted. Engines that overwrite
	// prefix blocks are unsupported without a copy-on-write adapter.
	ReadOnlyPrefix bool
	Engine         cudabatch.ExecutionEngine
	Wait           func(*cudabatch.ExecutionStep) error
	// Done optionally proves exact-step quiescence independently of a terminal
	// execution error from Wait. A false result retains the KV lease.
	Done func(*cudabatch.ExecutionStep) bool
}

// BatchCycleResult contains workload results and the branch's KV lease.
// A fully accepted branch stays live until the caller closes KV after use.
// Rejected branches are closed. Without independent Done proof, a drain error
// leaves KV live even though
// the cycle failed; the owner must establish quiescence before closing it.
type BatchCycleResult struct {
	Outcome  CycleOutcome
	BranchID BranchID
	KV       *BranchKV
}

// ExecuteBatchCycle computes MTP reference drafts and verifies them through
// validated decode rows. It is a synchronous, single-branch integration slice;
// it does not claim native MTP execution, sampled verification, or asynchronous
// separate-device speculation. Allocator and inputs must not change during it.
func ExecuteBatchCycle(ctx context.Context, a *cudakv.BlockAllocator, req BatchCycleRequest, v BatchVerifier) (result *BatchCycleResult, retErr error) {
	if ctx == nil || a == nil || v.Engine == nil || v.Wait == nil || !v.ReadOnlyPrefix {
		return nil, errors.New("cudaspec: missing batch cycle dependency")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := req.MTP
	if c.NumDraftHeads < 1 || c.NumDraftHeads > 16 || c.Dim < 1 || c.Dim > 4096 || c.VocabSize < 1 || c.VocabSize > 65536 || len(req.Prefix) < 1 || len(req.Prefix) > 4096 || req.Key.ModelDigest == ([32]byte{}) || req.Key.NumericalMode != "fp32" || req.Key.Generation == 0 || req.Key.BranchID != 1 {
		return nil, errors.New("cudaspec: invalid batch cycle bounds or identity")
	}
	if c.NumDraftHeads*c.Dim*c.Dim > 16<<20 || c.NumDraftHeads*c.Dim*c.VocabSize > 16<<20 {
		return nil, errors.New("cudaspec: MTP weight budget exceeded")
	}
	expected := ComputeSpeculationKey(req.Key.ModelDigest[:], req.Prefix, 1, len(req.Prefix), c.NumDraftHeads, req.Key.NumericalMode, req.Key.Generation)
	if req.Key != expected || req.ExpectedKey != expected {
		return nil, ErrCacheKeyMismatch
	}
	for _, xs := range [][]float32{req.Hidden, req.DraftWeights, req.HeadWeights} {
		for _, x := range xs {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, errors.New("cudaspec: nonfinite MTP input")
			}
		}
	}
	for _, tok := range req.Prefix {
		if tok < 0 || int(tok) >= c.VocabSize {
			return nil, errors.New("cudaspec: prefix token outside vocabulary")
		}
	}
	digest, err := MTPDraftDigest(c, req.DraftWeights, req.HeadWeights)
	if err != nil {
		return nil, err
	}
	if req.DraftDigest != digest {
		return nil, errors.New("cudaspec: draft weight digest mismatch")
	}
	logits, err := cudamtp.MTPReference(req.Hidden, req.DraftWeights, req.HeadWeights, c)
	if err != nil {
		return nil, err
	}
	drafts := make([]int32, c.NumDraftHeads)
	for h := range drafts {
		row := logits[h*c.VocabSize : (h+1)*c.VocabSize]
		best := 0
		for j, x := range row {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, errors.New("cudaspec: nonfinite MTP output")
			}
			if x > row[best] {
				best = j
			}
		}
		drafts[h] = int32(best)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	kv, err := NewBranchKV(a, req.PrefixSequence, req.Sequence, len(req.Prefix), len(drafts))
	if err != nil {
		return nil, err
	}
	result = &BatchCycleResult{KV: kv}
	release := true
	defer func() {
		if release {
			retErr = errors.Join(retErr, kv.Close())
		}
	}()
	bt := NewBranchTree(req.PrefixSequence, a.Table(req.PrefixSequence), req.Prefix[len(req.Prefix)-1])
	branch, err := bt.ProposeBranch(0, drafts, kv.private)
	if err != nil {
		return result, err
	}
	result.BranchID = branch.ID
	contextTokens := append([]int32(nil), req.Prefix...)
	predictions := make([]int32, 0, len(drafts)+1)
	rowID := fmt.Sprintf("spec-%d-%d", req.Sequence, req.Key.Generation)
	for i := 0; i <= len(drafts); i++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		table, err := kv.Table()
		if err != nil {
			return result, err
		}
		row := cudabatch.BatchRowBinding{RowIndex: 0, SeqID: rowID, Tokens: []int32{contextTokens[len(contextTokens)-1]}, StartPos: len(contextTokens) - 1, SequenceLength: len(contextTokens), BlockTable: table}
		metadata, err := cudabatch.ValidateAndBindRows([]cudabatch.BatchRowBinding{row}, cudabatch.ShapeBucket{BatchSize: 1, MaxSeqLen: len(req.Prefix) + len(drafts), MaxTokens: 1}, a.BlockSize())
		if err != nil {
			return result, err
		}
		step := &cudabatch.ExecutionStep{StepID: int64(i + 1), Metadata: metadata}
		tokens, rowErrors, runErr := v.Engine.ExecuteForward(ctx, step)
		if err := v.Wait(step); err != nil {
			release = v.Done != nil && v.Done(step)
			return result, fmt.Errorf("cudaspec: verification drain: %w", err)
		}
		if v.Done != nil && !v.Done(step) {
			release = false
			return result, errors.New("cudaspec: verification completion not established")
		}
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if runErr != nil {
			return result, runErr
		}
		if err := rowErrors[rowID]; err != nil {
			return result, err
		}
		tok, ok := tokens[rowID]
		if !ok || len(tokens) != 1 || len(rowErrors) != 0 || tok < 0 || int(tok) >= c.VocabSize {
			return result, errors.New("cudaspec: incomplete or foreign verification row")
		}
		predictions = append(predictions, tok)
		if i < len(drafts) && tok != drafts[i] {
			break
		}
		if i < len(drafts) {
			contextTokens = append(contextTokens, drafts[i])
		}
	}
	verified := VerifyGreedy(predictions, drafts)
	result.Outcome = CycleOutcome{DraftTokens: append([]int32(nil), drafts...), AcceptedTokens: verified.AcceptedTokens, BonusToken: verified.BonusToken, TotalEmitted: verified.TotalEmitted, DiscardedCount: len(drafts) - verified.AcceptedCount, AcceptedRatio: float64(verified.AcceptedCount) / float64(len(drafts))}
	if verified.AcceptedCount == len(drafts) {
		if err := bt.CommitBranch(branch.ID, verified.AcceptedTokens, kv); err != nil {
			return result, err
		}
		result.Outcome.Committed = true
		release = false
	} else {
		if err := bt.RollbackBranch(branch.ID, kv); err != nil {
			return result, err
		}
	}
	return result, nil
}

// MTPDraftDigest hashes bounded draft weights with their shape and float32 bits.
// It does not assert that these weights belong to a caller's target model.
func MTPDraftDigest(c cudamtp.MTPConfig, projection, heads []float32) ([32]byte, error) {
	if c.NumDraftHeads < 1 || c.NumDraftHeads > 16 || c.Dim < 1 || c.Dim > 4096 || c.VocabSize < 1 || c.VocabSize > 65536 || c.NumDraftHeads*c.Dim*c.Dim > 16<<20 || c.NumDraftHeads*c.Dim*c.VocabSize > 16<<20 {
		return [32]byte{}, errors.New("cudaspec: invalid draft weight bounds")
	}
	if len(projection) != c.NumDraftHeads*c.Dim*c.Dim || len(heads) != c.NumDraftHeads*c.Dim*c.VocabSize {
		return [32]byte{}, errors.New("cudaspec: draft weight shape mismatch")
	}
	h := sha256.New()
	var b [8]byte
	for _, n := range []int{c.NumDraftHeads, c.Dim, c.VocabSize} {
		binary.LittleEndian.PutUint64(b[:], uint64(n))
		h.Write(b[:])
	}
	for _, xs := range [][]float32{projection, heads} {
		for _, x := range xs {
			binary.LittleEndian.PutUint32(b[:4], math.Float32bits(x))
			h.Write(b[:4])
		}
	}
	var digest [32]byte
	copy(digest[:], h.Sum(nil))
	return digest, nil
}
