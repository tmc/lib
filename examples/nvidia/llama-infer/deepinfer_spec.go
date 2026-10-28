package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

// newSpecVarlenScheduler constructs a cudamtp.VarlenScheduler for a single
// sequence or micro-batch with up to maxDraft speculative draft tokens per step.
func newSpecVarlenScheduler(maxDraft int) (*cudamtp.VarlenScheduler, error) {
	if maxDraft <= 0 {
		return nil, fmt.Errorf("spec varlen scheduler: maxDraft %d must be > 0", maxDraft)
	}
	buckets := make([]int, maxDraft+1)
	for i := 0; i <= maxDraft; i++ {
		buckets[i] = i + 1
	}
	return cudamtp.NewVarlenScheduler(cudamtp.VarlenScheduleConfig{
		NumSpecSteps: maxDraft,
		Buckets:      buckets,
	})
}

// scheduleDraftBudget queries sched for the optimal speculative draft count in
// [0, maxDraft] for a sequence with seqID and current KV context length contextLen.
func scheduleDraftBudget(sched *cudamtp.VarlenScheduler, seqID int64, contextLen, maxDraft int) (int, error) {
	if sched == nil {
		return 0, fmt.Errorf("schedule draft budget: nil VarlenScheduler")
	}
	if maxDraft <= 0 {
		return 0, nil
	}
	ewma, ok := sched.EWMAProb(seqID)
	if !ok || ewma <= 0 {
		// Before any acceptance observation is recorded, default to full draft
		// exploration so the first speculative step can calibrate acceptance.
		ewma = 0.8
	}
	dec, err := sched.Schedule([]cudamtp.SequenceMTPState{
		{
			SeqID:      seqID,
			ContextLen: contextLen,
			EWMAProb:   ewma,
		},
	})
	if err != nil {
		return 0, err
	}
	if len(dec.SeqlensQ) == 0 {
		return 0, nil
	}
	draftBudget := dec.SeqlensQ[0] - 1
	if draftBudget < 0 {
		draftBudget = 0
	}
	if draftBudget > maxDraft {
		draftBudget = maxDraft
	}
	return draftBudget, nil
}

// planZOSDecodeTransition computes the Zero-Overhead Speculation (ZOS) token
// compaction and position schedule between speculative step N-1 and step N
// using cudabatch.BuildZOSRemapPlan and cudabatch.ApplyZOSRemapCPU.
//
// prevTokens holds the tokens fed in step N-1 ([next, draft_0, ..., draft_{k-1}]),
// acceptedCount is the number of accepted tokens in [1, len(prevTokens)],
// nextDrafts holds the extra draft tokens scheduled for step N, and pos is the
// starting KV position of step N-1. It returns the ZOSRemapPlan, the compacted
// step N input token IDs, and the updated KV sequence length.
func planZOSDecodeTransition(
	prevTokens []int32,
	acceptedCount int,
	nextDrafts []int32,
	pos int,
) (plan cudabatch.ZOSRemapPlan, nextInputIDs []int32, nextSeqlenK int, err error) {
	if len(prevTokens) == 0 {
		return cudabatch.ZOSRemapPlan{}, nil, 0, fmt.Errorf("zos transition: empty prevTokens")
	}
	if acceptedCount <= 0 || acceptedCount > len(prevTokens) {
		return cudabatch.ZOSRemapPlan{}, nil, 0, fmt.Errorf("zos transition: acceptedCount %d outside [1, %d]", acceptedCount, len(prevTokens))
	}
	if pos < 0 {
		return cudabatch.ZOSRemapPlan{}, nil, 0, fmt.Errorf("zos transition: negative position %d", pos)
	}
	prevCuSeqlensQ := []int32{0, int32(len(prevTokens))}
	acceptedCounts := []int32{int32(acceptedCount)}
	nextSeqlensQ := []int32{int32(1 + len(nextDrafts))}
	basePositions := []int32{int32(pos)}

	plan, err = cudabatch.BuildZOSRemapPlan(prevCuSeqlensQ, acceptedCounts, nextSeqlensQ, basePositions)
	if err != nil {
		return cudabatch.ZOSRemapPlan{}, nil, 0, err
	}
	nextInputIDs, nextSeqlensK, err := cudabatch.ApplyZOSRemapCPU(
		prevTokens,
		nextDrafts,
		[]int32{int32(pos)},
		acceptedCounts,
		plan,
	)
	if err != nil {
		return cudabatch.ZOSRemapPlan{}, nil, 0, err
	}
	return plan, nextInputIDs, int(nextSeqlensK[0]), nil
}

// rankDraftsByConfidence computes per-sequence query lengths seqlens_q and
// prefix sums cu_seqlens_q from per-step confidence probabilities using
// cudaspec.ComputeSeqlensQByConfidenceCPU.
func rankDraftsByConfidence(prob [][]float32, remap []int32, targetTotalBsz, maxStep int) (seqlensQ []int32, cuSeqlensQ []int32, err error) {
	return cudaspec.ComputeSeqlensQByConfidenceCPU(prob, remap, targetTotalBsz, maxStep)
}
