package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudagrammar"
	"github.com/tmc/lib/nvidia/x/cudasample"
)

// sampleConfig configures deterministic Philox-4x32-10 token sampling for a
// decode step.
type sampleConfig struct {
	seed           uint64
	temp           float32
	topK           int
	topP           float32
	minP           float32
	allowedBitmask []uint32
}

// sampleStepLogits samples one token ID from logits at decode step using
// batch-invariant Philox-4x32-10 seeded sampling (cudasample.SampleSeededCPU).
// When cfg.temp <= 0 and cfg.allowedBitmask is nil, it returns the greedy
// argmax token ID.
func sampleStepLogits(logits []float32, step int, cfg sampleConfig) (int32, error) {
	if step < 0 {
		return -1, fmt.Errorf("sample step %d must be non-negative", step)
	}
	if len(logits) == 0 {
		return -1, fmt.Errorf("sample step %d: empty logits", step)
	}
	params := cudasample.SequenceSampleParams{
		SeqSeed:        cfg.seed,
		StepOffset:     uint64(step),
		Temperature:    cfg.temp,
		TopK:           cfg.topK,
		TopP:           cfg.topP,
		MinP:           cfg.minP,
		AllowedBitmask: cfg.allowedBitmask,
	}
	tok, _, err := cudasample.SampleSeededCPU(logits, params)
	if err != nil {
		return -1, fmt.Errorf("sample step %d: %w", step, err)
	}
	return tok, nil
}

// speculativeGrammarMasks computes the per-step vocabulary bitmasks for a
// speculative draft chain using matcher.SpeculativeMasks, rolling back the
// matcher to its initial state before returning.
func speculativeGrammarMasks(matcher *cudagrammar.Matcher, vocab *cudagrammar.Vocabulary, drafts []int32) ([]*cudagrammar.Bitmask, error) {
	if matcher == nil {
		return nil, fmt.Errorf("speculative grammar masks: nil matcher")
	}
	if vocab == nil {
		return nil, fmt.Errorf("speculative grammar masks: nil vocabulary")
	}
	return matcher.SpeculativeMasks(vocab, drafts)
}

// verifySeededSpeculativeStep verifies draftTokens against targetProbs and
// draftProbs using deterministic Philox-4x32-10 rejection sampling
// (cudasample.VerifySpeculativeSeededCPU). When matcher and vocab are non-nil,
// it commits the accepted prefix and bonus token to matcher via CommitPrefix.
func verifySeededSpeculativeStep(
	targetProbs [][]float32,
	draftProbs [][]float32,
	draftTokens []int32,
	seqSeed uint64,
	baseStepOffset uint64,
	matcher *cudagrammar.Matcher,
	vocab *cudagrammar.Vocabulary,
) (accepted []int32, bonus int32, err error) {
	accepted, bonus, err = cudasample.VerifySpeculativeSeededCPU(targetProbs, draftProbs, draftTokens, seqSeed, baseStepOffset)
	if err != nil {
		return nil, -1, err
	}
	if matcher != nil && vocab != nil {
		commitTokens := append(append([]int32(nil), accepted...), bonus)
		if err := matcher.CommitPrefix(vocab, commitTokens); err != nil {
			return nil, -1, fmt.Errorf("commit grammar prefix: %w", err)
		}
	}
	return accepted, bonus, nil
}
