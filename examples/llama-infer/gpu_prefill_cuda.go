//go:build !darwin && !tpu

package main

import (
	"context"
	"errors"
	"fmt"
)

// Bound activation and matrix-plan storage independently of context length.
const cudaPrefillRows = 32

func (eng *ModelEngine) preparePrefillScratch() error {
	if eng.prefillScratch != nil {
		return nil
	}
	cfg := &eng.mCfg
	sp, err := newGPUBatchScratchpad(eng.dev, eng.memory, cfg, min(cudaPrefillRows, cfg.MaxSeqLen))
	if err != nil {
		return err
	}
	err = eng.dev.WithContext(func() error {
		for _, plan := range []struct {
			input, output int
			set           *[]cudaMatrixPlan
		}{
			{cfg.HiddenSize, cfg.NumAttentionHeads * cfg.HeadDim(), &sp.prefillQ},
			{cfg.HiddenSize, cfg.NumKeyValueHeads * cfg.HeadDim(), &sp.prefillKV},
			{cfg.NumAttentionHeads * cfg.HeadDim(), cfg.HiddenSize, &sp.prefillOut},
			{cfg.IntermediateSize, cfg.HiddenSize, &sp.prefillDown},
		} {
			*plan.set, err = sp.lmHead.addPlanSet(plan.input, plan.output, sp.capacity)
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		sp.Free()
		return fmt.Errorf("cuda prefill matrix plans: %w", err)
	}
	// Like the engine's decode scratch, this storage has engine lifetime.
	eng.prefillScratch = sp
	return nil
}

// gpuBatchedPrefillPassKVMeasured advances one causal history in prompt tiles.
// Each tile computes a layer for all rows before moving to the next layer.
// Within attention, rows are stored and attended in increasing position order,
// so a query can see its prefix and itself, but never a future prompt row.
func gpuBatchedPrefillPassKVMeasured(eng *ModelEngine, skv *SequenceKVCache, tokens []int32, startPos int) (int32, uint64, error) {
	if eng == nil || eng.dev == nil || eng.stream == nil || eng.weights == nil || eng.scratch == nil {
		return 0, 0, errors.New("cuda prefill: incomplete engine")
	}
	if startPos < 0 || startPos > eng.mCfg.MaxSeqLen || len(tokens) > eng.mCfg.MaxSeqLen-startPos {
		return 0, 0, errors.New("cuda prefill: prompt exceeds cache bounds")
	}
	for i, token := range tokens {
		if token < 0 || int(token) >= eng.mCfg.VocabSize {
			return 0, 0, fmt.Errorf("cuda prefill: invalid token at row %d", i)
		}
	}
	if len(tokens) == 0 {
		return 0, 0, nil
	}
	cache := skv
	if cache == nil {
		cache = &SequenceKVCache{Cache: eng.kvCache, paged: eng.pagedMain}
	}
	if !cache.usable() {
		return 0, 0, errors.New("cuda prefill: nil cache")
	}
	// A paged sequence has no fixed shape to check: its blocks are reserved
	// per tile when the step is bound.
	for layer := 0; cache.Cache != nil && layer < eng.mCfg.NumHiddenLayers; layer++ {
		state, err := cache.Cache.Layer(layer)
		if err != nil {
			return 0, 0, fmt.Errorf("cuda prefill cache: %w", err)
		}
		want := eng.mCfg.MaxSeqLen * eng.mCfg.NumKeyValueHeads * eng.mCfg.HeadDim()
		if state.K.Len() != want || state.V.Len() != want {
			return 0, 0, fmt.Errorf("cuda prefill: cache shape differs at layer %d", layer)
		}
	}
	if err := eng.preparePrefillScratch(); err != nil {
		return 0, 0, err
	}
	sp := eng.prefillScratch
	var caches [cudaPrefillRows]*SequenceKVCache
	var positions [cudaPrefillRows]int
	var next int32
	var dispatches uint64
	for base := 0; base < len(tokens); base += sp.capacity {
		n := min(sp.capacity, len(tokens)-base)
		for row := 0; row < n; row++ {
			caches[row] = cache
			positions[row] = startPos + base + row
		}
		if err := eng.bindKVBatch(sp, caches[:n], positions[:n]); err != nil {
			return 0, dispatches, fmt.Errorf("cuda prefill rows %d:%d: %w", base, base+n, err)
		}
		result, stats, err := gpuForwardBatchKV(context.Background(), eng, sp, caches[:n], tokens[base:base+n], positions[:n], true)
		dispatches += stats.Dispatches
		if err != nil {
			return 0, dispatches, fmt.Errorf("cuda prefill rows %d:%d: %w", base, base+n, err)
		}
		next = result[0]
	}
	return next, dispatches, nil
}
