//go:build !darwin && !tpu

package main

// The gated KV paths, paged (LLAMA_INFER_PAGED_KV=1) and FP8
// (LLAMA_INFER_FP8_KV=1), replace the dense cache's per-layer launches with a
// step the engine binds before each forward call. These two dispatchers let
// the call sites stay ignorant of which path, if any, the engine was compiled
// with: on a dense engine they do nothing.

// bindKVStep stages the engine scratchpad for one token of skv at pos; skv nil
// means the engine's single-stream sequence.
func (eng *ModelEngine) bindKVStep(skv *SequenceKVCache, pos int) error {
	switch {
	case eng.paged != nil:
		return eng.bindPagedSingle(skv, pos)
	case eng.fp8 != nil:
		return eng.bindFP8Single(skv, pos)
	}
	return nil
}

// bindKVBatch stages sp for one row per cache at positions.
func (eng *ModelEngine) bindKVBatch(sp *GPUBatchScratchpad, caches []*SequenceKVCache, positions []int) error {
	switch {
	case eng.paged != nil:
		return eng.bindPagedBatch(sp, caches, positions)
	case eng.fp8 != nil:
		return eng.bindFP8Batch(sp, caches, positions)
	}
	return nil
}
