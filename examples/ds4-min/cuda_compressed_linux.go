//go:build linux

package main

import (
	_ "embed"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaCompressedState struct {
	cache, stateKV, stateScore, pooled cudamem.Slice[float32]
	headDim, width, ratio, cap, n      int
}

// cudaScoreTileShared is the dynamic shared memory ds4_attention_mixed_scores_tile
// asks for: DS4_SCORE_TILE_ROWS key rows of DS4_SCORE_TILE_STRIDE floats,
// padded to keep a warp off one shared-memory bank, plus DS4_SCORE_TILE_HEADS
// query rows of DS4_SCORE_TILE_QSTRIDE binary16 values, which need no padding
// because every thread of a row reads the same offset and broadcasts. It must
// track both strides in cuda_compressed.cu.
const cudaScoreTileShared = 16*513*4 + 16*512*2

// cudaSelectBlock is the thread count of one ds4_indexer_select block. It
// also sets the tile width of that kernel's fallback threshold-and-sort path,
// which is correct for any tile: the threshold never exceeds the 512th largest
// key seen so far, so pruning against it cannot drop a true top-512 element.
const cudaSelectBlock = 128

// cudaIndexRowsPerBlock is the number of compressed cache rows one
// ds4_indexer_score block scores, one warp each. DS4_INDEX_ROWS_PER_BLOCK in
// cuda_compressed.cu must hold the same value.
const cudaIndexRowsPerBlock = 8

// cudaFinalizeHeadsPerBlock is the number of attention heads one
// ds4_attention_mixed_finalize block finishes. DS4_FINALIZE_HEADS_PER_BLOCK in
// cuda_compressed.cu must hold the same value. It divides the grid and
// multiplies the dynamic shared-memory request, because the kernel now holds
// one score plane per head it owns.
const cudaFinalizeHeadsPerBlock = 2

// cudaFinalizeThreads is the block ds4_attention_mixed_finalize is launched
// with: the 512 threads that each own one output element, plus a warp that
// owns none and computes the per-head denominators while the others stream the
// latent cache. The denominator is a serial sum in a pinned order and has
// nothing to overlap with inside its own thread. The kernel reads blockDim.x
// and is correct at a bare 512 too, where the last threads do both in turn.
const cudaFinalizeThreads = 512 + 32

// encodeCompressor appends an update and, at a ratio boundary, a finished row.
// The caller advances state.n after the launches succeed. Later work on the
// same stream observes the update before using the new row count.
func (r *cudaFoundation) encodeCompressor(kv, gate cudamem.Slice[float32], ape, norm cudaModelTensor, state *cudaCompressedState, position int, indexer bool) (bool, error) {
	head := 512
	if indexer {
		head = 128
	}
	if state == nil || state.headDim != head || (state.ratio != 4 && state.ratio != 128) || indexer && state.ratio != 4 || position < 0 {
		return false, fmt.Errorf("cuda: compressor: invalid shape")
	}
	width, rows := head, state.ratio
	if state.ratio == 4 {
		width *= 2
		rows *= 2
	}
	if state.width != width || kv.Len() != width || gate.Len() != width || state.stateKV.Len() != rows*width || state.stateScore.Len() != rows*width || state.pooled.Len() != head || state.cap < 1 || state.cache.Len() != state.cap*head || state.n < 0 || state.n > state.cap || ape.typeID != 1 || len(ape.dimensions) != 2 || ape.dimensions[0] != uint64(width) || ape.dimensions[1] != uint64(state.ratio) || norm.typeID != 0 || len(norm.dimensions) != 1 || norm.dimensions[0] != uint64(head) {
		return false, fmt.Errorf("cuda: compressor: invalid tensor profile")
	}
	emit := (position+1)%state.ratio == 0
	if emit && state.n == state.cap {
		return false, fmt.Errorf("cuda: compressor: cache capacity exceeded")
	}
	err := r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.compressorPoolFn).Grid(1, 1, 1).Block(uint32(head), 1, 1).On(r.stream()).Launch(kv, gate, state.stateKV, state.stateScore, state.pooled, ape.device, uint32(head), uint32(width), uint32(state.ratio), uint32(position)); err != nil {
			return fmt.Errorf("cuda: launch compressor pool: %w", err)
		}
		if !emit {
			return nil
		}
		fn := r.compressorAttentionFn
		if indexer {
			fn = r.compressorIndexerFn
		}
		if err := cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(uint32(head), 1, 1).On(r.stream()).Launch(state.pooled, norm.device, state.cache, uint32(state.n), uint32(position+1-state.ratio)); err != nil {
			return fmt.Errorf("cuda: launch compressor finish: %w", err)
		}
		return nil
	})
	return emit, err
}

// encodeCompressorBatch pools a whole group of tokens at once. kv and gate are
// token major at the state's own width, the layout the batched pair projection
// writes, and the group must open at a multiple of the ratio so that a token's
// index within the group is its index within its window. It returns the number
// of rows appended to the cache; the caller advances state.n after it returns.
//
// The windows are independent -- see ds4_compressor_window -- so this is three
// dispatches where the token at a time encoder was two per token.
func (r *cudaFoundation) encodeCompressorBatch(kv, gate cudamem.Slice[float32], ape, norm cudaModelTensor, state *cudaCompressedState, pooled cudamem.Slice[float32], position, n int, indexer bool) (int, error) {
	head := 512
	if indexer {
		head = 128
	}
	if state == nil || state.headDim != head || (state.ratio != 4 && state.ratio != 128) || indexer && state.ratio != 4 || position < 0 || n < 1 || position%state.ratio != 0 {
		return 0, fmt.Errorf("cuda: compressor batch: invalid shape")
	}
	width, rows := head, state.ratio
	if state.ratio == 4 {
		width *= 2
		rows *= 2
	}
	if state.width != width || kv.Len() != n*width || gate.Len() != n*width || state.stateKV.Len() != rows*width || state.stateScore.Len() != rows*width || state.cap < 1 || state.cache.Len() != state.cap*head || state.n < 0 || state.n > state.cap || ape.typeID != 1 || len(ape.dimensions) != 2 || ape.dimensions[0] != uint64(width) || ape.dimensions[1] != uint64(state.ratio) || norm.typeID != 0 || len(norm.dimensions) != 1 || norm.dimensions[0] != uint64(head) {
		return 0, fmt.Errorf("cuda: compressor batch: invalid tensor profile")
	}
	emits := n / state.ratio
	if pooled.Len() < emits*head {
		return 0, fmt.Errorf("cuda: compressor batch: invalid pool capacity")
	}
	if state.n+emits > state.cap {
		return 0, fmt.Errorf("cuda: compressor: cache capacity exceeded")
	}
	finish := r.compressorAttentionBatchFn
	if indexer {
		finish = r.compressorIndexerBatchFn
	}
	err := r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if emits > 0 {
			if err := cudalaunch.Kernel(r.compressorPoolBatchFn).Grid(uint32(emits), 1, 1).Block(uint32(head), 1, 1).On(r.stream()).Launch(kv, gate, state.stateKV, state.stateScore, pooled, ape.device, uint32(head), uint32(width), uint32(state.ratio)); err != nil {
				return fmt.Errorf("cuda: launch compressor pool batch: %w", err)
			}
		}
		// After the pool, which reads the window this leaves behind.
		if err := cudalaunch.Kernel(r.compressorStateBatchFn).Grid(uint32(rows), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(kv, gate, state.stateKV, state.stateScore, ape.device, uint32(width), uint32(state.ratio), uint32(n)); err != nil {
			return fmt.Errorf("cuda: launch compressor state batch: %w", err)
		}
		if emits > 0 {
			if err := cudalaunch.Kernel(finish).Grid(uint32(emits), 1, 1).Block(uint32(head), 1, 1).On(r.stream()).Launch(pooled, norm.device, state.cache, uint32(state.n), uint32(position), uint32(state.ratio)); err != nil {
				return fmt.Errorf("cuda: launch compressor finish batch: %w", err)
			}
		}
		return nil
	})
	return emits, err
}

// encodeMixedAttention retains the memoized single-block qualification path.
// Production uses encodeMixedAttentionParallel.
func (r *cudaFoundation) encodeMixedAttention(q, kv cudamem.Slice[float32], sinks cudaModelTensor, raw, comp cudamem.Slice[float32], nComp int, allowed cudamem.Slice[uint32], scores, out cudamem.Slice[float32], position, layer int) error {
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 || raw.Len() == 0 || raw.Len()%512 != 0 || comp.Len()%512 != 0 || nComp < 0 || nComp > comp.Len()/512 || allowed.Len() < max(1, nComp) || position < 0 || layer < 2 || out.Len() != 32768+512+32768 {
		return fmt.Errorf("cuda: mixed attention: invalid shape")
	}
	total := min(position+1, raw.Len()/512) + nComp
	if scores.Len() < 64*total {
		return fmt.Errorf("cuda: mixed attention: invalid score capacity")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionFn).Grid(1, 1, 1).Block(64, 1, 1).On(r.stream()).Launch(q, kv, sinks.device, raw, comp, allowed, scores, out, uint32(position), uint32(layer), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention: %w", err)
		}
		return nil
	})
}

// encodeMixedAttentionParallel queues rotation, independent scores, ordered
// normalization, and pairwise values. One stream orders all scratch access.
func (r *cudaFoundation) encodeMixedAttentionParallel(q, kv cudamem.Slice[float32], sinks cudaModelTensor, raw, comp cudamem.Slice[float32], nComp int, allowed cudamem.Slice[uint32], scores, denominators, out cudamem.Slice[float32], position, layer int) error {
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 || raw.Len() == 0 || raw.Len()%512 != 0 || comp.Len()%512 != 0 || nComp < 0 || nComp > comp.Len()/512 || allowed.Len() < max(1, nComp) || denominators.Len() != 64 || position < 0 || layer < 2 || out.Len() != 32768+512+32768 {
		return fmt.Errorf("cuda: mixed attention: invalid shape")
	}
	total := min(position+1, raw.Len()/512) + nComp
	if scores.Len() < 64*total {
		return fmt.Errorf("cuda: mixed attention: invalid score capacity")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionPrepareFn).Grid(65, 1, 1).Block(64, 1, 1).On(r.stream()).Launch(q, kv, sinks.device, raw, comp, allowed, scores, denominators, out, uint32(position), uint32(layer), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention preparation: %w", err)
		}
		if r.mixedAttentionTileScoresFn != 0 && r.mixedAttentionFinalizeFn != 0 && !r.disableMixedAttentionFused {
			tileGridX := uint32((total + 15) / 16)
			if err := cudalaunch.Kernel(r.mixedAttentionTileScoresFn).Grid(tileGridX, 4, 1).Block(256, 1, 1).SharedMem(cudaScoreTileShared).On(r.stream()).Launch(raw, comp, allowed, scores, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
				return fmt.Errorf("cuda: launch mixed attention tile scores: %w", err)
			}
			shmemFinal := uint32(total * 4 * cudaFinalizeHeadsPerBlock)
			if err := cudalaunch.Kernel(r.mixedAttentionFinalizeFn).Grid(64/cudaFinalizeHeadsPerBlock, 1, 1).Block(cudaFinalizeThreads, 1, 1).SharedMem(shmemFinal).On(r.stream()).Launch(sinks.device, raw, comp, allowed, scores, denominators, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
				return fmt.Errorf("cuda: launch mixed attention finalize: %w", err)
			}
			return nil
		}
		if r.mixedAttentionFusedFn != 0 && !r.disableMixedAttentionFused {
			shmem := uint32(total * 4)
			if err := cudalaunch.Kernel(r.mixedAttentionFusedFn).Grid(64, 1, 1).Block(256, 1, 1).SharedMem(shmem).On(r.stream()).Launch(sinks.device, raw, comp, allowed, denominators, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
				return fmt.Errorf("cuda: launch mixed attention fused: %w", err)
			}
			return nil
		}
		if err := cudalaunch.Kernel(r.mixedAttentionScoresFn).Grid(64, uint32((total+63)/64), 1).Block(64, 1, 1).On(r.stream()).Launch(raw, comp, allowed, scores, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention scores: %w", err)
		}
		if err := cudalaunch.Kernel(r.mixedAttentionNormalizeFn).Grid(1, 1, 1).Block(64, 1, 1).On(r.stream()).Launch(sinks.device, allowed, scores, denominators, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention normalization: %w", err)
		}
		if err := cudalaunch.Kernel(r.mixedAttentionValuesFn).Grid(64, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(raw, comp, allowed, scores, denominators, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention values: %w", err)
		}
		return nil
	})
}

// encodeIndexer consumes projected queries and head weights. Up to 512 rows
// need no projections; all are allowed. Above that boundary it selects exactly
// 512 rows, resolving score ties in favor of the lower cache index.
func (r *cudaFoundation) encodeIndexer(q, weights, cache cudamem.Slice[float32], nComp int, qScratch, scores cudamem.Slice[float32], allowed cudamem.Slice[uint32], position int, layer ...int) error {
	if nComp < 0 || nComp > cache.Len()/128 || cache.Len()%128 != 0 || allowed.Len() < max(1, nComp) || position < 0 || nComp > 512 && (q.Len() != 64*128 || weights.Len() != 64 || qScratch.Len() != 64*128 || scores.Len() < nComp) {
		return fmt.Errorf("cuda: indexer: invalid shape")
	}
	layerIdx := -1
	if len(layer) > 0 {
		layerIdx = layer[0]
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if nComp <= 512 {
			return r.encodeFillU32(allowed, nComp, 1)
		}
		if err := r.markCUDAPhase("indexer-query", position, layerIdx); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.indexerQueryFn).Grid(64, 1, 1).Block(128, 1, 1).On(r.stream()).Launch(q, qScratch, uint32(position)); err != nil {
			return fmt.Errorf("cuda: launch indexer query: %w", err)
		}
		if err := r.markCUDAPhase("indexer-score", position, layerIdx); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.indexerScoreFn).Grid(uint32((nComp+cudaIndexRowsPerBlock-1)/cudaIndexRowsPerBlock), 1, 1).Block(32*cudaIndexRowsPerBlock, 1, 1).On(r.stream()).Launch(qScratch, weights, cache, scores, uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch indexer score: %w", err)
		}
		if err := r.markCUDAPhase("indexer-select", position, layerIdx); err != nil {
			return err
		}
		// One block per cudaSelectBlock elements. This is an optimization, not
		// a contract: ds4_indexer_select strides over its input and stages the
		// scores in static shared memory, so it is correct at any grid, and its
		// fallback threshold-and-sort path parks the extra blocks itself.
		selectBlocks := (uint32(nComp) + cudaSelectBlock - 1) / cudaSelectBlock
		if selectBlocks == 0 {
			selectBlocks = 1
		}
		if err := cudalaunch.Kernel(r.indexerSelectFn).Grid(selectBlocks, 1, 1).Block(cudaSelectBlock, 1, 1).On(r.stream()).Launch(scores, allowed, uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch indexer selection: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeFillU32(dst cudamem.Slice[uint32], count int, value uint32) error {
	if count < 0 || count > dst.Len() {
		return fmt.Errorf("cuda: fill uint32: invalid length")
	}
	if count == 0 {
		return nil
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.fillU32Fn).Grid(uint32((count+255)/256), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(dst, uint32(count), value); err != nil {
			return fmt.Errorf("cuda: launch fill uint32: %w", err)
		}
		return nil
	})
}

// mixedAttentionPrepareBatch queues rotation for all Q heads across the batch,
// and normalizes and quantizes KV for all tokens across the batch.
func (r *cudaFoundation) mixedAttentionPrepareBatch(q, kv, out cudamem.Slice[float32], basePosition, rawCap, numTokens int, denseRoPE bool) error {
	if numTokens <= 0 || q.Len() < numTokens*64*512 || kv.Len() < numTokens*512 || out.Len() < numTokens*cudaAttentionStride || rawCap <= 0 || basePosition < 0 {
		return fmt.Errorf("cuda: mixed attention prepare batch: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var dense uint32
		if denseRoPE {
			dense = 1
		}
		if err := cudalaunch.Kernel(r.mixedAttentionPrepareBatchFn).
			Grid(65, uint32(numTokens), 1).
			Block(64, 1, 1).
			On(r.stream()).
			Launch(q, kv, out, uint32(basePosition), uint32(rawCap), uint32(numTokens), dense); err != nil {
			return fmt.Errorf("cuda: launch mixed attention prepare batch: %w", err)
		}
		return nil
	})
}

// mixedAttentionPrepareKVBatch queues the key/value half of
// mixedAttentionPrepareBatch, for query rows headRMSRoPEBatch already rotated
// in place. It leaves the query region of out untouched.
func (r *cudaFoundation) mixedAttentionPrepareKVBatch(kv, out cudamem.Slice[float32], basePosition, numTokens int, denseRoPE bool) error {
	if numTokens <= 0 || kv.Len() < numTokens*512 || out.Len() < numTokens*cudaAttentionStride || basePosition < 0 {
		return fmt.Errorf("cuda: mixed attention prepare key/value batch: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var dense uint32
		if denseRoPE {
			dense = 1
		}
		if err := cudalaunch.Kernel(r.mixedAttentionPrepareKVBatchFn).
			Grid(uint32(numTokens), 1, 1).
			Block(64, 1, 1).
			On(r.stream()).
			Launch(kv, out, uint32(basePosition), uint32(numTokens), dense); err != nil {
			return fmt.Errorf("cuda: launch mixed attention prepare key/value batch: %w", err)
		}
		return nil
	})
}

// headRMSRoPEBatch normalizes all 64 query heads of numTokens tokens in place
// and rotates each head's last 64 dimensions at its token's position. It
// equals encodeRMS over the heads followed by the query half of
// mixedAttentionPrepareBatch, bit for bit, without the copy into the
// attention scratch.
func (r *cudaFoundation) headRMSRoPEBatch(q cudamem.Slice[float32], basePosition, numTokens int, denseRoPE bool) error {
	if numTokens <= 0 || q.Len() < numTokens*cudaAttentionHeads || basePosition < 0 {
		return fmt.Errorf("cuda: head RMS RoPE batch: invalid shape")
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		var dense uint32
		if denseRoPE {
			dense = 1
		}
		if err := cudalaunch.Kernel(r.headRMSRoPEBatchFn).
			Grid(uint32(numTokens*64), 1, 1).
			Block(256, 1, 1).
			On(r.stream()).
			Launch(q, uint32(basePosition), uint32(numTokens), dense); err != nil {
			return fmt.Errorf("cuda: launch head RMS RoPE batch: %w", err)
		}
		return nil
	})
}

// mixedAttentionTileScoresBatch computes dot product scores against raw and compressed KV
// for all tokens in the batch.
func (r *cudaFoundation) mixedAttentionTileScoresBatch(raw, comp cudamem.Slice[float32], allowed cudamem.Slice[uint32], scores, out cudamem.Slice[float32], tokenNComp cudamem.Slice[uint32], maxTotal, basePosition, rawCap, maxComp, scoreStride, numTokens int) error {
	if numTokens <= 0 || raw.Len() == 0 || comp.Len()%512 != 0 || allowed.Len() < numTokens*maxComp || scores.Len() < numTokens*scoreStride || out.Len() < numTokens*cudaAttentionStride || tokenNComp.Len() < numTokens {
		return fmt.Errorf("cuda: mixed attention tile scores batch: invalid shape")
	}
	tileGridX := uint32((maxTotal + 31) / 32)
	if tileGridX == 0 {
		tileGridX = 1
	}
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionTileScoresBatchFn).
			Grid(tileGridX, 4, uint32(numTokens)).
			Block(512, 1, 1).
			SharedMem(99072).
			On(r.stream()).
			Launch(raw, comp, allowed, scores, out, tokenNComp, uint32(basePosition), uint32(rawCap), uint32(maxComp), uint32(scoreStride), uint32(numTokens)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention tile scores batch: %w", err)
		}
		return nil
	})
}

// mixedAttentionFinalizeBatch computes softmax and values against raw and compressed KV
// for all tokens in the batch.
func (r *cudaFoundation) mixedAttentionFinalizeBatch(sinks cudaModelTensor, raw, comp cudamem.Slice[float32], allowed cudamem.Slice[uint32], scores, denominators, out cudamem.Slice[float32], tokenNComp cudamem.Slice[uint32], maxTotal, basePosition, rawCap, maxComp, scoreStride, numTokens int) error {
	if numTokens <= 0 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 || raw.Len() == 0 || comp.Len()%512 != 0 || allowed.Len() < numTokens*maxComp || scores.Len() < numTokens*scoreStride || out.Len() < numTokens*cudaAttentionStride || tokenNComp.Len() < numTokens {
		return fmt.Errorf("cuda: mixed attention finalize batch: invalid shape")
	}
	shmemFinal := uint32(maxTotal * 4)
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionFinalizeBatchFn).
			Grid(64, uint32(numTokens), 1).
			Block(512, 1, 1).
			SharedMem(shmemFinal).
			On(r.stream()).
			Launch(sinks.device, raw, comp, allowed, scores, denominators, out, tokenNComp, uint32(basePosition), uint32(rawCap), uint32(maxComp), uint32(scoreStride), uint32(numTokens)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention finalize batch: %w", err)
		}
		return nil
	})
}

// mixedAttentionCommitRaw commits the last min(numTokens, rawCap) tokens into raw KV cache.
func (r *cudaFoundation) mixedAttentionCommitRaw(raw, out cudamem.Slice[float32], basePosition, rawCap, numTokens int) error {
	if numTokens <= 0 || rawCap <= 0 || raw.Len() < rawCap*512 || out.Len() < numTokens*cudaAttentionStride {
		return fmt.Errorf("cuda: mixed attention commit raw: invalid shape")
	}
	count := min(numTokens, rawCap)
	return r.withContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionCommitRawFn).
			Grid(uint32(count), 1, 1).
			Block(64, 1, 1).
			On(r.stream()).
			Launch(raw, out, uint32(basePosition), uint32(rawCap), uint32(numTokens)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention commit raw: %w", err)
		}
		return nil
	})
}

//go:embed cuda_compressed.cu
var cudaCompressedSource string
