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
	err := r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.compressorPoolFn).Grid(1).Block(uint32(head)).On(r.device.DefaultStream()).Launch(kv, gate, state.stateKV, state.stateScore, state.pooled, ape.device, uint32(head), uint32(width), uint32(state.ratio), uint32(position)); err != nil {
			return fmt.Errorf("cuda: launch compressor pool: %w", err)
		}
		if !emit {
			return nil
		}
		fn := r.compressorAttentionFn
		if indexer {
			fn = r.compressorIndexerFn
		}
		if err := cudalaunch.Kernel(fn).Grid(1).Block(1).On(r.device.DefaultStream()).Launch(state.pooled, norm.device, state.cache, uint32(state.n), uint32(position+1-state.ratio)); err != nil {
			return fmt.Errorf("cuda: launch compressor finish: %w", err)
		}
		return nil
	})
	return emit, err
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(q, kv, sinks.device, raw, comp, allowed, scores, out, uint32(position), uint32(layer), uint32(raw.Len()/512), uint32(nComp)); err != nil {
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.mixedAttentionPrepareFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(q, kv, sinks.device, raw, comp, allowed, scores, denominators, out, uint32(position), uint32(layer), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention preparation: %w", err)
		}
		if r.mixedAttentionTileScoresFn != 0 && r.mixedAttentionFinalizeFn != 0 && !r.disableMixedAttentionFused {
			tileGridX := uint32((total + 15) / 16)
			if err := cudalaunch.Kernel(r.mixedAttentionTileScoresFn).Grid(tileGridX, 4).Block(256).SharedMem(66048).On(r.device.DefaultStream()).Launch(raw, comp, allowed, scores, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
				return fmt.Errorf("cuda: launch mixed attention tile scores: %w", err)
			}
			shmemFinal := uint32(total * 4)
			if err := cudalaunch.Kernel(r.mixedAttentionFinalizeFn).Grid(64).Block(512).SharedMem(shmemFinal).On(r.device.DefaultStream()).Launch(sinks.device, raw, comp, allowed, scores, denominators, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
				return fmt.Errorf("cuda: launch mixed attention finalize: %w", err)
			}
			return nil
		}
		if r.mixedAttentionFusedFn != 0 && !r.disableMixedAttentionFused {
			shmem := uint32(total * 4)
			if err := cudalaunch.Kernel(r.mixedAttentionFusedFn).Grid(64).Block(256).SharedMem(shmem).On(r.device.DefaultStream()).Launch(sinks.device, raw, comp, allowed, denominators, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
				return fmt.Errorf("cuda: launch mixed attention fused: %w", err)
			}
			return nil
		}
		if err := cudalaunch.Kernel(r.mixedAttentionScoresFn).Grid(64, uint32((total+63)/64)).Block(64).On(r.device.DefaultStream()).Launch(raw, comp, allowed, scores, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention scores: %w", err)
		}
		if err := cudalaunch.Kernel(r.mixedAttentionNormalizeFn).Grid(1).Block(64).On(r.device.DefaultStream()).Launch(sinks.device, allowed, scores, denominators, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch mixed attention normalization: %w", err)
		}
		if err := cudalaunch.Kernel(r.mixedAttentionValuesFn).Grid(64).Block(256).On(r.device.DefaultStream()).Launch(raw, comp, allowed, scores, denominators, out, uint32(position), uint32(raw.Len()/512), uint32(nComp)); err != nil {
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if nComp <= 512 {
			return r.encodeFillU32(allowed, nComp, 1)
		}
		if err := r.markCUDAPhase("indexer-query", position, layerIdx); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.indexerQueryFn).Grid(64).Block(1).On(r.device.DefaultStream()).Launch(q, qScratch, uint32(position)); err != nil {
			return fmt.Errorf("cuda: launch indexer query: %w", err)
		}
		if err := r.markCUDAPhase("indexer-score", position, layerIdx); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.indexerScoreFn).Grid(uint32((nComp + 63) / 64)).Block(64).On(r.device.DefaultStream()).Launch(qScratch, weights, cache, scores, uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch indexer score: %w", err)
		}
		if err := r.markCUDAPhase("indexer-select", position, layerIdx); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.indexerSelectFn).Grid(1).Block(512).On(r.device.DefaultStream()).Launch(scores, allowed, uint32(nComp)); err != nil {
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
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.fillU32Fn).Grid(uint32((count+255)/256)).Block(256).On(r.device.DefaultStream()).Launch(dst, uint32(count), value); err != nil {
			return fmt.Errorf("cuda: launch fill uint32: %w", err)
		}
		return nil
	})
}

//go:embed cuda_compressed.cu
var cudaCompressedSource string
