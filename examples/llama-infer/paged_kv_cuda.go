//go:build !darwin && !tpu

package main

// Paged KV decode for CUDA.
//
// The CUDA path stores KV densely: one MaxSeqLen-sized f32 allocation for the
// single-stream engine cache and another per serving sequence, so a sequence
// pays for the whole context window from its first token and no storage is
// shared between sequences. serving_backend_linux.go records the consequence:
// CUDA retains no shared prefix cache.
//
// The pieces that fix this already exist in the library: cudakv.PagedCache is
// a block pool with a per-sequence block table, and cudaattn.CompilePaged
// emits the two kernels that read through one, PagedRoPEStoreKV and
// BatchedPagedAttentionDecode. This file owns the pool and the kernels;
// paged_kv_step_cuda.go binds them to a forward call, and paged_kv.go holds
// the host-only switches and table layout. It adds no kernel and changes no
// dense code path.
//
// UNQUALIFIED. Reachable, off by default: LLAMA_INFER_PAGED_KV=1 at engine
// compile time replaces the dense caches with this pool for the single-stream
// decode, the batched prefill and the serving decode batch. Paging changes the
// memory layout the attention kernel reads, so a type-check proves nothing
// about it, and nothing here has run on a real device: no host that built it
// has a GPU. The default path is byte-for-byte unchanged.
//
// The test that qualifies it is TestPagedKVEngineMatchesDense
// (paged_kv_engine_linux_test.go), which compiles a dense and a paged engine
// from one deterministic config and requires the same tokens and logits from
// both, single-stream and batched. TestPagedKVMatchesDense qualifies this file
// alone, below the engine. Both skip without a device, and a skip is not a
// pass: until one of them reports a real pass the paged path must not be
// described as working.
//
// Known limits of this first wiring, all deliberate:
//   - No sharing yet. Every sequence gets fresh blocks and frees them when it
//     ends. The pool is what a CUDA shared prefix cache needs to sit on, and
//     cudakv now has the join for it (Retain, Release, AllocSequenceWithPrefix,
//     RadixTree.Insert reporting adopted blocks), but no engine code calls it.
//   - Exclusive with LLAMA_INFER_CUDA_GRAPH=1: the paged step uploads its side
//     tables synchronously and is not graph-captured. The engine refuses the
//     pair at compile time.
//   - The pool is fixed at compile time from LLAMA_INFER_PAGED_KV_SEQS (default
//     8) plus one for the single-stream sequence, before any server names its
//     capacity. Admission past that fails with an out-of-blocks error.
//   - KV observation for inferbench (observed_output_linux.go) reads the dense
//     cache and records nothing for a paged sequence.
//   - Not safe for concurrent use: the side tables and the rotated-query
//     scratch are shared by every call, so callers serialize under the same
//     generateMu the dense path uses.

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// pagedKVRuntime owns the block pool and the two paged kernels.
type pagedKVRuntime struct {
	dev   *cuda.Device
	cache *cudakv.PagedCache
	mod   *cuda.Module

	store  *cuda.Kernel
	decode *boundedKernel

	cfg       pagedKVConfig
	layers    int
	qHeads    int
	kvHeads   int
	headDim   int
	maxSeqLen int

	// Scratch device buffers for the int32 side tables the kernels read.
	// They are sized once for the largest batch the pool can hold.
	blockTable cudamem.Slice[int32]
	seqIndex   cudamem.Slice[int32]
	positions  cudamem.Slice[int32]
	seqLens    cudamem.Slice[int32]
	status     cudamem.Slice[int32]

	// qRoped receives the rotated queries of one step, one row per sequence
	// entry; PagedRoPEStoreKV does not rotate q in place.
	qRoped cudamem.Slice[float32]

	maxSeqs   int
	maxTokens int

	nextSeq int64
}

// newPagedKVRuntime compiles the paged kernels and allocates the block pool.
// The pool and the side tables are charged to tracker, which may be nil.
// maxSeqs bounds the sequence entries of one step: serving rows, or the rows of
// one prefill tile.
func newPagedKVRuntime(dev *cuda.Device, tracker *cudamem.Tracker, cfg pagedKVConfig, layers, qHeads, kvHeads, headDim, maxSeqLen, maxSeqs int) (r *pagedKVRuntime, err error) {
	if dev == nil {
		return nil, errors.New("paged kv: nil device")
	}
	if maxSeqs < 1 {
		return nil, fmt.Errorf("paged kv: max sequences %d must be positive", maxSeqs)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("paged kv: compute capability: %w", err)
	}
	compiled, err := cudaattn.CompilePaged(major, minor)
	if err != nil {
		return nil, fmt.Errorf("paged kv: compile paged attention: %w", err)
	}
	mod, err := loadCUDAProgram(dev, compiled)
	if err != nil {
		return nil, fmt.Errorf("paged kv: load paged attention: %w", err)
	}
	rt := &pagedKVRuntime{
		dev: dev, mod: mod, cfg: cfg,
		layers: layers, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim,
		maxSeqLen: maxSeqLen, maxSeqs: maxSeqs, maxTokens: maxSeqs * maxSeqLen,
	}
	defer func() {
		if err != nil {
			rt.Close()
		}
	}()
	if rt.store, err = mod.Function("PagedRoPEStoreKV"); err != nil {
		return nil, fmt.Errorf("paged kv: PagedRoPEStoreKV: %w", err)
	}
	decodeRaw, err := mod.Function("BatchedPagedAttentionDecode")
	if err != nil {
		return nil, fmt.Errorf("paged kv: BatchedPagedAttentionDecode: %w", err)
	}
	if rt.decode, err = newBoundedKernel(dev, decodeRaw); err != nil {
		return nil, fmt.Errorf("paged kv: %w", err)
	}
	if rt.cache, err = cudakv.NewPagedCacheTracked(dev, tracker, layers, kvHeads, headDim, cfg.BlockSize, cfg.TotalBlocks); err != nil {
		return nil, fmt.Errorf("paged kv: block pool: %w", err)
	}
	if rt.blockTable, err = cudamem.AllocTracked[int32](dev, tracker, maxSeqs*cfg.MaxBlocksPerSeq); err != nil {
		return nil, fmt.Errorf("paged kv: block table: %w", err)
	}
	if rt.seqIndex, err = cudamem.AllocTracked[int32](dev, tracker, rt.maxTokens); err != nil {
		return nil, fmt.Errorf("paged kv: sequence index: %w", err)
	}
	if rt.positions, err = cudamem.AllocTracked[int32](dev, tracker, rt.maxTokens); err != nil {
		return nil, fmt.Errorf("paged kv: positions: %w", err)
	}
	if rt.seqLens, err = cudamem.AllocTracked[int32](dev, tracker, maxSeqs); err != nil {
		return nil, fmt.Errorf("paged kv: sequence lengths: %w", err)
	}
	if rt.status, err = cudamem.AllocTracked[int32](dev, tracker, 1); err != nil {
		return nil, fmt.Errorf("paged kv: status: %w", err)
	}
	if rt.qRoped, err = cudamem.AllocTracked[float32](dev, tracker, maxSeqs*qHeads*headDim); err != nil {
		return nil, fmt.Errorf("paged kv: rotated queries: %w", err)
	}
	return rt, nil
}

// layerPool returns whole-pool views of one layer's K and V storage.
//
// The kernels index the pool by physical slot, so they need its base and full
// extent, and cudakv.PagedCache exports no accessor for that: Page returns a
// single slot. Widening slot zero is sound rather than a trick, because
// cudamem.Slice.Slice carries the parent allocation's capacity, so the widened
// bounds are still checked against the real allocation and a mistake panics
// rather than reading past it. An exported pool accessor in x/cudakv would say
// this more directly.
func (r *pagedKVRuntime) layerPool(layer int) (k, v cudamem.Slice[float32], err error) {
	page, err := r.cache.Page(layer, 0)
	if err != nil {
		return k, v, fmt.Errorf("paged kv: layer %d pool: %w", layer, err)
	}
	elements := r.cfg.TotalBlocks * r.cfg.BlockSize * r.kvHeads * r.headDim
	return page.K.Slice(0, elements), page.V.Slice(0, elements), nil
}

// Close releases the pool, the side tables and the module. It is safe on a
// partially constructed runtime and safe to call more than once.
func (r *pagedKVRuntime) Close() error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.cache != nil {
		errs = append(errs, r.cache.Close())
		r.cache = nil
	}
	for _, s := range []interface{ Free() error }{r.blockTable, r.seqIndex, r.positions, r.seqLens, r.status, r.qRoped} {
		if err := s.Free(); err != nil {
			errs = append(errs, err)
		}
	}
	if r.mod != nil {
		errs = append(errs, r.mod.Close())
		r.mod = nil
	}
	return errors.Join(errs...)
}

// pagedThreads is the block width both paged kernels are launched with. The
// dense decode path launches one thread per head (Block(1)); the paged kernels
// are written for a warp, so this is also the only place the CUDA decode stops
// leaving 31 lanes idle.
const pagedThreads = 32

// UploadBatch stages the int32 side tables the kernels read, ordered on stream
// after whatever is already queued there. blockTable is row-major
// [seqs][MaxBlocksPerSeq] physical slot indices, seqIndex and positions are per
// token, and seqLens is per sequence. It must be called once per step, before
// StoreKV and Decode for that step, and it clears the kernels' status word.
func (r *pagedKVRuntime) UploadBatch(stream *cuda.Stream, seqIndex, positions, seqLens, blockTable []int32) error {
	if len(seqIndex) != len(positions) {
		return fmt.Errorf("paged kv: %d sequence indices for %d positions", len(seqIndex), len(positions))
	}
	if len(seqIndex) > r.maxTokens {
		return fmt.Errorf("paged kv: %d tokens exceeds pool capacity %d", len(seqIndex), r.maxTokens)
	}
	if len(seqLens) > r.maxSeqs {
		return fmt.Errorf("paged kv: %d sequences exceeds capacity %d", len(seqLens), r.maxSeqs)
	}
	if want := len(seqLens) * r.cfg.MaxBlocksPerSeq; len(blockTable) != want {
		return fmt.Errorf("paged kv: block table has %d entries, want %d", len(blockTable), want)
	}
	// The host slices may be reused by the caller as soon as this returns: a
	// copy from pageable memory is staged before the async call returns.
	return r.dev.WithContext(func() error {
		for _, up := range []struct {
			dst cudamem.Slice[int32]
			src []int32
		}{
			{r.seqIndex, seqIndex},
			{r.positions, positions},
			{r.seqLens, seqLens},
			{r.blockTable, blockTable},
			{r.status, []int32{0}},
		} {
			if len(up.src) == 0 {
				continue
			}
			if err := up.dst.Slice(0, len(up.src)).CopyFromHostAsync(up.src, stream); err != nil {
				return err
			}
		}
		return nil
	})
}

// StoreKV applies RoPE to one layer's q and k and scatters k and v into the
// block pool at the positions staged by UploadBatch. qRoped receives the
// rotated queries; the kernel does not write q in place. Launches are counted
// on counter, which may be nil.
func (r *pagedKVRuntime) StoreKV(stream *cuda.Stream, counter *cudalaunch.Counter, layer int, q, k, v, qRoped cudamem.Slice[float32], tokens, seqs int, ropeTheta float32) error {
	kPool, vPool, err := r.layerPool(layer)
	if err != nil {
		return err
	}
	width := r.qHeads * r.headDim
	if kv := r.kvHeads * r.headDim; kv > width {
		width = kv
	}
	gridX := uint32((width + pagedThreads - 1) / pagedThreads)
	return cudalaunch.OwnedKernel(r.store).Grid(gridX, uint32(tokens)).Block(pagedThreads).On(stream).Count(counter).Launch(
		q, k, v, qRoped, kPool, vPool, r.blockTable, r.seqIndex, r.positions,
		int32(tokens), int32(r.cfg.MaxBlocksPerSeq), int32(r.cfg.BlockSize),
		int32(r.qHeads), int32(r.kvHeads), int32(r.headDim),
		int32(seqs), int32(r.cfg.TotalBlocks), ropeTheta, r.status,
	)
}

// Decode computes one attention row per sequence for one layer, reading K and V
// through each sequence's block table. currentQ holds the rotated query for
// each sequence's current token and out receives seqs*qHeads*headDim values.
//
// The shared-memory request is sized from the context window, so it is the
// same on every step and is checked against the kernel's raised ceiling
// through the same boundedKernel the dense sites use.
func (r *pagedKVRuntime) Decode(stream *cuda.Stream, counter *cudalaunch.Counter, layer int, currentQ, out cudamem.Slice[float32], seqs int) error {
	kPool, vPool, err := r.layerPool(layer)
	if err != nil {
		return err
	}
	shared, err := cudaattn.PagedSharedMemoryBytes(r.maxSeqLen, pagedThreads)
	if err != nil {
		return fmt.Errorf("paged kv: shared memory for %d tokens: %w", r.maxSeqLen, err)
	}
	sharedMem, err := r.decode.sharedMem(shared)
	if err != nil {
		return fmt.Errorf("paged kv: attention over %d tokens: %w", r.maxSeqLen, err)
	}
	scale := float32(1 / math.Sqrt(float64(r.headDim)))
	return cudalaunch.OwnedKernel(r.decode.fn).Grid(uint32(r.qHeads), uint32(seqs)).Block(pagedThreads).SharedMem(sharedMem).On(stream).Count(counter).Launch(
		currentQ, kPool, vPool, r.blockTable, r.seqLens, out,
		int32(r.cfg.MaxBlocksPerSeq), int32(r.cfg.BlockSize),
		int32(r.qHeads), int32(r.kvHeads), int32(r.headDim),
		scale, int32(seqs), int32(r.cfg.TotalBlocks), int32(r.maxSeqLen), r.status,
	)
}

// CheckStatus reports the error status the kernels write. Both kernels set a
// nonzero status rather than faulting when a block table entry is out of range,
// so a caller that never reads it cannot tell a correct result from a refused
// one. Read it after synchronizing the stream.
func (r *pagedKVRuntime) CheckStatus() error {
	var status [1]int32
	if err := r.dev.WithContext(func() error { return r.status.CopyToHost(status[:]) }); err != nil {
		return fmt.Errorf("paged kv: read kernel status: %w", err)
	}
	if status[0] != 0 {
		return fmt.Errorf("paged kv: kernel reported status %d (out-of-range block table or position)", status[0])
	}
	return nil
}
