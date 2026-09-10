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
// The pieces that fix this already exist in the library and were simply never
// called: cudakv.PagedCache is a block pool with a per-sequence block table,
// and cudaattn.CompilePaged emits the two kernels that read through one --
// PagedRoPEStoreKV and BatchedPagedAttentionDecode. This file wires them
// together. It adds no kernel and changes no dense code path.
//
// UNQUALIFIED, AND NOT YET REACHABLE. Paging changes the memory layout the
// attention kernel reads, so a type-check proves nothing about it. Nothing here
// has run on a real device: this session has no GPU, and the L4 VM was released
// and terminated before the work began.
//
// Nothing in the engine calls any of this yet, pagedKVEnabled included. Calling
// it "behind a flag" would overstate it: the flag exists so that whoever wires
// this up has the off switch already written, but today the only caller is
// paged_kv_cuda_linux_test.go. Wiring it into the serving path is deliberately
// left until TestPagedKVMatchesDense reports a pass from real hardware, and
// until it can be coordinated with the engine_other.go work in flight. Until
// then this must not be described as working.
//
// Retained deliberately in this unreachable state rather than wired or deleted:
// the hardware qualification it needs is written (the test above) and the code
// is the smallest thing that test can run against. Whoever wires it should also
// read the two x/cudakv gaps recorded alongside it. PagedCache.FreeSequence
// returns a sequence's whole block list unconditionally, while RadixTree keeps
// its own pin/unpin refcount and hands blocks back to "the caller's allocator";
// nothing connects the two, so a naive shared prefix cache recycles blocks the
// tree still has pinned for another sequence. And NewPagedCache has no tracked
// constructor, which pagedKVUntrackedPoolBytes exists to make visible.

import (
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// pagedKVEnabled reports whether the paged KV path was explicitly requested.
// It is opt-in rather than a Config field so that enabling it is always a
// deliberate act by whoever is holding the device, and so that this file stays
// self-contained while another session edits the shared engine files.
func pagedKVEnabled() bool { return os.Getenv("LLAMA_INFER_PAGED_KV") == "1" }

// pagedKVConfig sizes the block pool. TotalBlocks is the pool shared by every
// resident sequence, which is the whole point: MaxBlocksPerSeq bounds one
// sequence's block table but reserves nothing.
type pagedKVConfig struct {
	BlockSize       int
	TotalBlocks     int
	MaxBlocksPerSeq int
}

// defaultPagedKVConfig derives a pool from the model's context window and the
// number of sequences the server admits. BlockSize 16 matches the Metal prefix
// cache's block size, so the two backends describe prefixes the same way.
func defaultPagedKVConfig(maxSeqLen, capacity int) pagedKVConfig {
	const blockSize = 16
	perSeq := (maxSeqLen + blockSize - 1) / blockSize
	if capacity < 1 {
		capacity = 1
	}
	return pagedKVConfig{
		BlockSize:       blockSize,
		TotalBlocks:     perSeq * capacity,
		MaxBlocksPerSeq: perSeq,
	}
}

// pagedKVRuntime owns the block pool and the two paged kernels.
//
// It is not safe for concurrent use: the scratch device buffers below are
// shared by every call, so a caller must serialize StoreKV and Decode the same
// way the dense path serializes its engine scratch.
type pagedKVRuntime struct {
	dev   *cuda.Device
	cache *cudakv.PagedCache
	mod   *cuda.Module

	store  *cuda.Kernel
	decode *cuda.Kernel

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

	maxSeqs   int
	maxTokens int
}

// newPagedKVRuntime compiles the paged kernels and allocates the block pool.
// tracker may be nil; when it is not, the side tables are charged to it. The
// pool itself is not tracked, because cudakv.NewPagedCache allocates through
// cudamem.Alloc and exposes no tracked constructor -- see the note on
// pagedKVUntrackedPoolBytes.
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
	if rt.decode, err = mod.Function("BatchedPagedAttentionDecode"); err != nil {
		return nil, fmt.Errorf("paged kv: BatchedPagedAttentionDecode: %w", err)
	}
	if rt.cache, err = cudakv.NewPagedCache(dev, layers, kvHeads, headDim, cfg.BlockSize, cfg.TotalBlocks); err != nil {
		return nil, fmt.Errorf("paged kv: block pool: %w", err)
	}
	alloc := func(n int) (cudamem.Slice[int32], error) {
		if tracker != nil {
			return cudamem.AllocTracked[int32](dev, tracker, n)
		}
		return cudamem.Alloc[int32](dev, n)
	}
	if rt.blockTable, err = alloc(maxSeqs * cfg.MaxBlocksPerSeq); err != nil {
		return nil, fmt.Errorf("paged kv: block table: %w", err)
	}
	if rt.seqIndex, err = alloc(rt.maxTokens); err != nil {
		return nil, fmt.Errorf("paged kv: sequence index: %w", err)
	}
	if rt.positions, err = alloc(rt.maxTokens); err != nil {
		return nil, fmt.Errorf("paged kv: positions: %w", err)
	}
	if rt.seqLens, err = alloc(maxSeqs); err != nil {
		return nil, fmt.Errorf("paged kv: sequence lengths: %w", err)
	}
	if rt.status, err = alloc(1); err != nil {
		return nil, fmt.Errorf("paged kv: status: %w", err)
	}
	return rt, nil
}

// pagedKVUntrackedPoolBytes reports the device memory the block pool holds
// outside the engine's tracker.
//
// cudakv.NewPagedCache allocates with cudamem.Alloc, so unlike the dense
// cudakv.NewTracked path the pool is invisible to the engine's memory report. A
// tracked constructor in x/cudakv would close this; that is a change to the
// nvidia module rather than to this example, so it is reported here instead of
// worked around.
func (r *pagedKVRuntime) pagedKVUntrackedPoolBytes() int64 {
	if r == nil {
		return 0
	}
	const bytesPerElement = 4
	const buffersPerLayer = 2 // K and V
	perLayer := int64(r.cfg.TotalBlocks) * int64(r.cfg.BlockSize) * int64(r.kvHeads) * int64(r.headDim)
	return perLayer * int64(r.layers) * buffersPerLayer * bytesPerElement
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
	for _, s := range []interface{ Free() error }{r.blockTable, r.seqIndex, r.positions, r.seqLens, r.status} {
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

// UploadBatch stages the int32 side tables the kernels read. blockTable is
// row-major [seqs][MaxBlocksPerSeq] physical slot indices, seqIndex and
// positions are per token, and seqLens is per sequence. It must be called once
// per step, before StoreKV and Decode for that step.
func (r *pagedKVRuntime) UploadBatch(seqIndex, positions, seqLens, blockTable []int32) error {
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
			if err := up.dst.Slice(0, len(up.src)).CopyFromHost(up.src); err != nil {
				return err
			}
		}
		return nil
	})
}

// StoreKV applies RoPE to one layer's q and k and scatters k and v into the
// block pool at the positions staged by UploadBatch. qRoped receives the
// rotated queries; the kernel does not write q in place.
func (r *pagedKVRuntime) StoreKV(stream *cuda.Stream, layer int, q, k, v, qRoped cudamem.Slice[float32], tokens, seqs int, ropeTheta float32) error {
	kPool, vPool, err := r.layerPool(layer)
	if err != nil {
		return err
	}
	width := r.qHeads * r.headDim
	if kv := r.kvHeads * r.headDim; kv > width {
		width = kv
	}
	gridX := uint32((width + pagedThreads - 1) / pagedThreads)
	return cudalaunch.OwnedKernel(r.store).Grid(gridX, uint32(tokens)).Block(pagedThreads).On(stream).Launch(
		q, k, v, qRoped, kPool, vPool, r.blockTable, r.seqIndex, r.positions,
		int32(tokens), int32(r.cfg.MaxBlocksPerSeq), int32(r.cfg.BlockSize),
		int32(r.qHeads), int32(r.kvHeads), int32(r.headDim),
		int32(seqs), int32(r.cfg.TotalBlocks), ropeTheta, r.status,
	)
}

// Decode computes one attention row per sequence for one layer, reading K and V
// through each sequence's block table. currentQ holds the rotated query for
// each sequence's current token and out receives seqs*qHeads*headDim values.
func (r *pagedKVRuntime) Decode(stream *cuda.Stream, layer int, currentQ, out cudamem.Slice[float32], seqs int) error {
	kPool, vPool, err := r.layerPool(layer)
	if err != nil {
		return err
	}
	shared, err := cudaattn.PagedSharedMemoryBytes(r.maxSeqLen, pagedThreads)
	if err != nil {
		return fmt.Errorf("paged kv: shared memory for %d tokens: %w", r.maxSeqLen, err)
	}
	// Unchecked against the device limit, deliberately and in company. Nothing
	// in nvidia/x offers the dynamic shared-memory opt-in (no
	// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES anywhere), so every
	// attention path here is capped at the default 48 KB per block: 12256
	// tokens for this kernel, 12288 for the dense sites, which hand-compute
	// the same quantity and check it just as little
	// (gpu_model_cuda.go:340, gpu_batch_cuda.go:242). Adding a limit check
	// here alone would imply the dense path has one. Default MaxSeqLen is 2048
	// (engine_other.go:415), so this is a ceiling, not a live defect. Note the
	// failure modes differ: this size comes from maxSeqLen and fails at the
	// first launch, while gpu_batch_cuda.go:242 sizes from the live position
	// and would fail partway through a long generation.
	scale := float32(1 / math.Sqrt(float64(r.headDim)))
	return cudalaunch.OwnedKernel(r.decode).Grid(uint32(r.qHeads), uint32(seqs)).Block(pagedThreads).SharedMem(uint32(shared)).On(stream).Launch(
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
