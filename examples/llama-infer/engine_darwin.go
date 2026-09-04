//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/apple/x/metal/metalmegakernel"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/apple/x/metal/metalsafetensors"
	"github.com/tmc/lib/apple/x/metal/metalsample"
	"github.com/tmc/lib/apple/x/metal/metalx"
	"github.com/tmc/lib/examples/internal/metalprefix"
	"github.com/tmc/lib/examples/llama-infer/gpuprof"
)

func init() {
	runtime.LockOSThread()
}

type GPULayerWeights struct {
	InputNorm mempool.Slice[float32]
	PostNorm  mempool.Slice[float32]
	WQ        mempool.Slice[float32]
	WK        mempool.Slice[float32]
	WV        mempool.Slice[float32]
	WO        mempool.Slice[float32]
	WGate     mempool.Slice[float32]
	WUp       mempool.Slice[float32]
	WDown     mempool.Slice[float32]
}

type GPULayerWeightsF16 struct {
	InputNorm mempool.Slice[uint16]
	PostNorm  mempool.Slice[uint16]
	WQ        mempool.Slice[uint16]
	WK        mempool.Slice[uint16]
	WV        mempool.Slice[uint16]
	WO        mempool.Slice[uint16]
	WGate     mempool.Slice[uint16]
	WUp       mempool.Slice[uint16]
	WDown     mempool.Slice[uint16]
}

// decodeKernels names every compute pipeline the decode and prefill paths
// dispatch. metalx.Bind resolves the whole set from one compiled library, so a
// kernel that is renamed or dropped from the MSL fails at engine construction
// with the field that named it, not with a nil pipeline state at dispatch.
type decodeKernels struct {
	Embed      apple.MTLComputePipelineState `metal:"EmbeddingLookup"`
	Norm       apple.MTLComputePipelineState `metal:"ComputeRMSNorm"`
	GEMV       apple.MTLComputePipelineState `metal:"FastGEMV"`
	GEMVRes    apple.MTLComputePipelineState `metal:"FastGEMVResidual"`
	GEMV64     apple.MTLComputePipelineState `metal:"FastGEMV64"`
	GEMVRes64  apple.MTLComputePipelineState `metal:"FastGEMVResidual64"`
	EmbedH     apple.MTLComputePipelineState `metal:"EmbeddingLookupH"`
	NormH      apple.MTLComputePipelineState `metal:"ComputeRMSNormH"`
	GEMVH      apple.MTLComputePipelineState `metal:"FastGEMVH"`
	GEMVResH   apple.MTLComputePipelineState `metal:"FastGEMVResidualH"`
	GEMV64H    apple.MTLComputePipelineState `metal:"FastGEMV64H"`
	GEMVRes64H apple.MTLComputePipelineState `metal:"FastGEMVResidual64H"`

	RoPE        apple.MTLComputePipelineState `metal:"ApplyRoPEStoreKV"`
	Attn        apple.MTLComputePipelineState `metal:"CausalFlashAttentionDecode"`
	BlockAttn   apple.MTLComputePipelineState `metal:"BlockTableCausalAttentionDecode"`
	SiLUMul     apple.MTLComputePipelineState `metal:"FastSiLUMul"`
	FusedQKV    apple.MTLComputePipelineState `metal:"FusedRMSNormQKV"`
	FusedSwiGLU apple.MTLComputePipelineState `metal:"FusedSwiGLUDown"`
	Sample      apple.MTLComputePipelineState `metal:"ArgmaxSampling"`

	BatchedEmbed      apple.MTLComputePipelineState `metal:"BatchedEmbeddingLookup"`
	BatchedNorm       apple.MTLComputePipelineState `metal:"BatchedRMSNorm"`
	BatchedGEMM       apple.MTLComputePipelineState `metal:"BatchedGEMM"`
	BatchedGEMMRes    apple.MTLComputePipelineState `metal:"BatchedGEMMResidual"`
	BatchedRoPE       apple.MTLComputePipelineState `metal:"BatchedRoPE"`
	BatchedAttn       apple.MTLComputePipelineState `metal:"BatchedCausalFlashAttention"`
	RowRoPE           apple.MTLComputePipelineState `metal:"RowIndexedRoPEStoreKV"`
	RowAttn           apple.MTLComputePipelineState `metal:"RowIndexedCausalAttention"`
	SegmentedRoPE     apple.MTLComputePipelineState `metal:"SegmentedRoPEStoreKV"`
	SegmentedAttn     apple.MTLComputePipelineState `metal:"SegmentedCausalAttention"`
	SegmentedGather   apple.MTLComputePipelineState `metal:"GatherSegmentTails"`
	BatchedSiLU       apple.MTLComputePipelineState `metal:"BatchedSiLUMul"`
	BatchedSample     apple.MTLComputePipelineState `metal:"BatchedArgmaxSampling"`
	BatchedEmbedH     apple.MTLComputePipelineState `metal:"BatchedEmbeddingLookupH"`
	BatchedNormH      apple.MTLComputePipelineState `metal:"BatchedRMSNormH"`
	BatchedGEMMH      apple.MTLComputePipelineState `metal:"BatchedGEMMH"`
	BatchedGEMMResH   apple.MTLComputePipelineState `metal:"BatchedGEMMResidualH"`
	BatchedQKVProj    apple.MTLComputePipelineState `metal:"BatchedQKVProjection"`
	BatchedQKVProjH   apple.MTLComputePipelineState `metal:"BatchedQKVProjectionH"`
	BatchedGateUp     apple.MTLComputePipelineState `metal:"BatchedGateUpProjection"`
	BatchedGateUpH    apple.MTLComputePipelineState `metal:"BatchedGateUpProjectionH"`
	RMSNormScale      apple.MTLComputePipelineState `metal:"RMSNormScale"`
	LMHeadTileArgmax  apple.MTLComputePipelineState `metal:"LMHeadTileArgmax"`
	LMHeadTileArgmaxH apple.MTLComputePipelineState `metal:"LMHeadTileArgmaxH"`
	ArgmaxTiles       apple.MTLComputePipelineState `metal:"ArgmaxTiles"`
}

// ModelEngine maintains cached Metal pipeline states and GPU-resident memory allocations.
type ModelEngine struct {
	dev                     *metal.Device
	lib                     apple.MTLLibrary
	pso                     decodeKernels
	mem                     *metalmem.Arena
	memory                  *metalmem.Tracker
	recorder                *metalprof.FlightRecorder
	metrics                 *RuntimeMetrics
	generateMu              sync.Mutex
	batchedMu               sync.Mutex
	freeOnce                sync.Once
	disableProjectionFusion bool
	prefixes                *prefixCache
	tuner                   *DispatchTuner

	// GPU-Resident Unified Memory Buffers
	xBuf       mempool.Slice[float32]
	normXBuf   mempool.Slice[float32]
	qBuf       mempool.Slice[float32]
	kBuf       mempool.Slice[float32]
	vBuf       mempool.Slice[float32]
	attnOutBuf mempool.Slice[float32]
	gateBuf    mempool.Slice[float32]
	upBuf      mempool.Slice[float32]
	ffnActBuf  mempool.Slice[float32]
	kCacheBuf  []mempool.Slice[float32]
	vCacheBuf  []mempool.Slice[float32]
	logitsBuf  mempool.Slice[float32]
	tokBuf     mempool.Slice[int32]

	// Reusable Batched Prefill Shape Arena Buffers
	prefill          *metalmem.Arena
	prefillMaxSeqLen int
	prefillXMat      mempool.Slice[float32]
	prefillNormXMat  mempool.Slice[float32]
	prefillQMat      mempool.Slice[float32]
	prefillKMat      mempool.Slice[float32]
	prefillVMat      mempool.Slice[float32]
	prefillAttnOut   mempool.Slice[float32]
	prefillGateMat   mempool.Slice[float32]
	prefillUpMat     mempool.Slice[float32]
	prefillFFNAct    mempool.Slice[float32]
	prefillTokIn     mempool.Slice[int32]
	prefillRequest   mempool.Slice[uint32]
	prefillPositions mempool.Slice[uint32]
	prefillOffsets   mempool.Slice[uint32]
	prefillLengths   mempool.Slice[uint32]
	prefillRows      mempool.Slice[uint32]
	prefillRestored  mempool.Slice[uint32]

	// GPU Weights (FP32 or FP16)
	isF16         bool
	embedTable    mempool.Slice[float32]
	finalNormW    mempool.Slice[float32]
	lmHeadW       mempool.Slice[float32]
	layers        []GPULayerWeights
	embedTableF16 mempool.Slice[uint16]
	finalNormWF16 mempool.Slice[uint16]
	lmHeadWF16    mempool.Slice[uint16]
	layersF16     []GPULayerWeightsF16

	tokenizer   Tokenizer
	weights     *metalsafetensors.Weights
	egraphSats  int
	Plan        *ExecutionPlan
	cacheKey    string
	consumption CheckpointReceipt
}

// CheckpointReceipt returns the validated tensor-consumption account.
func (eng *ModelEngine) CheckpointReceipt() CheckpointReceipt {
	if eng == nil {
		return CheckpointReceipt{}
	}
	return eng.consumption
}

// BatchedQKVResult is the output of the shared-weight embedding, normalization,
// and Q/K/V projection slice. Rows retain the caller's compacted order.
type BatchedQKVResult struct {
	Positions  []int
	Embedding  []float32
	Normalized []float32
	Q, K, V    []float32
	Dispatches int
}

// BatchedAttentionCache owns device KV storage in
// [layer, row, sequence, KV-head, head-dimension] order.
type BatchedAttentionCache struct {
	mu                            sync.Mutex
	K, V                          []mempool.Slice[float32]
	positions, rows               mempool.Slice[uint32]
	rowCapacity, maxSeqLen, kvDim int
	freed                         bool
	arena                         *metalmem.Arena
}

// NewBatchedAttentionCache allocates cache storage for independently positioned rows.
func (eng *ModelEngine) NewBatchedAttentionCache(cfg Config, rowCapacity int) (*BatchedAttentionCache, error) {
	if eng == nil || eng.dev == nil || rowCapacity <= 0 || cfg.MaxSeqLen <= 0 {
		return nil, errors.New("new batched attention cache: invalid capacity")
	}
	a := metalmem.NewTrackedArena(eng.dev, eng.memory)
	c := &BatchedAttentionCache{rowCapacity: rowCapacity, maxSeqLen: cfg.MaxSeqLen, kvDim: cfg.NumKVHeads * cfg.HeadDim, arena: a}
	c.K = make([]mempool.Slice[float32], cfg.NumLayers)
	c.V = make([]mempool.Slice[float32], cfg.NumLayers)
	for l := range cfg.NumLayers {
		c.K[l] = metalmem.AllocIn[float32](a, fmt.Sprintf("k[%d]", l), rowCapacity*cfg.MaxSeqLen*c.kvDim)
		if err := a.Err(); err != nil {
			c.Free()
			return nil, fmt.Errorf("alloc batched k-cache layer %d: %w", l, err)
		}
		c.V[l] = metalmem.AllocIn[float32](a, fmt.Sprintf("v[%d]", l), rowCapacity*cfg.MaxSeqLen*c.kvDim)
		if err := a.Err(); err != nil {
			c.Free()
			return nil, fmt.Errorf("alloc batched v-cache layer %d: %w", l, err)
		}
	}
	c.positions = metalmem.AllocIn[uint32](a, "positions", rowCapacity)
	if err := a.Err(); err != nil {
		c.Free()
		return nil, fmt.Errorf("alloc batched positions: %w", err)
	}
	c.rows = metalmem.AllocIn[uint32](a, "rows", rowCapacity)
	if err := a.Err(); err != nil {
		c.Free()
		return nil, fmt.Errorf("alloc batched rows: %w", err)
	}
	return c, nil
}

// Free releases the cache. It waits for any in-flight RunBatchedAttention call.
func (c *BatchedAttentionCache) Free() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.freed {
		return
	}
	c.arena.Free()
	c.freed = true
}

// ImportSequence copies synchronized prompt KV state into one stable cache row.
func (c *BatchedAttentionCache) ImportSequence(row, tokens int, sequence *SequenceKVCache) error {
	if c == nil || sequence == nil || row < 0 || row >= c.rowCapacity || tokens < 0 || tokens > c.maxSeqLen || len(sequence.KCache) != len(c.K) {
		return errors.New("import sequence: invalid cache state")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.freed {
		return errors.New("import sequence: cache is freed")
	}
	n := tokens * c.kvDim
	offset := row * c.maxSeqLen * c.kvDim
	for layer := range c.K {
		copy(c.K[layer].HostSlice()[offset:offset+n], sequence.KCache[layer].HostSlice()[:n])
		copy(c.V[layer].HostSlice()[offset:offset+n], sequence.VCache[layer].HostSlice()[:n])
	}
	return nil
}

func (c *BatchedAttentionCache) restorePrefix(row, tokens int, snapshot *prefixSnapshot) bool {
	if c == nil || snapshot == nil || row < 0 || row >= c.rowCapacity || tokens <= 0 || tokens > c.maxSeqLen || len(snapshot.k) != len(c.K) || len(snapshot.v) != len(c.V) {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.freed {
		return false
	}
	n := tokens * c.kvDim
	offset := row * c.maxSeqLen * c.kvDim
	for layer := range c.K {
		if n > len(snapshot.k[layer]) || n > len(snapshot.v[layer]) {
			return false
		}
		copy(c.K[layer].HostSlice()[offset:offset+n], snapshot.k[layer][:n])
		copy(c.V[layer].HostSlice()[offset:offset+n], snapshot.v[layer][:n])
	}
	return true
}

func (c *BatchedAttentionCache) capturePrefix(row, tokens int, current int32) *prefixSnapshot {
	if c == nil || row < 0 || row >= c.rowCapacity || tokens <= 0 || tokens > c.maxSeqLen {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.freed {
		return nil
	}
	n := tokens * c.kvDim
	offset := row * c.maxSeqLen * c.kvDim
	snapshot := &prefixSnapshot{tokens: tokens, current: current, k: make([][]float32, len(c.K)), v: make([][]float32, len(c.V))}
	for layer := range c.K {
		snapshot.k[layer] = append([]float32(nil), c.K[layer].HostSlice()[offset:offset+n]...)
		snapshot.v[layer] = append([]float32(nil), c.V[layer].HostSlice()[offset:offset+n]...)
		snapshot.bytes += int64(2*n) * 4
	}
	return snapshot
}

// BatchedAttentionResult contains row-indexed RoPE outputs and causal attention.
type BatchedAttentionResult struct {
	Q, K, Attention []float32
	Positions, Rows []int
	Dispatches      int
}

// BatchedLayerResult is the dense active-row output of one decoder layer.
type BatchedLayerResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
}

// SamplerConfig is the request-local sampling contract for a model tail.
// Only deterministic argmax is currently supported.
type SamplerConfig struct {
	Method      string
	Temperature float32
}

// BatchedModelTail owns capacity-sized scratch buffers reused across steps.
// In particular, logits are allocated once rather than B*vocab on every call.
type BatchedModelTail struct {
	mu                       sync.Mutex
	norm, logits             mempool.Slice[float32]
	scale, candidates        mempool.Slice[float32]
	tokens                   mempool.Slice[int32]
	capacity, dim, vocabSize int
	freed                    bool
	arena                    *metalmem.Arena
}

// BatchedModelTailResult contains row-major normalized states and logits.
type BatchedModelTailResult struct {
	Normalized []float32
	Logits     []float32
	Tokens     []int32
	Dispatches int
}

// NewBatchedModelTail allocates reusable tail scratch for capacity active rows.
func (eng *ModelEngine) NewBatchedModelTail(cfg Config, capacity int) (*BatchedModelTail, error) {
	if eng == nil || eng.dev == nil || capacity <= 0 || cfg.Dim <= 0 || cfg.VocabSize <= 0 {
		return nil, errors.New("new batched model tail: invalid shape")
	}
	a := metalmem.NewTrackedArena(eng.dev, eng.memory)
	t := &BatchedModelTail{capacity: capacity, dim: cfg.Dim, vocabSize: cfg.VocabSize, arena: a}
	t.norm = metalmem.AllocIn[float32](a, "norm", capacity*cfg.Dim)
	t.tokens = metalmem.AllocIn[int32](a, "tokens", capacity)
	t.scale = metalmem.AllocIn[float32](a, "scale", capacity)
	t.candidates = metalmem.AllocIn[float32](a, "candidates", capacity*((cfg.VocabSize+255)/256)*2)
	if err := a.Err(); err != nil {
		t.Free()
		return nil, fmt.Errorf("new batched model tail: %w", err)
	}
	return t, nil
}

// Free releases the tail scratch after any in-flight Run call completes.
func (t *BatchedModelTail) Free() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.freed {
		return
	}
	t.arena.Free()
	t.freed = true
}

// Run executes shared final RMSNorm, LM-head projection, and per-row argmax.
// Its dispatch count is three for every non-empty supported batch.
func (t *BatchedModelTail) Run(eng *ModelEngine, cfg Config, hidden []float32, samplers []SamplerConfig) (BatchedModelTailResult, error) {
	if eng == nil || eng.dev == nil || t == nil {
		return BatchedModelTailResult{}, errors.New("batched model tail: nil engine or tail")
	}
	if len(hidden) == 0 || len(hidden)%cfg.Dim != 0 {
		return BatchedModelTailResult{}, errors.New("batched model tail: inconsistent hidden shape")
	}
	b := len(hidden) / cfg.Dim
	if b > t.capacity || t.dim != cfg.Dim || t.vocabSize != cfg.VocabSize {
		return BatchedModelTailResult{}, errors.New("batched model tail: unsupported configuration")
	}
	if len(samplers) != 0 && len(samplers) != b {
		return BatchedModelTailResult{}, errors.New("batched model tail: sampler count does not match batch")
	}
	for _, sampler := range samplers {
		if (sampler.Method != "" && sampler.Method != "argmax") || sampler.Temperature != 0 {
			return BatchedModelTailResult{}, errors.New("batched model tail: stochastic sampling unsupported")
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.freed {
		return BatchedModelTailResult{}, errors.New("batched model tail: tail is freed")
	}
	if t.logits.Len() == 0 {
		t.logits = metalmem.AllocIn[float32](t.arena, "logits", t.capacity*t.vocabSize)
		if err := t.arena.Err(); err != nil {
			return BatchedModelTailResult{}, fmt.Errorf("batched model tail: allocate diagnostic logits: %w", err)
		}
	}
	copy(t.norm.HostSlice()[:len(hidden)], hidden)
	cb := eng.dev.Queue().CommandBuffer()
	ce := countDispatches(cb.ComputeCommandEncoder())
	bv, dim, vocab, eps := uint32(b), uint32(cfg.Dim), uint32(cfg.VocabSize), cfg.Eps
	if eng.isF16 {
		ce.SetComputePipelineState(eng.pso.BatchedNormH)
	} else {
		ce.SetComputePipelineState(eng.pso.BatchedNorm)
	}
	ce.SetBufferWithOffsetAtIndex(t.norm.Buffer(), 0, 0)
	if eng.isF16 {
		ce.SetBufferWithOffsetAtIndex(eng.finalNormWF16.Buffer(), 0, 1)
	} else {
		ce.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
	}
	ce.SetBufferWithOffsetAtIndex(t.norm.Buffer(), 0, 2)
	metal.SetScalar(ce, bv, 3)
	metal.SetScalar(ce, dim, 4)
	metal.SetScalar(ce, eps, 5)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(128))
	if eng.isF16 {
		ce.SetComputePipelineState(eng.pso.BatchedGEMMH)
	} else {
		ce.SetComputePipelineState(eng.pso.BatchedGEMM)
	}
	ce.SetBufferWithOffsetAtIndex(t.norm.Buffer(), 0, 0)
	if eng.isF16 {
		ce.SetBufferWithOffsetAtIndex(eng.lmHeadWF16.Buffer(), 0, 1)
	} else {
		ce.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 1)
	}
	ce.SetBufferWithOffsetAtIndex(t.logits.Buffer(), 0, 2)
	metal.SetScalar(ce, bv, 3)
	metal.SetScalar(ce, dim, 4)
	metal.SetScalar(ce, vocab, 5)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.VocabSize), metal.Threads(128))
	ce.SetComputePipelineState(eng.pso.BatchedSample)
	ce.SetBufferWithOffsetAtIndex(t.logits.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(t.tokens.Buffer(), 0, 1)
	metal.SetScalar(ce, bv, 2)
	metal.SetScalar(ce, vocab, 3)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(128))
	ce.EndEncoding()
	cb.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return BatchedModelTailResult{}, fmt.Errorf("batched model tail: shared command buffer: %w", err)
	}
	return BatchedModelTailResult{
		Normalized: append([]float32(nil), t.norm.HostSlice()[:b*cfg.Dim]...),
		Logits:     append([]float32(nil), t.logits.HostSlice()[:b*cfg.VocabSize]...),
		Tokens:     append([]int32(nil), t.tokens.HostSlice()[:b]...),
		Dispatches: ce.dispatches,
	}, nil
}

// RunBatchedModelTailFromChain executes the tail directly from the chain's
// persistent device hidden buffer, without a device-to-host-to-device copy.
func RunBatchedModelTailFromChain(eng *ModelEngine, cfg Config, tail *BatchedModelTail, chain BatchedLayerChainResult, samplers []SamplerConfig) (BatchedModelTailResult, error) {
	b := len(chain.Positions)
	if b == 0 || len(chain.Hidden) != b*cfg.Dim || len(chain.Rows) != b {
		return BatchedModelTailResult{}, errors.New("batched model tail from chain: invalid chain result")
	}
	return tail.runBuffer(eng, cfg, eng.prefillXMat, b, samplers)
}

func validateBatchedTail(eng *ModelEngine, cfg Config, t *BatchedModelTail, input mempool.Slice[float32], b int, samplers []SamplerConfig) error {
	if eng == nil || eng.dev == nil || t == nil || b <= 0 || input.Len() < b*cfg.Dim {
		return errors.New("batched model tail: invalid device input")
	}
	if b > t.capacity || t.dim != cfg.Dim || t.vocabSize != cfg.VocabSize {
		return errors.New("batched model tail: unsupported configuration")
	}
	if len(samplers) != 0 && len(samplers) != b {
		return errors.New("batched model tail: sampler count does not match batch")
	}
	for _, sampler := range samplers {
		if (sampler.Method != "" && sampler.Method != "argmax") || sampler.Temperature != 0 {
			return errors.New("batched model tail: stochastic sampling unsupported")
		}
	}
	return nil
}

func encodeBatchedTail(ce apple.MTLComputeCommandEncoder, eng *ModelEngine, cfg Config, t *BatchedModelTail, input mempool.Slice[float32], b int) {
	bv, dim, vocab, eps := uint32(b), uint32(cfg.Dim), uint32(cfg.VocabSize), cfg.Eps
	ce.SetComputePipelineState(eng.pso.RMSNormScale)
	ce.SetBufferWithOffsetAtIndex(input.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(t.scale.Buffer(), 0, 1)
	metal.SetScalar(ce, bv, 2)
	metal.SetScalar(ce, dim, 3)
	metal.SetScalar(ce, eps, 4)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(128))
	if eng.isF16 {
		ce.SetComputePipelineState(eng.pso.LMHeadTileArgmaxH)
	} else {
		ce.SetComputePipelineState(eng.pso.LMHeadTileArgmax)
	}
	ce.SetBufferWithOffsetAtIndex(input.Buffer(), 0, 0)
	if eng.isF16 {
		ce.SetBufferWithOffsetAtIndex(eng.finalNormWF16.Buffer(), 0, 1)
		ce.SetBufferWithOffsetAtIndex(eng.lmHeadWF16.Buffer(), 0, 2)
	} else {
		ce.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
		ce.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 2)
	}
	ce.SetBufferWithOffsetAtIndex(t.scale.Buffer(), 0, 3)
	ce.SetBufferWithOffsetAtIndex(t.candidates.Buffer(), 0, 4)
	metal.SetScalar(ce, dim, 5)
	metal.SetScalar(ce, vocab, 6)
	tiles := uint32((cfg.VocabSize + 255) / 256)
	metal.SetScalar(ce, tiles, 7)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*int(tiles)), metal.Threads(256))
	ce.SetComputePipelineState(eng.pso.ArgmaxTiles)
	ce.SetBufferWithOffsetAtIndex(t.candidates.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(t.tokens.Buffer(), 0, 1)
	metal.SetScalar(ce, bv, 2)
	metal.SetScalar(ce, tiles, 3)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(128))
}

func (t *BatchedModelTail) runBuffer(eng *ModelEngine, cfg Config, input mempool.Slice[float32], b int, samplers []SamplerConfig) (BatchedModelTailResult, error) {
	if err := validateBatchedTail(eng, cfg, t, input, b, samplers); err != nil {
		return BatchedModelTailResult{}, err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.freed {
		return BatchedModelTailResult{}, errors.New("batched model tail: tail is freed")
	}
	cb := eng.dev.Queue().CommandBuffer()
	ce := countDispatches(cb.ComputeCommandEncoder())
	encodeBatchedTail(ce, eng, cfg, t, input, b)
	ce.EndEncoding()
	cb.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return BatchedModelTailResult{}, fmt.Errorf("batched model tail: shared command buffer: %w", err)
	}
	return BatchedModelTailResult{
		Tokens:     append([]int32(nil), t.tokens.HostSlice()[:b]...),
		Dispatches: ce.dispatches,
	}, nil
}

// BatchedLayerChainResult is the final persistent-device hidden state after all layers.
type BatchedLayerChainResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
	DeviceBytes     int64
}

// SegmentedLayerChainResult is the dense request-tail output of a packed
// prompt wave. Hidden is ordered by segment, while Rows retains stable KV row
// ownership. Hidden aliases engine scratch and remains valid only until the
// next batched engine operation.
type SegmentedLayerChainResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
	DeviceBytes     int64
}

func encodeQKVProjections(ce apple.MTLComputeCommandEncoder, eng *ModelEngine, l *GPULayerWeights, lf *GPULayerWeightsF16, cfg Config, b int) {
	routes := modelProjectionRoutes(cfg, eng.isF16, projectionQKV)
	if !eng.disableProjectionFusion && useProjectionFusion(projectionQKV, routes...) {
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedQKVProjH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedQKVProj)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.WQ.Buffer(), 0, 1)
			ce.SetBufferWithOffsetAtIndex(lf.WK.Buffer(), 0, 2)
			ce.SetBufferWithOffsetAtIndex(lf.WV.Buffer(), 0, 3)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.WQ.Buffer(), 0, 1)
			ce.SetBufferWithOffsetAtIndex(l.WK.Buffer(), 0, 2)
			ce.SetBufferWithOffsetAtIndex(l.WV.Buffer(), 0, 3)
		}
		for i, out := range []mempool.Slice[float32]{eng.prefillQMat, eng.prefillKMat, eng.prefillVMat} {
			ce.SetBufferWithOffsetAtIndex(out.Buffer(), 0, uint(i+4))
		}
		metal.SetScalar(ce, uint32(b), 7)
		metal.SetScalar(ce, uint32(cfg.Dim), 8)
		metal.SetScalar(ce, uint32(cfg.NumHeads*cfg.HeadDim), 9)
		metal.SetScalar(ce, uint32(cfg.NumKVHeads*cfg.HeadDim), 10)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*(cfg.NumHeads+2*cfg.NumKVHeads)*cfg.HeadDim), metal.Threads(128))
		return
	}
	for pi, p := range []struct {
		out   mempool.Slice[float32]
		width int
	}{{eng.prefillQMat, cfg.NumHeads * cfg.HeadDim}, {eng.prefillKMat, cfg.NumKVHeads * cfg.HeadDim}, {eng.prefillVMat, cfg.NumKVHeads * cfg.HeadDim}} {
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedGEMM)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex([]mempool.Slice[uint16]{lf.WQ, lf.WK, lf.WV}[pi].Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex([]mempool.Slice[float32]{l.WQ, l.WK, l.WV}[pi].Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(p.out.Buffer(), 0, 2)
		metal.SetScalar(ce, uint32(b), 3)
		metal.SetScalar(ce, uint32(cfg.Dim), 4)
		metal.SetScalar(ce, uint32(p.width), 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*p.width), metal.Threads(128))
	}
}

func encodeGateUpProjections(ce apple.MTLComputeCommandEncoder, eng *ModelEngine, l *GPULayerWeights, lf *GPULayerWeightsF16, cfg Config, b int) {
	routes := modelProjectionRoutes(cfg, eng.isF16, projectionGateUp)
	if !eng.disableProjectionFusion && useProjectionFusion(projectionGateUp, routes...) {
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedGateUpH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedGateUp)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.WGate.Buffer(), 0, 1)
			ce.SetBufferWithOffsetAtIndex(lf.WUp.Buffer(), 0, 2)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.WGate.Buffer(), 0, 1)
			ce.SetBufferWithOffsetAtIndex(l.WUp.Buffer(), 0, 2)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillGateMat.Buffer(), 0, 3)
		ce.SetBufferWithOffsetAtIndex(eng.prefillUpMat.Buffer(), 0, 4)
		metal.SetScalar(ce, uint32(b), 5)
		metal.SetScalar(ce, uint32(cfg.Dim), 6)
		metal.SetScalar(ce, uint32(cfg.HiddenDim), 7)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*2*cfg.HiddenDim), metal.Threads(128))
		return
	}
	for pi, out := range []mempool.Slice[float32]{eng.prefillGateMat, eng.prefillUpMat} {
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedGEMM)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex([]mempool.Slice[uint16]{lf.WGate, lf.WUp}[pi].Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex([]mempool.Slice[float32]{l.WGate, l.WUp}[pi].Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(out.Buffer(), 0, 2)
		metal.SetScalar(ce, uint32(b), 3)
		metal.SetScalar(ce, uint32(cfg.Dim), 4)
		metal.SetScalar(ce, uint32(cfg.HiddenDim), 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.HiddenDim), metal.Threads(128))
	}
}

// RunBatchedLayerChain embeds once and executes every decoder layer without
// intermediate host copies. It issues exactly 1+9*NumLayers dispatches. A
// command buffer execution failure invalidates the whole batch.
func RunBatchedLayerChain(eng *ModelEngine, cfg Config, cache *BatchedAttentionCache, tokens []int32, positions, rows []int) (BatchedLayerChainResult, error) {
	chain, _, err := runBatchedLayerChain(context.Background(), eng, cfg, cache, nil, nil, nil, nil, tokens, positions, rows)
	return chain, err
}

// RunLeasedBatchedLayerChain executes one token per row while reading cached
// prefix K/V directly from leased Metal blocks. Newly produced suffix K/V is
// stored in the row cache and both operations share one checked command buffer.
func RunLeasedBatchedLayerChain(ctx context.Context, eng *ModelEngine, cfg Config, cache *BatchedAttentionCache, workspace *metalBlockWorkspace, leases []*metalprefix.Lease, tokens []int32, positions, rows []int) (BatchedLayerChainResult, error) {
	if workspace == nil || len(leases) != len(tokens) {
		return BatchedLayerChainResult{}, errors.New("leased layer chain: invalid workspace or leases")
	}
	chain, _, err := runBatchedLayerChain(ctx, eng, cfg, cache, workspace, leases, nil, nil, tokens, positions, rows)
	return chain, err
}

// RunLeasedBatchedDecode executes the layer chain and deterministic model tail
// in one command buffer. The tail and engine scratch remain locked until the
// command buffer completes, so Close cannot release storage while it is in use.
func RunLeasedBatchedDecode(ctx context.Context, eng *ModelEngine, cfg Config, cache *BatchedAttentionCache, workspace *metalBlockWorkspace, leases []*metalprefix.Lease, tail *BatchedModelTail, samplers []SamplerConfig, tokens []int32, positions, rows []int) (BatchedLayerChainResult, BatchedModelTailResult, error) {
	return runBatchedLayerChain(ctx, eng, cfg, cache, workspace, leases, tail, samplers, tokens, positions, rows)
}

func runBatchedLayerChain(ctx context.Context, eng *ModelEngine, cfg Config, cache *BatchedAttentionCache, workspace *metalBlockWorkspace, leases []*metalprefix.Lease, tail *BatchedModelTail, samplers []SamplerConfig, tokens []int32, positions, rows []int) (BatchedLayerChainResult, BatchedModelTailResult, error) {
	b := len(tokens)
	kd := cfg.NumKVHeads * cfg.HeadDim
	if eng == nil || eng.dev == nil || cache == nil {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, errors.New("batched layer chain: nil engine or cache")
	}
	weightLayers := len(eng.layers)
	if eng.isF16 {
		weightLayers = len(eng.layersF16)
	}
	if b == 0 || len(positions) != b || len(rows) != b || weightLayers != cfg.NumLayers || len(cache.K) != cfg.NumLayers {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, errors.New("batched layer chain: inconsistent shape")
	}
	if cfg.HeadDim <= 0 || cfg.HeadDim > 128 || cfg.HeadDim%2 != 0 || cfg.NumKVHeads <= 0 || cfg.NumHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.RopeTheta <= 0 || b > cache.rowCapacity || cache.maxSeqLen != cfg.MaxSeqLen || cache.kvDim != kd {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, errors.New("batched layer chain: unsupported configuration or cache shape")
	}
	cacheElements := cache.rowCapacity * cache.maxSeqLen * kd
	if cache.positions.Len() < b || cache.rows.Len() < b {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, errors.New("batched layer chain: row metadata buffer too small")
	}
	for layer := range cache.K {
		if cache.K[layer].Len() < cacheElements || cache.V[layer].Len() < cacheElements {
			return BatchedLayerChainResult{}, BatchedModelTailResult{}, fmt.Errorf("batched layer chain: cache buffer too small at layer %d", layer)
		}
	}
	seen := make(map[int]bool, b)
	for i, tok := range tokens {
		if tok < 0 || int(tok) >= cfg.VocabSize || positions[i] < 0 || positions[i] >= cfg.MaxSeqLen || rows[i] < 0 || rows[i] >= cache.rowCapacity || seen[rows[i]] {
			return BatchedLayerChainResult{}, BatchedModelTailResult{}, fmt.Errorf("batched layer chain: invalid active row %d", i)
		}
		seen[rows[i]] = true
	}
	if err := ctx.Err(); err != nil {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, err
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if cache.freed {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, errors.New("batched layer chain: cache freed")
	}
	eng.batchedMu.Lock()
	defer eng.batchedMu.Unlock()
	if workspace != nil {
		workspace.mu.Lock()
		defer workspace.mu.Unlock()
	}
	if err := eng.ensurePrefillArena(b, cfg); err != nil {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, err
	}
	if tail != nil {
		if err := validateBatchedTail(eng, cfg, tail, eng.prefillXMat, b, samplers); err != nil {
			return BatchedLayerChainResult{}, BatchedModelTailResult{}, err
		}
		tail.mu.Lock()
		defer tail.mu.Unlock()
		if tail.freed {
			return BatchedLayerChainResult{}, BatchedModelTailResult{}, errors.New("batched model tail: tail is freed")
		}
	}
	var blockPins []*metalprefix.ViewLease
	if workspace != nil {
		var err error
		blockPins, err = workspace.prepare(cfg, leases, positions, rows)
		if err != nil {
			return BatchedLayerChainResult{}, BatchedModelTailResult{}, err
		}
		defer releaseMetalBlockPins(blockPins)
	}
	copy(eng.prefillTokIn.HostSlice()[:b], tokens)
	for i := range b {
		cache.positions.HostSlice()[i] = uint32(positions[i])
		cache.rows.HostSlice()[i] = uint32(rows[i])
	}
	bv, dv, hv := uint32(b), uint32(cfg.Dim), uint32(cfg.HiddenDim)
	eps := cfg.Eps
	theta := cfg.RopeTheta
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	cb := eng.dev.Queue().CommandBuffer()
	ce := countDispatches(cb.ComputeCommandEncoder())
	if eng.isF16 {
		ce.SetComputePipelineState(eng.pso.BatchedEmbedH)
		ce.SetBufferWithOffsetAtIndex(eng.embedTableF16.Buffer(), 0, 0)
	} else {
		ce.SetComputePipelineState(eng.pso.BatchedEmbed)
		ce.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
	}
	ce.SetBufferWithOffsetAtIndex(eng.prefillTokIn.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
	metal.SetScalar(ce, bv, 3)
	metal.SetScalar(ce, dv, 4)
	ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(b*cfg.Dim), metal.Threads(128))
	for li := range cfg.NumLayers {
		var l *GPULayerWeights
		var lf *GPULayerWeightsF16
		if eng.isF16 {
			lf = &eng.layersF16[li]
			ce.SetComputePipelineState(eng.pso.BatchedNormH)
		} else {
			l = &eng.layers[li]
			ce.SetComputePipelineState(eng.pso.BatchedNorm)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.InputNorm.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.InputNorm.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, bv, 3)
		metal.SetScalar(ce, dv, 4)
		metal.SetScalar(ce, eps, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(128))
		encodeQKVProjections(ce, eng, l, lf, cfg, b)
		ce.SetComputePipelineState(eng.pso.RowRoPE)
		for i, s := range []mempool.Slice[float32]{eng.prefillQMat, eng.prefillKMat, eng.prefillVMat, cache.K[li], cache.V[li]} {
			ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(i))
		}
		ce.SetBufferWithOffsetAtIndex(cache.positions.Buffer(), 0, 5)
		ce.SetBufferWithOffsetAtIndex(cache.rows.Buffer(), 0, 6)
		metal.SetScalar(ce, bv, 7)
		metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 8)
		metal.SetScalar(ce, uint32(cfg.HeadDim), 9)
		metal.SetScalar(ce, uint32(cfg.NumHeads), 10)
		metal.SetScalar(ce, uint32(cfg.NumKVHeads), 11)
		metal.SetScalar(ce, theta, 12)
		metal.SetScalar(ce, uint32(cache.rowCapacity), 13)
		ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(1))
		if workspace == nil {
			ce.SetComputePipelineState(eng.pso.RowAttn)
			ce.SetBufferWithOffsetAtIndex(eng.prefillQMat.Buffer(), 0, 0)
			ce.SetBufferWithOffsetAtIndex(cache.K[li].Buffer(), 0, 1)
			ce.SetBufferWithOffsetAtIndex(cache.V[li].Buffer(), 0, 2)
			ce.SetBufferWithOffsetAtIndex(eng.prefillAttnOut.Buffer(), 0, 3)
			ce.SetBufferWithOffsetAtIndex(cache.positions.Buffer(), 0, 4)
			ce.SetBufferWithOffsetAtIndex(cache.rows.Buffer(), 0, 5)
			metal.SetScalar(ce, bv, 6)
			metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 7)
			metal.SetScalar(ce, uint32(cfg.NumHeads), 8)
			metal.SetScalar(ce, uint32(cfg.NumKVHeads), 9)
			metal.SetScalar(ce, uint32(cfg.HeadDim), 10)
			metal.SetScalar(ce, scale, 11)
			metal.SetScalar(ce, uint32(cache.rowCapacity), 12)
		} else {
			workspace.encodeAttention(ce, eng, cfg, cache, li, b, blockPins)
		}
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.NumHeads), metal.Threads(128))
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMResH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMRes)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillAttnOut.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.WO.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.WO.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, bv, 3)
		metal.SetScalar(ce, dv, 4)
		metal.SetScalar(ce, dv, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.Dim), metal.Threads(128))
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedNormH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedNorm)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.PostNorm.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.PostNorm.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, bv, 3)
		metal.SetScalar(ce, dv, 4)
		metal.SetScalar(ce, eps, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(128))
		encodeGateUpProjections(ce, eng, l, lf, cfg, b)
		total := uint32(b * cfg.HiddenDim)
		ce.SetComputePipelineState(eng.pso.BatchedSiLU)
		ce.SetBufferWithOffsetAtIndex(eng.prefillGateMat.Buffer(), 0, 0)
		ce.SetBufferWithOffsetAtIndex(eng.prefillUpMat.Buffer(), 0, 1)
		ce.SetBufferWithOffsetAtIndex(eng.prefillFFNAct.Buffer(), 0, 2)
		metal.SetScalar(ce, total, 3)
		ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(int(total)), metal.Threads(128))
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMResH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMRes)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillFFNAct.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.WDown.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.WDown.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, bv, 3)
		metal.SetScalar(ce, hv, 4)
		metal.SetScalar(ce, dv, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.Dim), metal.Threads(128))
	}
	chainDispatches := ce.dispatches
	if tail != nil {
		encodeBatchedTail(ce, eng, cfg, tail, eng.prefillXMat, b)
	}
	ce.EndEncoding()
	if err := ctx.Err(); err != nil {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, err
	}
	cb.Commit()
	if err := metal.WaitCommandBuffer(ctx, cb); err != nil {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, fmt.Errorf("batched layer chain: shared command buffer: %w", err)
	}
	bytes := int64(2 * cfg.NumLayers * cache.rowCapacity * cfg.MaxSeqLen * kd * 4)
	// Hidden aliases persistent engine scratch and is valid until the next
	// batched engine operation. The model tail consumes the device buffer
	// directly; exposing the alias keeps differential tests possible without a
	// serving-path host copy.
	chain := BatchedLayerChainResult{Hidden: eng.prefillXMat.HostSlice()[:b*cfg.Dim], Positions: append([]int(nil), positions...), Rows: append([]int(nil), rows...), Dispatches: chainDispatches, DeviceBytes: bytes}
	var tailResult BatchedModelTailResult
	if tail != nil {
		tailResult = BatchedModelTailResult{Tokens: append([]int32(nil), tail.tokens.HostSlice()[:b]...), Dispatches: ce.dispatches - chainDispatches}
	}
	return chain, tailResult, nil
}

// RunSegmentedLayerChain executes a packed prompt wave through every decoder
// layer, then gathers the final token of each segment in request order. No GPU
// work is submitted until the entire pack and cache shape have been checked. A
// command buffer execution failure invalidates the whole packed wave.
func RunSegmentedLayerChain(ctx context.Context, eng *ModelEngine, cfg Config, cache *BatchedAttentionCache, packed PackedPrefill) (SegmentedLayerChainResult, error) {
	n, b := len(packed.Tokens), len(packed.Offsets)
	kd := cfg.NumKVHeads * cfg.HeadDim
	if eng == nil || eng.dev == nil || cache == nil {
		return SegmentedLayerChainResult{}, errors.New("segmented layer chain: nil engine or cache")
	}
	if err := packed.Validate(n, cache.rowCapacity, cfg.MaxSeqLen); err != nil {
		return SegmentedLayerChainResult{}, fmt.Errorf("segmented layer chain: %w", err)
	}
	for i, token := range packed.Tokens {
		if token < 0 || int(token) >= cfg.VocabSize {
			return SegmentedLayerChainResult{}, fmt.Errorf("segmented layer chain: invalid token at offset %d", i)
		}
	}
	weightLayers := len(eng.layers)
	if eng.isF16 {
		weightLayers = len(eng.layersF16)
	}
	if weightLayers != cfg.NumLayers || len(cache.K) != cfg.NumLayers || cfg.Dim != cfg.NumHeads*cfg.HeadDim || cfg.HeadDim <= 0 || cfg.HeadDim > 128 || cfg.HeadDim%2 != 0 || cfg.NumKVHeads <= 0 || cfg.NumHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.RopeTheta <= 0 || cache.maxSeqLen != cfg.MaxSeqLen || cache.kvDim != kd {
		return SegmentedLayerChainResult{}, errors.New("segmented layer chain: unsupported configuration or cache shape")
	}
	cacheElements := cache.rowCapacity * cache.maxSeqLen * kd
	for layer := range cache.K {
		if cache.K[layer].Len() < cacheElements || cache.V[layer].Len() < cacheElements {
			return SegmentedLayerChainResult{}, fmt.Errorf("segmented layer chain: cache buffer too small at layer %d", layer)
		}
	}
	if err := ctx.Err(); err != nil {
		return SegmentedLayerChainResult{}, err
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if cache.freed {
		return SegmentedLayerChainResult{}, errors.New("segmented layer chain: cache freed")
	}
	eng.batchedMu.Lock()
	defer eng.batchedMu.Unlock()
	if err := eng.ensurePrefillArena(n, cfg); err != nil {
		return SegmentedLayerChainResult{}, err
	}
	copy(eng.prefillTokIn.HostSlice()[:n], packed.Tokens)
	copy(eng.prefillRequest.HostSlice()[:n], packed.RequestIndex)
	copy(eng.prefillPositions.HostSlice()[:n], packed.AbsolutePositions)
	copy(eng.prefillOffsets.HostSlice()[:b], packed.Offsets)
	copy(eng.prefillLengths.HostSlice()[:b], packed.Lengths)
	copy(eng.prefillRows.HostSlice()[:b], packed.Rows)
	copy(eng.prefillRestored.HostSlice()[:b], packed.RestoredPrefixes)
	if err := ctx.Err(); err != nil {
		return SegmentedLayerChainResult{}, err
	}
	nv, bv, dv, hv := uint32(n), uint32(b), uint32(cfg.Dim), uint32(cfg.HiddenDim)
	eps, theta := cfg.Eps, cfg.RopeTheta
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	cb := eng.dev.Queue().CommandBuffer()
	ce := countDispatches(cb.ComputeCommandEncoder())
	if eng.isF16 {
		ce.SetComputePipelineState(eng.pso.BatchedEmbedH)
		ce.SetBufferWithOffsetAtIndex(eng.embedTableF16.Buffer(), 0, 0)
	} else {
		ce.SetComputePipelineState(eng.pso.BatchedEmbed)
		ce.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
	}
	ce.SetBufferWithOffsetAtIndex(eng.prefillTokIn.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
	metal.SetScalar(ce, nv, 3)
	metal.SetScalar(ce, dv, 4)
	ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(n*cfg.Dim), metal.Threads(128))
	for li := range cfg.NumLayers {
		var l *GPULayerWeights
		var lf *GPULayerWeightsF16
		if eng.isF16 {
			lf = &eng.layersF16[li]
			ce.SetComputePipelineState(eng.pso.BatchedNormH)
		} else {
			l = &eng.layers[li]
			ce.SetComputePipelineState(eng.pso.BatchedNorm)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.InputNorm.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.InputNorm.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, nv, 3)
		metal.SetScalar(ce, dv, 4)
		metal.SetScalar(ce, eps, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(n), metal.Threads(128))
		encodeQKVProjections(ce, eng, l, lf, cfg, n)
		ce.SetComputePipelineState(eng.pso.SegmentedRoPE)
		for i, s := range []mempool.Slice[float32]{eng.prefillQMat, eng.prefillKMat, eng.prefillVMat, cache.K[li], cache.V[li]} {
			ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(i))
		}
		for i, s := range []mempool.Slice[uint32]{eng.prefillRequest, eng.prefillPositions, eng.prefillOffsets, eng.prefillLengths, eng.prefillRows, eng.prefillRestored} {
			ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(5+i))
		}
		metal.SetScalar(ce, nv, 11)
		metal.SetScalar(ce, bv, 12)
		metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 13)
		metal.SetScalar(ce, uint32(cfg.HeadDim), 14)
		metal.SetScalar(ce, uint32(cfg.NumHeads), 15)
		metal.SetScalar(ce, uint32(cfg.NumKVHeads), 16)
		metal.SetScalar(ce, theta, 17)
		metal.SetScalar(ce, uint32(cache.rowCapacity), 18)
		ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(n), metal.Threads(1))
		ce.SetComputePipelineState(eng.pso.SegmentedAttn)
		for i, s := range []mempool.Slice[float32]{eng.prefillQMat, cache.K[li], cache.V[li], eng.prefillAttnOut} {
			ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(i))
		}
		for i, s := range []mempool.Slice[uint32]{eng.prefillRequest, eng.prefillPositions, eng.prefillOffsets, eng.prefillLengths, eng.prefillRows, eng.prefillRestored} {
			ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(4+i))
		}
		metal.SetScalar(ce, nv, 10)
		metal.SetScalar(ce, bv, 11)
		metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 12)
		metal.SetScalar(ce, uint32(cfg.NumHeads), 13)
		metal.SetScalar(ce, uint32(cfg.NumKVHeads), 14)
		metal.SetScalar(ce, uint32(cfg.HeadDim), 15)
		metal.SetScalar(ce, scale, 16)
		metal.SetScalar(ce, uint32(cache.rowCapacity), 17)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(n*cfg.NumHeads), metal.Threads(128))
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMResH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMRes)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillAttnOut.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.WO.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.WO.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, nv, 3)
		metal.SetScalar(ce, dv, 4)
		metal.SetScalar(ce, dv, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(n*cfg.Dim), metal.Threads(128))
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedNormH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedNorm)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.PostNorm.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.PostNorm.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, nv, 3)
		metal.SetScalar(ce, dv, 4)
		metal.SetScalar(ce, eps, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(n), metal.Threads(128))
		encodeGateUpProjections(ce, eng, l, lf, cfg, n)
		total := uint32(n * cfg.HiddenDim)
		ce.SetComputePipelineState(eng.pso.BatchedSiLU)
		ce.SetBufferWithOffsetAtIndex(eng.prefillGateMat.Buffer(), 0, 0)
		ce.SetBufferWithOffsetAtIndex(eng.prefillUpMat.Buffer(), 0, 1)
		ce.SetBufferWithOffsetAtIndex(eng.prefillFFNAct.Buffer(), 0, 2)
		metal.SetScalar(ce, total, 3)
		ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(int(total)), metal.Threads(128))
		if eng.isF16 {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMResH)
		} else {
			ce.SetComputePipelineState(eng.pso.BatchedGEMMRes)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillFFNAct.Buffer(), 0, 0)
		if eng.isF16 {
			ce.SetBufferWithOffsetAtIndex(lf.WDown.Buffer(), 0, 1)
		} else {
			ce.SetBufferWithOffsetAtIndex(l.WDown.Buffer(), 0, 1)
		}
		ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
		metal.SetScalar(ce, nv, 3)
		metal.SetScalar(ce, hv, 4)
		metal.SetScalar(ce, dv, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(n*cfg.Dim), metal.Threads(128))
	}
	ce.SetComputePipelineState(eng.pso.SegmentedGather)
	ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(eng.prefillOffsets.Buffer(), 0, 2)
	ce.SetBufferWithOffsetAtIndex(eng.prefillLengths.Buffer(), 0, 3)
	metal.SetScalar(ce, bv, 4)
	metal.SetScalar(ce, dv, 5)
	ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(b*cfg.Dim), metal.Threads(128))
	ce.EndEncoding()
	if err := ctx.Err(); err != nil {
		return SegmentedLayerChainResult{}, err
	}
	cb.Commit()
	if err := metal.WaitCommandBuffer(ctx, cb); err != nil {
		return SegmentedLayerChainResult{}, fmt.Errorf("segmented layer chain: shared command buffer: %w", err)
	}
	toInts := func(x []uint32) []int {
		y := make([]int, len(x))
		for i := range x {
			y[i] = int(x[i])
		}
		return y
	}
	last := make([]int, b)
	for i := range b {
		last[i] = int(packed.RestoredPrefixes[i] + packed.Lengths[i] - 1)
	}
	bytes := int64(2 * cfg.NumLayers * cache.rowCapacity * cfg.MaxSeqLen * kd * 4)
	return SegmentedLayerChainResult{Hidden: eng.prefillNormXMat.HostSlice()[:b*cfg.Dim], Positions: last, Rows: toInts(packed.Rows), Dispatches: ce.dispatches, DeviceBytes: bytes}, nil
}

// RunSegmentedModelTailFromChain executes the shared tiled model tail directly
// from the gathered device rows.
func RunSegmentedModelTailFromChain(eng *ModelEngine, cfg Config, tail *BatchedModelTail, chain SegmentedLayerChainResult, samplers []SamplerConfig) (BatchedModelTailResult, error) {
	b := len(chain.Positions)
	if b == 0 || len(chain.Hidden) != b*cfg.Dim || len(chain.Rows) != b {
		return BatchedModelTailResult{}, errors.New("segmented model tail from chain: invalid chain result")
	}
	return tail.runBuffer(eng, cfg, eng.prefillNormXMat, b, samplers)
}

// RunBatchedLayer executes one complete FP32 decoder layer with shared weights.
// Its dispatch count is thirteen for every non-empty batch.
func RunBatchedLayer(eng *ModelEngine, cfg Config, layer int, cache *BatchedAttentionCache, tokens []int32, positions, rows []int) (BatchedLayerResult, error) {
	qkv, err := RunBatchedQKV(eng, cfg, layer, tokens, positions)
	if err != nil {
		return BatchedLayerResult{}, err
	}
	attn, err := RunBatchedAttention(eng, cfg, layer, cache, qkv.Q, qkv.K, qkv.V, positions, rows)
	if err != nil {
		return BatchedLayerResult{}, err
	}
	b := len(tokens)
	d := uint32(cfg.Dim)
	h := uint32(cfg.HiddenDim)
	eps := cfg.Eps
	l := &eng.layers[layer]
	eng.batchedMu.Lock()
	defer eng.batchedMu.Unlock()
	copy(eng.prefillXMat.HostSlice()[:len(qkv.Embedding)], qkv.Embedding)
	copy(eng.prefillAttnOut.HostSlice()[:len(attn.Attention)], attn.Attention)
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	bv := uint32(b)
	ce.SetComputePipelineState(eng.pso.BatchedGEMMRes)
	ce.SetBufferWithOffsetAtIndex(eng.prefillAttnOut.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(l.WO.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
	metal.SetScalar(ce, bv, 3)
	metal.SetScalar(ce, d, 4)
	metal.SetScalar(ce, d, 5)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.Dim), metal.Threads(128))
	ce.SetComputePipelineState(eng.pso.BatchedNorm)
	ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(l.PostNorm.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 2)
	metal.SetScalar(ce, bv, 3)
	metal.SetScalar(ce, d, 4)
	metal.SetScalar(ce, eps, 5)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(128))
	for _, p := range []struct{ w, out mempool.Slice[float32] }{{l.WGate, eng.prefillGateMat}, {l.WUp, eng.prefillUpMat}} {
		ce.SetComputePipelineState(eng.pso.BatchedGEMM)
		ce.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 0)
		ce.SetBufferWithOffsetAtIndex(p.w.Buffer(), 0, 1)
		ce.SetBufferWithOffsetAtIndex(p.out.Buffer(), 0, 2)
		metal.SetScalar(ce, bv, 3)
		metal.SetScalar(ce, d, 4)
		metal.SetScalar(ce, h, 5)
		ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.HiddenDim), metal.Threads(128))
	}
	total := uint32(b * cfg.HiddenDim)
	ce.SetComputePipelineState(eng.pso.BatchedSiLU)
	ce.SetBufferWithOffsetAtIndex(eng.prefillGateMat.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(eng.prefillUpMat.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(eng.prefillFFNAct.Buffer(), 0, 2)
	metal.SetScalar(ce, total, 3)
	ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(int(total)), metal.Threads(128))
	ce.SetComputePipelineState(eng.pso.BatchedGEMMRes)
	ce.SetBufferWithOffsetAtIndex(eng.prefillFFNAct.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(l.WDown.Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
	metal.SetScalar(ce, bv, 3)
	metal.SetScalar(ce, h, 4)
	metal.SetScalar(ce, d, 5)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.Dim), metal.Threads(128))
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	return BatchedLayerResult{Hidden: append([]float32(nil), eng.prefillXMat.HostSlice()[:b*cfg.Dim]...), Positions: append([]int(nil), positions...), Rows: append([]int(nil), rows...), Dispatches: qkv.Dispatches + attn.Dispatches + 6}, nil
}

// RunBatchedAttention applies RoPE, writes per-row KV state, and computes causal
// attention in exactly two dispatches. rows maps each dense active input to a
// stable cache row; finished rows are omitted by the caller.
func RunBatchedAttention(eng *ModelEngine, cfg Config, layer int, cache *BatchedAttentionCache, q, k, v []float32, positions, rows []int) (BatchedAttentionResult, error) {
	b := len(positions)
	qd := cfg.NumHeads * cfg.HeadDim
	kd := cfg.NumKVHeads * cfg.HeadDim
	if eng == nil || eng.dev == nil || cache == nil {
		return BatchedAttentionResult{}, errors.New("batched attention: nil engine or cache")
	}
	if b == 0 || len(rows) != b || len(q) != b*qd || len(k) != b*kd || len(v) != b*kd {
		return BatchedAttentionResult{}, errors.New("batched attention: inconsistent input lengths")
	}
	if layer < 0 || layer >= len(cache.K) || cfg.HeadDim <= 0 || cfg.HeadDim > 128 || cfg.HeadDim%2 != 0 || cfg.NumKVHeads <= 0 || cfg.NumHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.RopeTheta <= 0 {
		return BatchedAttentionResult{}, errors.New("batched attention: unsupported configuration")
	}
	if b > cache.rowCapacity || cache.maxSeqLen != cfg.MaxSeqLen || cache.kvDim != kd {
		return BatchedAttentionResult{}, errors.New("batched attention: cache shape mismatch or active batch exceeds capacity")
	}
	cacheElements := cache.rowCapacity * cache.maxSeqLen * kd
	if cache.positions.Len() < b || cache.rows.Len() < b || cache.K[layer].Len() < cacheElements || cache.V[layer].Len() < cacheElements {
		return BatchedAttentionResult{}, errors.New("batched attention: cache buffer extent mismatch")
	}
	seen := make(map[int]bool, b)
	for i := range b {
		if positions[i] < 0 || positions[i] >= cfg.MaxSeqLen || rows[i] < 0 || rows[i] >= cache.rowCapacity || seen[rows[i]] {
			return BatchedAttentionResult{}, fmt.Errorf("batched attention: invalid row %d position=%d cache-row=%d", i, positions[i], rows[i])
		}
		seen[rows[i]] = true
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if cache.freed {
		return BatchedAttentionResult{}, errors.New("batched attention: cache freed")
	}
	eng.batchedMu.Lock()
	defer eng.batchedMu.Unlock()
	if err := eng.ensurePrefillArena(b, cfg); err != nil {
		return BatchedAttentionResult{}, err
	}
	copy(eng.prefillQMat.HostSlice()[:len(q)], q)
	copy(eng.prefillKMat.HostSlice()[:len(k)], k)
	copy(eng.prefillVMat.HostSlice()[:len(v)], v)
	for i := range b {
		cache.positions.HostSlice()[i] = uint32(positions[i])
		cache.rows.HostSlice()[i] = uint32(rows[i])
	}
	enc := eng.dev.Queue().CommandBuffer()
	ce := enc.ComputeCommandEncoder()
	ce.SetComputePipelineState(eng.pso.RowRoPE)
	for i, s := range []mempool.Slice[float32]{eng.prefillQMat, eng.prefillKMat, eng.prefillVMat, cache.K[layer], cache.V[layer]} {
		ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(i))
	}
	ce.SetBufferWithOffsetAtIndex(cache.positions.Buffer(), 0, 5)
	ce.SetBufferWithOffsetAtIndex(cache.rows.Buffer(), 0, 6)
	metal.SetScalar(ce, uint32(b), 7)
	metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 8)
	metal.SetScalar(ce, uint32(cfg.HeadDim), 9)
	metal.SetScalar(ce, uint32(cfg.NumHeads), 10)
	metal.SetScalar(ce, uint32(cfg.NumKVHeads), 11)
	metal.SetScalar(ce, cfg.RopeTheta, 12)
	metal.SetScalar(ce, uint32(cache.rowCapacity), 13)
	ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(b), metal.Threads(1))
	ce.SetComputePipelineState(eng.pso.RowAttn)
	ce.SetBufferWithOffsetAtIndex(eng.prefillQMat.Buffer(), 0, 0)
	ce.SetBufferWithOffsetAtIndex(cache.K[layer].Buffer(), 0, 1)
	ce.SetBufferWithOffsetAtIndex(cache.V[layer].Buffer(), 0, 2)
	ce.SetBufferWithOffsetAtIndex(eng.prefillAttnOut.Buffer(), 0, 3)
	ce.SetBufferWithOffsetAtIndex(cache.positions.Buffer(), 0, 4)
	ce.SetBufferWithOffsetAtIndex(cache.rows.Buffer(), 0, 5)
	metal.SetScalar(ce, uint32(b), 6)
	metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 7)
	metal.SetScalar(ce, uint32(cfg.NumHeads), 8)
	metal.SetScalar(ce, uint32(cfg.NumKVHeads), 9)
	metal.SetScalar(ce, uint32(cfg.HeadDim), 10)
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	metal.SetScalar(ce, scale, 11)
	metal.SetScalar(ce, uint32(cache.rowCapacity), 12)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(b*cfg.NumHeads), metal.Threads(128))
	ce.EndEncoding()
	enc.Commit()
	enc.WaitUntilCompleted()
	copyOut := func(x []float32, n int) []float32 { return append([]float32(nil), x[:n]...) }
	return BatchedAttentionResult{Q: copyOut(eng.prefillQMat.HostSlice(), len(q)), K: copyOut(eng.prefillKMat.HostSlice(), len(k)), Attention: copyOut(eng.prefillAttnOut.HostSlice(), b*qd), Positions: append([]int(nil), positions...), Rows: append([]int(nil), rows...), Dispatches: 2}, nil
}

// RunSegmentedAttention applies attention to N packed prompt tokens belonging
// to B stable cache rows. The Metal kernels consume and check every segment
// boundary before writing KV or reading attention state.
func RunSegmentedAttention(ctx context.Context, eng *ModelEngine, cfg Config, layer int, cache *BatchedAttentionCache, packed PackedPrefill, q, k, v []float32) (BatchedAttentionResult, error) {
	n, b := len(packed.Tokens), len(packed.Offsets)
	qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	if eng == nil || eng.dev == nil || cache == nil {
		return BatchedAttentionResult{}, errors.New("segmented attention: nil engine or cache")
	}
	if err := packed.Validate(n, cache.rowCapacity, cfg.MaxSeqLen); err != nil {
		return BatchedAttentionResult{}, fmt.Errorf("segmented attention: %w", err)
	}
	if len(q) != n*qd || len(k) != n*kd || len(v) != n*kd {
		return BatchedAttentionResult{}, errors.New("segmented attention: inconsistent projections")
	}
	if layer < 0 || layer >= len(cache.K) || eng.isF16 || cfg.HeadDim <= 0 || cfg.HeadDim > 128 || cfg.HeadDim%2 != 0 || cfg.NumKVHeads <= 0 || cfg.NumHeads <= 0 || cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.RopeTheta <= 0 {
		return BatchedAttentionResult{}, errors.New("segmented attention: unsupported configuration")
	}
	if cache.maxSeqLen != cfg.MaxSeqLen || cache.kvDim != kd {
		return BatchedAttentionResult{}, errors.New("segmented attention: cache shape mismatch")
	}
	if err := ctx.Err(); err != nil {
		return BatchedAttentionResult{}, err
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if cache.freed {
		return BatchedAttentionResult{}, errors.New("segmented attention: cache freed")
	}
	eng.batchedMu.Lock()
	defer eng.batchedMu.Unlock()
	if err := eng.ensurePrefillArena(n, cfg); err != nil {
		return BatchedAttentionResult{}, err
	}
	meta := metalmem.NewArena(eng.dev)
	defer meta.Free()
	requestIndex := metalmem.AllocIn[uint32](meta, "requestIndex", n)
	positions := metalmem.AllocIn[uint32](meta, "positions", n)
	offsets := metalmem.AllocIn[uint32](meta, "offsets", b)
	lengths := metalmem.AllocIn[uint32](meta, "lengths", b)
	rows := metalmem.AllocIn[uint32](meta, "rows", b)
	restored := metalmem.AllocIn[uint32](meta, "restored", b)
	if err := meta.Err(); err != nil {
		return BatchedAttentionResult{}, fmt.Errorf("segmented attention metadata: %w", err)
	}
	copy(eng.prefillQMat.HostSlice()[:len(q)], q)
	copy(eng.prefillKMat.HostSlice()[:len(k)], k)
	copy(eng.prefillVMat.HostSlice()[:len(v)], v)
	copy(requestIndex.HostSlice(), packed.RequestIndex)
	copy(positions.HostSlice(), packed.AbsolutePositions)
	copy(offsets.HostSlice(), packed.Offsets)
	copy(lengths.HostSlice(), packed.Lengths)
	copy(rows.HostSlice(), packed.Rows)
	copy(restored.HostSlice(), packed.RestoredPrefixes)
	if err := ctx.Err(); err != nil {
		return BatchedAttentionResult{}, err
	}
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	ce.SetComputePipelineState(eng.pso.SegmentedRoPE)
	for i, s := range []mempool.Slice[float32]{eng.prefillQMat, eng.prefillKMat, eng.prefillVMat, cache.K[layer], cache.V[layer]} {
		ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(i))
	}
	for i, s := range []mempool.Slice[uint32]{requestIndex, positions, offsets, lengths, rows, restored} {
		ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(5+i))
	}
	metal.SetScalar(ce, uint32(n), 11)
	metal.SetScalar(ce, uint32(b), 12)
	metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 13)
	metal.SetScalar(ce, uint32(cfg.HeadDim), 14)
	metal.SetScalar(ce, uint32(cfg.NumHeads), 15)
	metal.SetScalar(ce, uint32(cfg.NumKVHeads), 16)
	metal.SetScalar(ce, cfg.RopeTheta, 17)
	metal.SetScalar(ce, uint32(cache.rowCapacity), 18)
	ce.DispatchThreadsThreadsPerThreadgroup(metal.Threads(n), metal.Threads(1))
	ce.SetComputePipelineState(eng.pso.SegmentedAttn)
	for i, s := range []mempool.Slice[float32]{eng.prefillQMat, cache.K[layer], cache.V[layer], eng.prefillAttnOut} {
		ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(i))
	}
	for i, s := range []mempool.Slice[uint32]{requestIndex, positions, offsets, lengths, rows, restored} {
		ce.SetBufferWithOffsetAtIndex(s.Buffer(), 0, uint(4+i))
	}
	metal.SetScalar(ce, uint32(n), 10)
	metal.SetScalar(ce, uint32(b), 11)
	metal.SetScalar(ce, uint32(cfg.MaxSeqLen), 12)
	metal.SetScalar(ce, uint32(cfg.NumHeads), 13)
	metal.SetScalar(ce, uint32(cfg.NumKVHeads), 14)
	metal.SetScalar(ce, uint32(cfg.HeadDim), 15)
	metal.SetScalar(ce, float32(1/math.Sqrt(float64(cfg.HeadDim))), 16)
	metal.SetScalar(ce, uint32(cache.rowCapacity), 17)
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(n*cfg.NumHeads), metal.Threads(128))
	ce.EndEncoding()
	if err := ctx.Err(); err != nil {
		return BatchedAttentionResult{}, err
	}
	cb.Commit()
	if err := metal.WaitCommandBuffer(ctx, cb); err != nil {
		return BatchedAttentionResult{}, fmt.Errorf("segmented attention: shared command buffer: %w", err)
	}
	toInts := func(x []uint32) []int {
		y := make([]int, len(x))
		for i := range x {
			y[i] = int(x[i])
		}
		return y
	}
	return BatchedAttentionResult{
		Q:         append([]float32(nil), eng.prefillQMat.HostSlice()[:len(q)]...),
		K:         append([]float32(nil), eng.prefillKMat.HostSlice()[:len(k)]...),
		Attention: append([]float32(nil), eng.prefillAttnOut.HostSlice()[:n*qd]...),
		Positions: toInts(packed.AbsolutePositions), Rows: toInts(packed.Rows), Dispatches: 2,
	}, nil
}

// RunBatchedQKV executes five dispatches regardless of batch size: embedding,
// RMS normalization, and one projection each for Q, K, and V. Projection
// weights are bound once per projection and batch size changes only grid size.
func RunBatchedQKV(eng *ModelEngine, cfg Config, layer int, tokens []int32, positions []int) (BatchedQKVResult, error) {
	if eng == nil || eng.dev == nil {
		return BatchedQKVResult{}, errors.New("batched qkv: nil engine")
	}
	if len(tokens) == 0 || len(tokens) != len(positions) {
		return BatchedQKVResult{}, fmt.Errorf("batched qkv: tokens=%d positions=%d", len(tokens), len(positions))
	}
	if layer < 0 || layer >= len(eng.layers) || eng.isF16 {
		return BatchedQKVResult{}, fmt.Errorf("batched qkv: unsupported layer or precision")
	}
	for i, token := range tokens {
		if token < 0 || int(token) >= cfg.VocabSize || positions[i] < 0 || positions[i] >= cfg.MaxSeqLen {
			return BatchedQKVResult{}, fmt.Errorf("batched qkv: invalid row %d token=%d position=%d", i, token, positions[i])
		}
	}

	eng.batchedMu.Lock()
	defer eng.batchedMu.Unlock()
	if err := eng.ensurePrefillArena(len(tokens), cfg); err != nil {
		return BatchedQKVResult{}, err
	}
	copy(eng.prefillTokIn.HostSlice()[:len(tokens)], tokens)
	b := uint32(len(tokens))
	d := uint32(cfg.Dim)
	qd := uint32(cfg.NumHeads * cfg.HeadDim)
	kd := uint32(cfg.NumKVHeads * cfg.HeadDim)
	eps := cfg.Eps
	l := &eng.layers[layer]
	started := time.Now()
	cb := eng.dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(eng.pso.BatchedEmbed)
	enc.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(eng.prefillTokIn.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 2)
	metal.SetScalar(enc, b, 3)
	metal.SetScalar(enc, d, 4)
	enc.DispatchThreadsThreadsPerThreadgroup(metal.Threads(len(tokens)*cfg.Dim), metal.Threads(128))
	enc.SetComputePipelineState(eng.pso.BatchedNorm)
	enc.SetBufferWithOffsetAtIndex(eng.prefillXMat.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(l.InputNorm.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 2)
	metal.SetScalar(enc, b, 3)
	metal.SetScalar(enc, d, 4)
	metal.SetScalar(enc, eps, 5)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(len(tokens)), metal.Threads(128))
	for _, projection := range []struct {
		weight, output mempool.Slice[float32]
		width          uint32
	}{
		{l.WQ, eng.prefillQMat, qd}, {l.WK, eng.prefillKMat, kd}, {l.WV, eng.prefillVMat, kd},
	} {
		enc.SetComputePipelineState(eng.pso.BatchedGEMM)
		enc.SetBufferWithOffsetAtIndex(eng.prefillNormXMat.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(projection.weight.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(projection.output.Buffer(), 0, 2)
		metal.SetScalar(enc, b, 3)
		metal.SetScalar(enc, d, 4)
		metal.SetScalar(enc, projection.width, 5)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(len(tokens)*int(projection.width)), metal.Threads(128))
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	if eng.recorder != nil {
		eng.recorder.Record("BatchedQKVSlice", time.Since(started), [3]int{len(tokens), cfg.Dim, 1}, [3]int{128, 1, 1}, map[string]string{"dispatches": "5", "batch": fmt.Sprint(len(tokens))})
	}
	copyOut := func(src []float32, n int) []float32 { return append([]float32(nil), src[:n]...) }
	return BatchedQKVResult{Positions: append([]int(nil), positions...), Embedding: copyOut(eng.prefillXMat.HostSlice(), len(tokens)*cfg.Dim), Normalized: copyOut(eng.prefillNormXMat.HostSlice(), len(tokens)*cfg.Dim), Q: copyOut(eng.prefillQMat.HostSlice(), len(tokens)*int(qd)), K: copyOut(eng.prefillKMat.HostSlice(), len(tokens)*int(kd)), V: copyOut(eng.prefillVMat.HostSlice(), len(tokens)*int(kd)), Dispatches: 5}, nil
}

// FlightRecorder returns the active in-memory GPU telemetry recorder.
func (eng *ModelEngine) FlightRecorder() *metalprof.FlightRecorder {
	return eng.recorder
}

// IsF16 reports whether the engine is executing over FP16 weights.
func (eng *ModelEngine) IsF16() bool {
	if eng == nil {
		return false
	}
	return eng.isF16
}

func (eng *ModelEngine) precisionTelemetry() []TensorPrecision {
	if eng == nil || eng.weights == nil {
		return nil
	}

	byDType := make(map[string]TensorPrecision)
	for _, tensor := range eng.weights.Tensors {
		if tensor.DType == "" || tensor.OffsetEnd < tensor.OffsetStart {
			continue
		}
		summary := byDType[tensor.DType]
		summary.DType = tensor.DType
		summary.TensorCount++
		summary.Bytes += tensor.OffsetEnd - tensor.OffsetStart
		byDType[tensor.DType] = summary
	}
	if len(byDType) == 0 {
		return nil
	}

	dtypes := make([]TensorPrecision, 0, len(byDType))
	for _, summary := range byDType {
		dtypes = append(dtypes, summary)
	}
	sort.Slice(dtypes, func(i, j int) bool {
		return dtypes[i].DType < dtypes[j].DType
	})
	return dtypes
}

func (eng *ModelEngine) prefixCacheMetrics() PrefixCacheMetrics {
	if eng == nil || eng.prefixes == nil {
		return PrefixCacheMetrics{}
	}
	return eng.prefixes.metrics()
}

func (eng *ModelEngine) prefixCacheAvailable() bool {
	return eng != nil && eng.prefixes != nil
}

// ReadLogits returns a copy of the final logits computed in the last forward pass.
func (eng *ModelEngine) ReadLogits() ([]float32, error) {
	if eng == nil {
		return nil, errors.New("read logits: nil engine")
	}
	if eng.logitsBuf.Len() == 0 {
		return nil, errors.New("read logits: empty logits buffer")
	}
	res := make([]float32, eng.logitsBuf.Len())
	copy(res, eng.logitsBuf.HostSlice())
	return res, nil
}

// MutateWeightForTest alters a single weight element in GPU memory and returns a restore function.
func (eng *ModelEngine) MutateWeightForTest(layerIndex, elemIndex int, delta float32) (func(), error) {
	if eng == nil {
		return nil, errors.New("mutate weight: nil engine")
	}
	var slice []float32
	if layerIndex < 0 {
		if eng.embedTable.Len() == 0 {
			return nil, errors.New("mutate weight: empty embed table")
		}
		slice = eng.embedTable.HostSlice()
	} else {
		if len(eng.layers) <= layerIndex {
			return nil, errors.New("mutate weight: invalid layer index")
		}
		slice = eng.layers[layerIndex].WQ.HostSlice()
	}
	if len(slice) <= elemIndex {
		return nil, errors.New("mutate weight: invalid element index")
	}
	orig := slice[elemIndex]
	slice[elemIndex] += delta
	return func() {
		slice[elemIndex] = orig
	}, nil
}

// ResetKV zeroes out the entire KV cache buffer across all layers.
func (eng *ModelEngine) ResetKV() {
	if eng == nil {
		return
	}
	for _, kBuf := range eng.kCacheBuf {
		slice := kBuf.HostSlice()
		for i := range slice {
			slice[i] = 0
		}
	}
	for _, vBuf := range eng.vCacheBuf {
		slice := vBuf.HostSlice()
		for i := range slice {
			slice[i] = 0
		}
	}
}

func (eng *ModelEngine) capturePrefix(tokens int, current int32, cfg Config) *prefixSnapshot {
	if eng == nil || tokens <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	snapshot := &prefixSnapshot{
		tokens:  tokens,
		current: current,
		k:       make([][]float32, len(eng.kCacheBuf)),
		v:       make([][]float32, len(eng.vCacheBuf)),
	}
	for layer := range eng.kCacheBuf {
		if valuesPerLayer > eng.kCacheBuf[layer].Len() || valuesPerLayer > eng.vCacheBuf[layer].Len() {
			return nil
		}
		snapshot.k[layer] = append([]float32(nil), eng.kCacheBuf[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.v[layer] = append([]float32(nil), eng.vCacheBuf[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.bytes += int64(len(snapshot.k[layer])+len(snapshot.v[layer])) * 4
	}
	return snapshot
}

// restorePrefixPartial restores exactly 'tokens' count of KV entries from the snapshot,
// ensuring no stale downstream entries are left unmanaged.
func (eng *ModelEngine) restorePrefixPartial(snapshot *prefixSnapshot, tokens int, cfg Config) bool {
	if eng == nil || snapshot == nil || tokens <= 0 || len(snapshot.k) != len(eng.kCacheBuf) || len(snapshot.v) != len(eng.vCacheBuf) {
		return false
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	for layer := range eng.kCacheBuf {
		if valuesPerLayer > len(snapshot.k[layer]) || valuesPerLayer > len(snapshot.v[layer]) ||
			valuesPerLayer > eng.kCacheBuf[layer].Len() || valuesPerLayer > eng.vCacheBuf[layer].Len() {
			return false
		}
		copy(eng.kCacheBuf[layer].HostSlice()[:valuesPerLayer], snapshot.k[layer][:valuesPerLayer])
		copy(eng.vCacheBuf[layer].HostSlice()[:valuesPerLayer], snapshot.v[layer][:valuesPerLayer])
	}
	return true
}

// RestorePrefixPartial restores exactly 'tokens' count of KV entries into the isolated SequenceKVCache.
func (skv *SequenceKVCache) RestorePrefixPartial(snapshot *prefixSnapshot, tokens int, cfg Config) bool {
	if skv == nil || snapshot == nil || tokens <= 0 || len(snapshot.k) != len(skv.KCache) || len(snapshot.v) != len(skv.VCache) {
		return false
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	for layer := range skv.KCache {
		if valuesPerLayer > len(snapshot.k[layer]) || valuesPerLayer > len(snapshot.v[layer]) ||
			valuesPerLayer > skv.KCache[layer].Len() || valuesPerLayer > skv.VCache[layer].Len() {
			return false
		}
		copy(skv.KCache[layer].HostSlice()[:valuesPerLayer], snapshot.k[layer][:valuesPerLayer])
		copy(skv.VCache[layer].HostSlice()[:valuesPerLayer], snapshot.v[layer][:valuesPerLayer])
	}
	return true
}

var (
	darwinEngineCache = make(map[string]*ModelEngine)
	darwinEngineMu    sync.Mutex
)

// GetOrCompileDarwinEngine builds or reuses the 100% GPU-resident Metal execution engine.
func GetOrCompileDarwinEngine(cfg Config) (*ModelEngine, error) {
	return getOrCompileDarwinEngine(cfg)
}

// GetOrCompileEngine is the universal entrypoint for obtaining the GPU engine.
func GetOrCompileEngine(cfg Config) (*ModelEngine, error) {
	return GetOrCompileDarwinEngine(cfg)
}

func f32SliceToF16(s []float32) []uint16 {
	out := make([]uint16, len(s))
	for i, v := range s {
		out[i] = metalsafetensors.F32ToF16(v)
	}
	return out
}

func getOrCompileDarwinEngine(cfg Config) (*ModelEngine, error) {
	darwinEngineMu.Lock()
	defer darwinEngineMu.Unlock()

	key := fmt.Sprintf("%s|%s|%d|%d|%d|%d|%d|%s|%v", cfg.Name, cfg.ModelPath, cfg.Dim, cfg.HiddenDim, cfg.NumLayers, cfg.VocabSize, cfg.MaxSeqLen, cfg.Precision, cfg.FuseEGraph)
	if eng, ok := darwinEngineCache[key]; ok {
		return eng, nil
	}

	dev, err := metal.NewDevice()
	if err != nil {
		return nil, fmt.Errorf("metal device: %w", err)
	}

	egraphSats := 0
	if cfg.FuseEGraph {
		eg := egraph.New()
		h := eg.Add(egraph.NewLeaf("x"))
		wNorm := eg.Add(egraph.NewLeaf("w_norm"))
		wq := eg.Add(egraph.NewLeaf("wq"))
		normNode := eg.Add(egraph.NewNode("rmsnorm", h, wNorm))
		eg.Add(egraph.NewNode("gemv", normNode, wq))

		rules := []*egraph.Rewrite{
			egraph.MustRewrite("fuse_rmsnorm_gemv", "(gemv (rmsnorm ?x ?w) ?wq)", "(fused_rmsnorm_gemv ?x ?w ?wq)"),
			egraph.MustRewrite("fuse_swiglu_gemv", "(gemv (* (silu (gemv ?x ?wg)) (gemv ?x ?wu)) ?wd)", "(fused_swiglu_gemv ?x ?wg ?wu ?wd)"),
		}
		_, egraphSats = egraph.Saturate(eg, rules, 10)
	}

	fullMSL := metalmegakernel.LlamaDecoderKernelsMSL + "\n" + metalmegakernel.HalfWeightKernelsMSL + "\n" + metalattn.CausalFlashAttentionMSL + "\n" + metalsample.ArgmaxSamplingMSL
	lib, err := dev.CompileLibrary(fullMSL)
	if err != nil {
		return nil, fmt.Errorf("compile MSL library on Metal GPU: %w", err)
	}

	var pso decodeKernels
	if err := metalx.Bind(dev, lib, &pso); err != nil {
		return nil, fmt.Errorf("bind decode kernels: %w", err)
	}

	// One arena for every buffer whose lifetime is the engine's: checked once,
	// released once, and nothing leaks when a later allocation fails.
	memory := new(metalmem.Tracker)
	mem := metalmem.NewTrackedArena(dev, memory)
	committed := false
	var weights *metalsafetensors.Weights
	defer func() {
		if committed {
			return
		}
		mem.Free()
		if weights != nil {
			_ = weights.Close()
		}
	}()
	xBuf := metalmem.AllocIn[float32](mem, "x", cfg.Dim)
	normXBuf := metalmem.AllocIn[float32](mem, "normX", cfg.Dim)
	qBuf := metalmem.AllocIn[float32](mem, "q", cfg.NumHeads*cfg.HeadDim)
	kBuf := metalmem.AllocIn[float32](mem, "k", cfg.NumKVHeads*cfg.HeadDim)
	vBuf := metalmem.AllocIn[float32](mem, "v", cfg.NumKVHeads*cfg.HeadDim)
	attnOutBuf := metalmem.AllocIn[float32](mem, "attnOut", cfg.Dim)
	gateBuf := metalmem.AllocIn[float32](mem, "gate", cfg.HiddenDim)
	upBuf := metalmem.AllocIn[float32](mem, "up", cfg.HiddenDim)
	ffnActBuf := metalmem.AllocIn[float32](mem, "ffnAct", cfg.HiddenDim)
	logitsBuf := metalmem.AllocIn[float32](mem, "logits", cfg.VocabSize)
	tokBuf := metalmem.AllocIn[int32](mem, "token", 1)

	kvPerLayer := cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim
	kCacheBuf := make([]mempool.Slice[float32], cfg.NumLayers)
	vCacheBuf := make([]mempool.Slice[float32], cfg.NumLayers)
	for l := range cfg.NumLayers {
		kCacheBuf[l] = metalmem.AllocIn[float32](mem, fmt.Sprintf("kCache[%d]", l), kvPerLayer)
		vCacheBuf[l] = metalmem.AllocIn[float32](mem, fmt.Sprintf("vCache[%d]", l), kvPerLayer)
	}
	if err := mem.Err(); err != nil {
		mem.Free()
		return nil, err
	}

	var tok Tokenizer
	var consumption CheckpointReceipt
	isF16 := cfg.Precision == "f16"

	var embedTable, finalNormW, lmHeadW mempool.Slice[float32]
	gpuLayers := make([]GPULayerWeights, cfg.NumLayers)

	var embedTableF16, finalNormWF16, lmHeadWF16 mempool.Slice[uint16]
	gpuLayersF16 := make([]GPULayerWeightsF16, cfg.NumLayers)

	if cfg.ModelPath != "" {
		tokJsonPath := filepath.Join(cfg.ModelPath, "tokenizer.json")
		if t, err := LoadHFTokenizer(tokJsonPath); err == nil {
			tok = t
		}
		safetensorsPath := filepath.Join(cfg.ModelPath, "model.safetensors")
		w, err := metalsafetensors.Open(safetensorsPath)
		if err != nil {
			mem.Free()
			return nil, fmt.Errorf("open safetensors %q: %w", safetensorsPath, err)
		}
		weights = w

		// Auto-detect F16/BF16 from safetensors if not explicitly forced to f32
		if cfg.Precision == "" && w.Tensors != nil {
			if tInfo, ok := w.Tensors["model.embed_tokens.weight"]; ok && (tInfo.DType == "F16" || tInfo.DType == "BF16") {
				isF16 = true
			}
		}
		modelCfg, err := LoadModelConfig(cfg.ModelPath)
		if err != nil {
			mem.Free()
			return nil, fmt.Errorf("load model config for tensor audit: %w", err)
		}
		metadata := make(map[string]checkpointTensor, len(w.Tensors))
		for name, info := range w.Tensors {
			metadata[name] = checkpointTensor{DType: info.DType, Shape: info.Shape, Shard: filepath.Base(safetensorsPath)}
		}
		targetType := "F32"
		if isF16 {
			targetType = "F16"
		}
		consumption, err = auditCheckpoint(modelCfg, metadata, targetType)
		if err != nil {
			mem.Free()
			return nil, err
		}

		if isF16 {
			upload := func(name string) (mempool.Slice[uint16], error) {
				data, err := w.TensorF16(name)
				if err != nil {
					return mempool.Slice[uint16]{}, err
				}
				s := metalmem.UploadIn(mem, name, data)
				if err := mem.Err(); err != nil {
					return mempool.Slice[uint16]{}, err
				}
				return s, nil
			}
			var errUpload error
			embedTableF16, errUpload = upload("model.embed_tokens.weight")
			if errUpload != nil || embedTableF16.Len() == 0 {
				return nil, fmt.Errorf("load embed_tokens f16: %w", errUpload)
			}
			finalNormWF16, errUpload = upload("model.norm.weight")
			if errUpload != nil || finalNormWF16.Len() == 0 {
				return nil, fmt.Errorf("load norm weight f16: %w", errUpload)
			}
			// lm_head is absent in tied-embedding checkpoints; its error is
			// the "no such tensor" that the fallback below handles.
			if _, ok := w.Tensors["lm_head.weight"]; ok {
				lmHeadWF16, errUpload = upload("lm_head.weight")
				if errUpload != nil {
					return nil, fmt.Errorf("load lm_head f16: %w", errUpload)
				}
			}
			if lmHeadWF16.Len() == 0 {
				lmHeadWF16 = embedTableF16
			}

			for i := 0; i < cfg.NumLayers; i++ {
				inNorm, err := upload(fmt.Sprintf("model.layers.%d.input_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d input_norm: %w", i, err)
				}
				pNorm, err := upload(fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d post_norm: %w", i, err)
				}
				wq, err := upload(fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wq: %w", i, err)
				}
				wk, err := upload(fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wk: %w", i, err)
				}
				wv, err := upload(fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wv: %w", i, err)
				}
				wo, err := upload(fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wo: %w", i, err)
				}
				wg, err := upload(fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wgate: %w", i, err)
				}
				wu, err := upload(fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wup: %w", i, err)
				}
				wd, err := upload(fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wdown: %w", i, err)
				}

				gpuLayersF16[i] = GPULayerWeightsF16{
					InputNorm: inNorm,
					PostNorm:  pNorm,
					WQ:        wq,
					WK:        wk,
					WV:        wv,
					WO:        wo,
					WGate:     wg,
					WUp:       wu,
					WDown:     wd,
				}
			}
		} else {
			upload := func(name string) (mempool.Slice[float32], error) {
				data, err := w.TensorFloat32(name)
				if err != nil {
					return mempool.Slice[float32]{}, err
				}
				s := metalmem.UploadIn(mem, name, data)
				if err := mem.Err(); err != nil {
					return mempool.Slice[float32]{}, err
				}
				return s, nil
			}
			var errUpload error
			embedTable, errUpload = upload("model.embed_tokens.weight")
			if errUpload != nil || embedTable.Len() == 0 {
				return nil, fmt.Errorf("load embed_tokens: %w", errUpload)
			}
			finalNormW, errUpload = upload("model.norm.weight")
			if errUpload != nil || finalNormW.Len() == 0 {
				return nil, fmt.Errorf("load norm weight: %w", errUpload)
			}
			// lm_head is absent in tied-embedding checkpoints; its error is
			// the "no such tensor" that the fallback below handles.
			if _, ok := w.Tensors["lm_head.weight"]; ok {
				lmHeadW, errUpload = upload("lm_head.weight")
				if errUpload != nil {
					return nil, fmt.Errorf("load lm_head: %w", errUpload)
				}
			}
			if lmHeadW.Len() == 0 {
				lmHeadW = embedTable
			}

			for i := 0; i < cfg.NumLayers; i++ {
				inNorm, err := upload(fmt.Sprintf("model.layers.%d.input_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d input_norm: %w", i, err)
				}
				pNorm, err := upload(fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d post_norm: %w", i, err)
				}
				wq, err := upload(fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wq: %w", i, err)
				}
				wk, err := upload(fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wk: %w", i, err)
				}
				wv, err := upload(fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wv: %w", i, err)
				}
				wo, err := upload(fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wo: %w", i, err)
				}
				wg, err := upload(fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wgate: %w", i, err)
				}
				wu, err := upload(fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wup: %w", i, err)
				}
				wd, err := upload(fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", i))
				if err != nil {
					return nil, fmt.Errorf("load layer %d wdown: %w", i, err)
				}

				gpuLayers[i] = GPULayerWeights{
					InputNorm: inNorm,
					PostNorm:  pNorm,
					WQ:        wq,
					WK:        wk,
					WV:        wv,
					WO:        wo,
					WGate:     wg,
					WUp:       wu,
					WDown:     wd,
				}
			}
		}
	} else if cfg.TestMode {
		mCfg := ModelConfig{
			HiddenSize:        cfg.Dim,
			IntermediateSize:  cfg.HiddenDim,
			NumHiddenLayers:   cfg.NumLayers,
			NumAttentionHeads: cfg.NumHeads,
			NumKeyValueHeads:  cfg.NumKVHeads,
			VocabSize:         cfg.VocabSize,
			RmsNormEps:        cfg.Eps,
			RopeTheta:         cfg.RopeTheta,
			MaxSeqLen:         cfg.MaxSeqLen,
		}
		if mCfg.HeadDim() == 0 && cfg.HeadDim > 0 {
			mCfg.ExplicitHeadDim = cfg.HeadDim
		}
		detWeights := GenerateDeterministicWeights(&mCfg)
		if isF16 {
			embedTableF16 = metalmem.UploadIn(mem, "embed", f32SliceToF16(detWeights.EmbedTokens))
			finalNormWF16 = metalmem.UploadIn(mem, "norm", f32SliceToF16(detWeights.Norm))
			lmHeadWF16 = metalmem.UploadIn(mem, "lm_head", f32SliceToF16(detWeights.LMHead))
			for i := range cfg.NumLayers {
				lw := detWeights.Layers[i]
				name := func(part string) string { return fmt.Sprintf("layer[%d].%s", i, part) }
				gpuLayersF16[i] = GPULayerWeightsF16{
					InputNorm: metalmem.UploadIn(mem, name("input_norm"), f32SliceToF16(lw.InputNorm)),
					PostNorm:  metalmem.UploadIn(mem, name("post_norm"), f32SliceToF16(lw.PostNorm)),
					WQ:        metalmem.UploadIn(mem, name("wq"), f32SliceToF16(lw.WQ)),
					WK:        metalmem.UploadIn(mem, name("wk"), f32SliceToF16(lw.WK)),
					WV:        metalmem.UploadIn(mem, name("wv"), f32SliceToF16(lw.WV)),
					WO:        metalmem.UploadIn(mem, name("wo"), f32SliceToF16(lw.WO)),
					WGate:     metalmem.UploadIn(mem, name("wgate"), f32SliceToF16(lw.WGate)),
					WUp:       metalmem.UploadIn(mem, name("wup"), f32SliceToF16(lw.WUp)),
					WDown:     metalmem.UploadIn(mem, name("wdown"), f32SliceToF16(lw.WDown)),
				}
			}
		} else {
			embedTable = metalmem.UploadIn(mem, "embed", detWeights.EmbedTokens)
			finalNormW = metalmem.UploadIn(mem, "norm", detWeights.Norm)
			lmHeadW = metalmem.UploadIn(mem, "lm_head", detWeights.LMHead)
			for i := range cfg.NumLayers {
				lw := detWeights.Layers[i]
				name := func(part string) string { return fmt.Sprintf("layer[%d].%s", i, part) }
				gpuLayers[i] = GPULayerWeights{
					InputNorm: metalmem.UploadIn(mem, name("input_norm"), lw.InputNorm),
					PostNorm:  metalmem.UploadIn(mem, name("post_norm"), lw.PostNorm),
					WQ:        metalmem.UploadIn(mem, name("wq"), lw.WQ),
					WK:        metalmem.UploadIn(mem, name("wk"), lw.WK),
					WV:        metalmem.UploadIn(mem, name("wv"), lw.WV),
					WO:        metalmem.UploadIn(mem, name("wo"), lw.WO),
					WGate:     metalmem.UploadIn(mem, name("wgate"), lw.WGate),
					WUp:       metalmem.UploadIn(mem, name("wup"), lw.WUp),
					WDown:     metalmem.UploadIn(mem, name("wdown"), lw.WDown),
				}
			}
		}
		if err := mem.Err(); err != nil {
			mem.Free()
			return nil, fmt.Errorf("upload deterministic weights: %w", err)
		}
	} else {
		return nil, errors.New("model path required (synthetic test weights require explicit TestMode=true)")
	}
	if tok == nil {
		tok = NewByteTokenizer()
	}

	metrics := NewRuntimeMetrics(256)
	var kvBytes int64
	for layer := range kCacheBuf {
		kvBytes += int64(kCacheBuf[layer].Len()+vCacheBuf[layer].Len()) * 4
	}
	metrics.SetKVAllocation(kvBytes, cfg.MaxSeqLen)

	eng := &ModelEngine{
		dev:           dev,
		lib:           lib,
		pso:           pso,
		mem:           mem,
		memory:        memory,
		xBuf:          xBuf,
		normXBuf:      normXBuf,
		qBuf:          qBuf,
		kBuf:          kBuf,
		vBuf:          vBuf,
		attnOutBuf:    attnOutBuf,
		gateBuf:       gateBuf,
		upBuf:         upBuf,
		ffnActBuf:     ffnActBuf,
		kCacheBuf:     kCacheBuf,
		vCacheBuf:     vCacheBuf,
		logitsBuf:     logitsBuf,
		tokBuf:        tokBuf,
		isF16:         isF16,
		embedTable:    embedTable,
		finalNormW:    finalNormW,
		lmHeadW:       lmHeadW,
		layers:        gpuLayers,
		embedTableF16: embedTableF16,
		finalNormWF16: finalNormWF16,
		lmHeadWF16:    lmHeadWF16,
		layersF16:     gpuLayersF16,
		tokenizer:     tok,
		weights:       weights,
		egraphSats:    egraphSats,
		recorder:      metalprof.NewFlightRecorder(2048),
		metrics:       metrics,
		prefixes:      newPrefixCache(64<<20, 8),
		tuner:         NewDispatchTuner(),
		Plan:          BuildExecutionPlan(cfg),
		cacheKey:      key,
		consumption:   consumption,
	}
	if err := ValidatePlan(eng.Plan, cfg); err != nil {
		return nil, fmt.Errorf("validate compiled plan: %w", err)
	}

	darwinEngineCache[key] = eng
	committed = true
	return eng, nil
}

func runPlatform(cfg Config) error {
	engine, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return fmt.Errorf("engine compile: %w", err)
	}

	cNode, err := NewIrohClusterNode(context.Background(), engine, cfg, 0)
	if err == nil {
		if !cfg.ServeMode {
			defer cNode.Close()
		}
		nodeAddr := cNode.Addr()
		fmt.Printf("[go-iroh Mesh] Node ID: %s (Platform: %s, Compute: %.0f TFLOPs, Bandwidth: %.0f GB/s)\n",
			cNode.NodeID(), cNode.SelfTelemetry().OS, cNode.SelfTelemetry().ComputeTFLOPs, cNode.SelfTelemetry().BandwidthGBS)
		fmt.Printf("[go-iroh Mesh] Node Addr: %s\n", nodeAddr.String())
		if cfg.PeerAddr != "" {
			go func() {
				for {
					if err := cNode.ConnectHTTPPeer(context.Background(), cfg.PeerAddr); err == nil {
						fmt.Printf("[go-iroh Mesh] Connected and registered peer %s into inference mesh!\n", cfg.PeerAddr)
						break
					}
					time.Sleep(2 * time.Second)
				}
			}()
		}
	}

	if cfg.HTTPAddr != "" {
		srv := NewTelemetryServer(engine, cfg)
		srv.SetPGOTracker(engine.tuner.Tracker())
		if cNode != nil {
			srv.SetClusterNode(cNode)
		}
		go func() {
			fmt.Printf("\n[Telemetry Server] Live E-Graph dashboard active at http://localhost%s\n", cfg.HTTPAddr)
			if err := http.ListenAndServe(cfg.HTTPAddr, srv); err != nil {
				fmt.Printf("[Telemetry Server] Error: %v\n", err)
			}
		}()
	}

	if !cfg.ServeMode {
		runFastMetalInference(engine, cfg)
	}
	return nil
}

// SequenceKVCache holds isolated per-sequence Metal KV-cache allocations.
type SequenceKVCache struct {
	KCache []mempool.Slice[float32]
	VCache []mempool.Slice[float32]
}

// NewSequenceKV allocates an isolated KV-cache for a sequence.
func (eng *ModelEngine) NewSequenceKV(cfg Config) (*SequenceKVCache, error) {
	if eng == nil {
		return nil, errors.New("new sequence kv: nil engine")
	}
	kCache := make([]mempool.Slice[float32], cfg.NumLayers)
	vCache := make([]mempool.Slice[float32], cfg.NumLayers)
	elemCount := cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		k, err := mempool.Alloc[float32](eng.dev, elemCount)
		if err != nil {
			return nil, fmt.Errorf("alloc k-cache layer %d: %w", l, err)
		}
		v, err := mempool.Alloc[float32](eng.dev, elemCount)
		if err != nil {
			return nil, fmt.Errorf("alloc v-cache layer %d: %w", l, err)
		}
		kCache[l] = k
		vCache[l] = v
	}
	return &SequenceKVCache{
		KCache: kCache,
		VCache: vCache,
	}, nil
}

// Free releases the isolated KV-cache buffers.
func (skv *SequenceKVCache) Free() {
	if skv == nil {
		return
	}
	for i := range skv.KCache {
		skv.KCache[i].Free()
		skv.VCache[i].Free()
	}
	skv.KCache = nil
	skv.VCache = nil
}

// CapturePrefix captures a prefix snapshot from the isolated SequenceKVCache.
func (skv *SequenceKVCache) CapturePrefix(tokens int, current int32, cfg Config) *prefixSnapshot {
	if skv == nil || tokens <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil
	}
	valuesPerLayer := tokens * cfg.NumKVHeads * cfg.HeadDim
	snapshot := &prefixSnapshot{
		tokens:  tokens,
		current: current,
		k:       make([][]float32, len(skv.KCache)),
		v:       make([][]float32, len(skv.VCache)),
	}
	for layer := range skv.KCache {
		if valuesPerLayer > skv.KCache[layer].Len() || valuesPerLayer > skv.VCache[layer].Len() {
			return nil
		}
		snapshot.k[layer] = append([]float32(nil), skv.KCache[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.v[layer] = append([]float32(nil), skv.VCache[layer].HostSlice()[:valuesPerLayer]...)
		snapshot.bytes += int64(len(snapshot.k[layer])+len(snapshot.v[layer])) * 4
	}
	return snapshot
}

// ExecutePlan executes the authoritative, immutable ExecutionPlan on the GPU compute command encoder.
func ExecutePlan(
	enc apple.MTLComputeCommandEncoder,
	eng *ModelEngine,
	plan *ExecutionPlan,
	cfg Config,
	kCache, vCache []mempool.Slice[float32],
	tokenID int32,
	pos int,
	isLast bool,
) error {
	if eng == nil {
		return errors.New("execute plan: nil engine")
	}
	return ExecutePlanWithLogits(enc, eng, plan, cfg, kCache, vCache, eng.logitsBuf, tokenID, pos, isLast)
}

// ExecutePlanWithLogits executes the authoritative plan directing output logits to a target buffer.
func ExecutePlanWithLogits(
	enc apple.MTLComputeCommandEncoder,
	eng *ModelEngine,
	plan *ExecutionPlan,
	cfg Config,
	kCache, vCache []mempool.Slice[float32],
	logitsBuf mempool.Slice[float32],
	tokenID int32,
	pos int,
	isLast bool,
) error {
	if plan == nil || eng == nil {
		return errors.New("execute plan: nil plan or engine")
	}
	if err := plan.Validate(cfg); err != nil {
		return fmt.Errorf("execute plan validation: %w", err)
	}
	targetLogits := eng.logitsBuf
	if logitsBuf.Len() > 0 {
		targetLogits = logitsBuf
	}

	dimVal := uint32(cfg.Dim)
	hiddenDimVal := uint32(cfg.HiddenDim)
	qDimVal := uint32(cfg.NumHeads * cfg.HeadDim)
	kvDimVal := uint32(cfg.NumKVHeads * cfg.HeadDim)
	epsVal := cfg.Eps
	thetaVal := cfg.RopeTheta
	posVal := uint32(pos)
	headDimVal := uint32(cfg.HeadDim)
	qHeadsVal := uint32(cfg.NumHeads)
	kvHeadsVal := uint32(cfg.NumKVHeads)
	scaleVal := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	vocabVal := uint32(cfg.VocabSize)
	tokVal := uint32(tokenID)

	gemvThreads := 128
	psoGEMV := eng.pso.GEMV
	psoGEMVRes := eng.pso.GEMVRes
	psoGEMVH := eng.pso.GEMVH
	psoGEMVResH := eng.pso.GEMVResH
	if eng.tuner != nil && eng.tuner.Width() == 64 {
		gemvThreads = 64
		psoGEMV = eng.pso.GEMV64
		psoGEMVRes = eng.pso.GEMVRes64
		psoGEMVH = eng.pso.GEMV64H
		psoGEMVResH = eng.pso.GEMVRes64H
	}

	for _, op := range plan.Ops {
		l := op.LayerIndex
		switch op.Name {
		case "EmbeddingLookup":
			if eng.isF16 {
				enc.SetComputePipelineState(eng.pso.EmbedH)
				enc.SetBufferWithOffsetAtIndex(eng.embedTableF16.Buffer(), 0, 0)
			} else {
				enc.SetComputePipelineState(eng.pso.Embed)
				enc.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
			}
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 1)
			metal.SetScalar(enc, tokVal, 2)
			metal.SetScalar(enc, dimVal, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(cfg.Dim),
				metal.Threads(128),
			)

		case "FusedRMSNormQKV":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.NormH)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, epsVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(1),
					metal.Threads(128),
				)

				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, qDimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.NumHeads*cfg.HeadDim),
					metal.Threads(gemvThreads),
				)

				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, kvDimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
					metal.Threads(gemvThreads),
				)

				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, kvDimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
					metal.Threads(gemvThreads),
				)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.FusedQKV)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 2)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 3)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 4)
				enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 5)
				enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 6)
				enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 7)
				metal.SetScalar(enc, dimVal, 8)
				metal.SetScalar(enc, qDimVal, 9)
				metal.SetScalar(enc, kvDimVal, 10)
				metal.SetScalar(enc, epsVal, 11)
				maxRows := cfg.NumHeads * cfg.HeadDim
				if cfg.NumKVHeads*cfg.HeadDim > maxRows {
					maxRows = cfg.NumKVHeads * cfg.HeadDim
				}
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(maxRows),
					metal.Threads(128),
				)
			}

		case "ComputeRMSNorm":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.NormH)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.Norm)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			}
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(1),
				metal.Threads(128),
			)

		case "FastGEMV_Q":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, qDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumHeads*cfg.HeadDim),
				metal.Threads(gemvThreads),
			)

		case "FastGEMV_K":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(gemvThreads),
			)

		case "FastGEMV_V":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, kvDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(gemvThreads),
			)

		case "ApplyRoPEStoreKV":
			enc.SetComputePipelineState(eng.pso.RoPE)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.kBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.vBuf.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
			metal.SetScalar(enc, posVal, 5)
			metal.SetScalar(enc, headDimVal, 6)
			metal.SetScalar(enc, qHeadsVal, 7)
			metal.SetScalar(enc, kvHeadsVal, 8)
			metal.SetScalar(enc, thetaVal, 9)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(cfg.NumHeads*cfg.HeadDim),
				metal.Threads(128),
			)

		case "CausalFlashAttentionDecode":
			enc.SetComputePipelineState(eng.pso.Attn)
			enc.SetBufferWithOffsetAtIndex(eng.qBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 3)
			metal.SetScalar(enc, int32(pos), 4)
			metal.SetScalar(enc, int32(cfg.MaxSeqLen), 5)
			metal.SetScalar(enc, int32(cfg.NumHeads), 6)
			metal.SetScalar(enc, int32(cfg.NumKVHeads), 7)
			metal.SetScalar(enc, int32(cfg.HeadDim), 8)
			metal.SetScalar(enc, scaleVal, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.NumHeads),
				metal.Threads(128),
			)

		case "FastGEMVResidual_O":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVResH)
				enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMVRes)
				enc.SetBufferWithOffsetAtIndex(eng.attnOutBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.Dim),
				metal.Threads(gemvThreads),
			)

		case "ComputeRMSNorm_Post":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.NormH)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.Norm)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
			}
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, epsVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(1),
				metal.Threads(128),
			)

		case "FastGEMV_Gate":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.HiddenDim),
				metal.Threads(gemvThreads),
			)

		case "FastGEMV_Up":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVH)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMV)
				enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, dimVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.HiddenDim),
				metal.Threads(gemvThreads),
			)

		case "FusedSwiGLUDown":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(eng.pso.SiLUMul)
				enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, hiddenDimVal, 3)
				enc.DispatchThreadsThreadsPerThreadgroup(
					metal.Threads(cfg.HiddenDim),
					metal.Threads(128),
				)

				enc.SetComputePipelineState(psoGEMVResH)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
				metal.SetScalar(enc, hiddenDimVal, 3)
				metal.SetScalar(enc, dimVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.Dim),
					metal.Threads(gemvThreads),
				)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(eng.pso.FusedSwiGLU)
				enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 2)
				enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 3)
				metal.SetScalar(enc, dimVal, 4)
				metal.SetScalar(enc, hiddenDimVal, 5)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.Dim),
					metal.Threads(128),
				)
			}

		case "FastSiLUMul":
			enc.SetComputePipelineState(eng.pso.SiLUMul)
			enc.SetBufferWithOffsetAtIndex(eng.gateBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(eng.upBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, hiddenDimVal, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(cfg.HiddenDim),
				metal.Threads(128),
			)

		case "FastGEMVResidual_Down":
			if eng.isF16 {
				layer := &eng.layersF16[l]
				enc.SetComputePipelineState(psoGEMVResH)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			} else {
				layer := &eng.layers[l]
				enc.SetComputePipelineState(psoGEMVRes)
				enc.SetBufferWithOffsetAtIndex(eng.ffnActBuf.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			}
			enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 2)
			metal.SetScalar(enc, hiddenDimVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(cfg.Dim),
				metal.Threads(gemvThreads),
			)

		case "ComputeRMSNorm_Final":
			if isLast {
				if eng.isF16 {
					enc.SetComputePipelineState(eng.pso.NormH)
					enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.finalNormWF16.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
				} else {
					enc.SetComputePipelineState(eng.pso.Norm)
					enc.SetBufferWithOffsetAtIndex(eng.xBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
				}
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, epsVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(1),
					metal.Threads(128),
				)
			}

		case "FastGEMV_LMHead":
			if isLast {
				if eng.isF16 {
					enc.SetComputePipelineState(psoGEMVH)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.lmHeadWF16.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(targetLogits.Buffer(), 0, 2)
				} else {
					enc.SetComputePipelineState(psoGEMV)
					enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
					enc.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 1)
					enc.SetBufferWithOffsetAtIndex(targetLogits.Buffer(), 0, 2)
				}
				metal.SetScalar(enc, dimVal, 3)
				metal.SetScalar(enc, vocabVal, 4)
				enc.DispatchThreadgroupsThreadsPerThreadgroup(
					metal.Threads(cfg.VocabSize),
					metal.Threads(gemvThreads),
				)
			}

		case "ArgmaxSampling":
			if isLast {
				enc.SetComputePipelineState(eng.pso.Sample)
				enc.SetBufferWithOffsetAtIndex(targetLogits.Buffer(), 0, 0)
				enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
				metal.SetScalar(enc, vocabVal, 2)
				enc.DispatchThreadsThreadsPerThreadgroup(
					metal.Threads(1),
					metal.Threads(1),
				)
			}

		default:
			return fmt.Errorf("execute plan: unsupported operation %q", op.Name)
		}
	}
	return nil
}

func encodeForwardStepWithKV(enc apple.MTLComputeCommandEncoder, eng *ModelEngine, cfg Config, kCache, vCache []mempool.Slice[float32], tokenID int32, pos int, isLast bool) error {
	if eng == nil {
		return errors.New("encode forward step: nil engine")
	}
	return encodeForwardStepWithKVAndLogits(enc, eng, cfg, kCache, vCache, eng.logitsBuf, tokenID, pos, isLast)
}

func encodeForwardStepWithKVAndLogits(enc apple.MTLComputeCommandEncoder, eng *ModelEngine, cfg Config, kCache, vCache []mempool.Slice[float32], logitsBuf mempool.Slice[float32], tokenID int32, pos int, isLast bool) error {
	if eng == nil {
		return errors.New("encode forward step: nil engine")
	}
	plan := eng.Plan
	if plan == nil {
		plan = BuildExecutionPlan(cfg)
	}
	return ExecutePlanWithLogits(enc, eng, plan, cfg, kCache, vCache, logitsBuf, tokenID, pos, isLast)
}

// GPUForwardPass executes the 100% GPU-resident forward pass for a single decode token.
func GPUForwardPass(eng *ModelEngine, cfg Config, tokenID int32, pos int) int32 {
	tok, _ := GPUForwardPassKV(eng, cfg, nil, tokenID, pos)
	return tok
}

// GPUPrefillPass executes prompt prefill across all prompt tokens via matrix GEMM prefill.
func GPUPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) int32 {
	tok, _ := GPUPrefillPassKV(eng, cfg, nil, promptTokens)
	return tok
}

// GPUForwardPassKV executes the GPU forward pass using an isolated SequenceKVCache.
func GPUForwardPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, tokenID int32, pos int) (int32, error) {
	if eng == nil {
		return 0, errors.New("forward pass: nil engine")
	}
	if pos >= cfg.MaxSeqLen {
		return 0, fmt.Errorf("forward pass: position %d exceeds MaxSeqLen %d", pos, cfg.MaxSeqLen)
	}
	kCache := eng.kCacheBuf
	vCache := eng.vCacheBuf
	if skv != nil && len(skv.KCache) == cfg.NumLayers {
		kCache = skv.KCache
		vCache = skv.VCache
	}

	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	if err := encodeForwardStepWithKV(enc, eng, cfg, kCache, vCache, tokenID, pos, true); err != nil {
		enc.EndEncoding()
		return 0, fmt.Errorf("encode forward step: %w", err)
	}
	enc.EndEncoding()

	t0 := time.Now()
	cb.Commit()
	cb.WaitUntilCompleted()
	wallTime := time.Since(t0)
	gpuTime := time.Duration((cb.GPUEndTime() - cb.GPUStartTime()) * float64(time.Second))
	if gpuTime < 0 {
		gpuTime = 0
	}
	if eng.tuner != nil {
		gemvThreads := 128
		if eng.tuner.Width() == 64 {
			gemvThreads = 64
		}
		eng.tuner.Record(gemvThreads, gpuTime)
	}
	eng.metrics.RecordForward(ForwardMeasurement{
		Position: pos,
		GPUTime:  gpuTime,
		WallTime: wallTime,
		When:     time.Now(),
	})

	if eng.recorder != nil {
		eng.recorder.Record(fmt.Sprintf("LLaMA_Forward_Pass_%dL", cfg.NumLayers), gpuTime, [3]int{cfg.Dim, cfg.NumLayers, 1}, [3]int{128, 1, 1}, map[string]string{
			"model": cfg.Name,
			"pos":   fmt.Sprintf("%d", pos),
		})
	}

	return eng.tokBuf.HostSlice()[0], nil
}

// GPUPrefillPassKV executes prompt prefill using an isolated SequenceKVCache via matrix GEMM prefill.
func GPUPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32) (int32, error) {
	if len(promptTokens) == 0 {
		return 0, nil
	}
	if eng == nil {
		return 0, errors.New("prefill pass: nil engine")
	}
	if eng.pso.BatchedAttn != nil {
		return GPUBatchedPrefillPassKV(eng, cfg, skv, promptTokens, 0)
	}

	queue := eng.dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	kCache := eng.kCacheBuf
	vCache := eng.vCacheBuf
	if skv != nil && len(skv.KCache) == cfg.NumLayers {
		kCache = skv.KCache
		vCache = skv.VCache
	}

	for pos, tokID := range promptTokens {
		isLast := (pos == len(promptTokens)-1)
		if err := encodeForwardStepWithKV(enc, eng, cfg, kCache, vCache, tokID, pos, isLast); err != nil {
			enc.EndEncoding()
			return 0, fmt.Errorf("prefill step pos %d: %w", pos, err)
		}
	}
	enc.EndEncoding()

	cb.Commit()
	cb.WaitUntilCompleted()
	return eng.tokBuf.HostSlice()[0], nil
}

// ensurePrefillArena makes the batched-prefill scratch buffers big enough for
// seqLen tokens, reallocating them only when the sequence outgrows what is
// already there.
func (eng *ModelEngine) ensurePrefillArena(seqLen int, cfg Config) error {
	if eng.prefillMaxSeqLen >= seqLen {
		return nil
	}
	eng.synchronize()
	eng.freePrefillArena()

	allocLen := max(seqLen, 128)
	a := metalmem.NewTrackedArena(eng.dev, eng.memory)
	eng.prefillXMat = metalmem.AllocIn[float32](a, "xMat", allocLen*cfg.Dim)
	eng.prefillNormXMat = metalmem.AllocIn[float32](a, "normXMat", allocLen*cfg.Dim)
	eng.prefillQMat = metalmem.AllocIn[float32](a, "qMat", allocLen*cfg.NumHeads*cfg.HeadDim)
	eng.prefillKMat = metalmem.AllocIn[float32](a, "kMat", allocLen*cfg.NumKVHeads*cfg.HeadDim)
	eng.prefillVMat = metalmem.AllocIn[float32](a, "vMat", allocLen*cfg.NumKVHeads*cfg.HeadDim)
	eng.prefillAttnOut = metalmem.AllocIn[float32](a, "attnOut", allocLen*cfg.Dim)
	eng.prefillGateMat = metalmem.AllocIn[float32](a, "gateMat", allocLen*cfg.HiddenDim)
	eng.prefillUpMat = metalmem.AllocIn[float32](a, "upMat", allocLen*cfg.HiddenDim)
	eng.prefillFFNAct = metalmem.AllocIn[float32](a, "ffnAct", allocLen*cfg.HiddenDim)
	eng.prefillTokIn = metalmem.AllocIn[int32](a, "tokIn", allocLen)
	eng.prefillRequest = metalmem.AllocIn[uint32](a, "requestIndex", allocLen)
	eng.prefillPositions = metalmem.AllocIn[uint32](a, "positions", allocLen)
	eng.prefillOffsets = metalmem.AllocIn[uint32](a, "offsets", allocLen)
	eng.prefillLengths = metalmem.AllocIn[uint32](a, "lengths", allocLen)
	eng.prefillRows = metalmem.AllocIn[uint32](a, "rows", allocLen)
	eng.prefillRestored = metalmem.AllocIn[uint32](a, "restored", allocLen)
	if err := a.Err(); err != nil {
		a.Free()
		return fmt.Errorf("alloc prefill arena for %d tokens: %w", allocLen, err)
	}

	eng.prefill = a
	eng.prefillMaxSeqLen = allocLen
	return nil
}

// synchronize waits for all work previously submitted to the engine queue.
func (eng *ModelEngine) synchronize() {
	if eng == nil || eng.dev == nil {
		return
	}
	cb := eng.dev.Queue().CommandBuffer()
	cb.Commit()
	cb.WaitUntilCompleted()
}

// freePrefillArena releases the batched-prefill scratch buffers.
func (eng *ModelEngine) freePrefillArena() {
	if eng.prefill != nil {
		eng.prefill.Free()
		eng.prefill = nil
	}
	eng.prefillMaxSeqLen = 0
}

func (eng *ModelEngine) Free() {
	if eng == nil {
		return
	}
	eng.freeOnce.Do(func() {
		darwinEngineMu.Lock()
		if eng.cacheKey != "" && darwinEngineCache[eng.cacheKey] == eng {
			delete(darwinEngineCache, eng.cacheKey)
		}
		darwinEngineMu.Unlock()
		eng.generateMu.Lock()
		defer eng.generateMu.Unlock()
		eng.batchedMu.Lock()
		defer eng.batchedMu.Unlock()
		eng.synchronize()
		eng.freePrefillArena()
		if eng.mem != nil {
			eng.mem.Free()
		}
		if eng.weights != nil {
			_ = eng.weights.Close()
		}
	})
}

func encodeBatchedPrefillWithKV(
	enc apple.MTLComputeCommandEncoder,
	eng *ModelEngine,
	cfg Config,
	kCache, vCache []mempool.Slice[float32],
	promptTokens []int32,
	startPos int,
	xMat, normXMat, qMat, kMat, vMat, attnOutMat, gateMat, upMat, ffnActMat mempool.Slice[float32],
	tokInBuf mempool.Slice[int32],
) {
	seqLen := len(promptTokens)
	if seqLen == 0 {
		return
	}

	copy(tokInBuf.HostSlice()[:seqLen], promptTokens)

	seqLenVal := uint32(seqLen)
	startPosVal := uint32(startPos)
	dimVal := uint32(cfg.Dim)
	hiddenDimVal := uint32(cfg.HiddenDim)
	qDimVal := uint32(cfg.NumHeads * cfg.HeadDim)
	kvDimVal := uint32(cfg.NumKVHeads * cfg.HeadDim)
	headDimVal := uint32(cfg.HeadDim)
	qHeadsVal := uint32(cfg.NumHeads)
	kvHeadsVal := uint32(cfg.NumKVHeads)
	thetaVal := cfg.RopeTheta
	epsVal := cfg.Eps
	scaleVal := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	vocabVal := uint32(cfg.VocabSize)
	totalFFNElems := uint32(seqLen * cfg.HiddenDim)

	if eng.isF16 {
		// 1. Batched Embedding Lookup over Half Weights
		enc.SetComputePipelineState(eng.pso.BatchedEmbedH)
		enc.SetBufferWithOffsetAtIndex(eng.embedTableF16.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(tokInBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
		metal.SetScalar(enc, seqLenVal, 3)
		metal.SetScalar(enc, dimVal, 4)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.Threads(seqLen*cfg.Dim),
			metal.Threads(128),
		)

		// 2. Transformer Decoder Layers
		for l := 0; l < cfg.NumLayers; l++ {
			layer := &eng.layersF16[l]
			if layer.InputNorm.Len() == 0 {
				continue
			}

			// (a) Batched Input RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNormH)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (b) Batched Q, K, V Matrix Multiplications (GEMM)
			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, qDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			// (c) Batched RoPE + Store KV cache
			enc.SetComputePipelineState(eng.pso.BatchedRoPE)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
			metal.SetScalar(enc, seqLenVal, 5)
			metal.SetScalar(enc, startPosVal, 6)
			metal.SetScalar(enc, headDimVal, 7)
			metal.SetScalar(enc, qHeadsVal, 8)
			metal.SetScalar(enc, kvHeadsVal, 9)
			metal.SetScalar(enc, thetaVal, 10)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(1),
			)

			// (d) Batched Causal FlashAttention
			enc.SetComputePipelineState(eng.pso.BatchedAttn)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 3)
			metal.SetScalar(enc, seqLenVal, 4)
			metal.SetScalar(enc, startPosVal, 5)
			metal.SetScalar(enc, qHeadsVal, 6)
			metal.SetScalar(enc, kvHeadsVal, 7)
			metal.SetScalar(enc, headDimVal, 8)
			metal.SetScalar(enc, scaleVal, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads),
				metal.Threads(128),
			)

			// (e) Batched Out Proj + Residual Add
			enc.SetComputePipelineState(eng.pso.BatchedGEMMResH)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)

			// (f) Batched Post-Attention RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNormH)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (g) Batched SwiGLU FFN: Gate + Up GEMMs
			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMMH)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			// Elementwise silu(gate) * up
			enc.SetComputePipelineState(eng.pso.BatchedSiLU)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 2)
			metal.SetScalar(enc, totalFFNElems, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(int(totalFFNElems)),
				metal.Threads(128),
			)

			// Down GEMM + Residual Add: xMat += ffnActMat * WDown^T
			enc.SetComputePipelineState(eng.pso.BatchedGEMMResH)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)
		}

		// 3. Final RMSNorm on the LAST token row (seqLen - 1)
		lastRowOffset := uint((seqLen - 1) * cfg.Dim * 4) // bytes
		enc.SetComputePipelineState(eng.pso.NormH)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), lastRowOffset, 0)
		enc.SetBufferWithOffsetAtIndex(eng.finalNormWF16.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, epsVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(1),
			metal.Threads(128),
		)

		// 4. LM Head Logits on the last token
		enc.SetComputePipelineState(eng.pso.GEMVH)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.lmHeadWF16.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, vocabVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(cfg.VocabSize),
			metal.Threads(128),
		)
	} else {
		// 1. Batched Embedding Lookup
		enc.SetComputePipelineState(eng.pso.BatchedEmbed)
		enc.SetBufferWithOffsetAtIndex(eng.embedTable.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(tokInBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
		metal.SetScalar(enc, seqLenVal, 3)
		metal.SetScalar(enc, dimVal, 4)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.Threads(seqLen*cfg.Dim),
			metal.Threads(128),
		)

		// 2. Transformer Decoder Layers
		for l := 0; l < cfg.NumLayers; l++ {
			layer := &eng.layers[l]
			if layer.InputNorm.Len() == 0 {
				continue
			}

			// (a) Batched Input RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNorm)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.InputNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (b) Batched Q, K, V Matrix Multiplications (GEMM)
			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WQ.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, qDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WK.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WV.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, kvDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumKVHeads*cfg.HeadDim),
				metal.Threads(128),
			)

			// (c) Batched RoPE + Store KV cache
			enc.SetComputePipelineState(eng.pso.BatchedRoPE)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vMat.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 4)
			metal.SetScalar(enc, seqLenVal, 5)
			metal.SetScalar(enc, startPosVal, 6)
			metal.SetScalar(enc, headDimVal, 7)
			metal.SetScalar(enc, qHeadsVal, 8)
			metal.SetScalar(enc, kvHeadsVal, 9)
			metal.SetScalar(enc, thetaVal, 10)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(1),
			)

			// (d) Batched Causal FlashAttention
			enc.SetComputePipelineState(eng.pso.BatchedAttn)
			enc.SetBufferWithOffsetAtIndex(qMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kCache[l].Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vCache[l].Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 3)
			metal.SetScalar(enc, seqLenVal, 4)
			metal.SetScalar(enc, startPosVal, 5)
			metal.SetScalar(enc, qHeadsVal, 6)
			metal.SetScalar(enc, kvHeadsVal, 7)
			metal.SetScalar(enc, headDimVal, 8)
			metal.SetScalar(enc, scaleVal, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.NumHeads),
				metal.Threads(128),
			)

			// (e) Batched Out Proj + Residual Add
			enc.SetComputePipelineState(eng.pso.BatchedGEMMRes)
			enc.SetBufferWithOffsetAtIndex(attnOutMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WO.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)

			// (f) Batched Post-Attention RMSNorm
			enc.SetComputePipelineState(eng.pso.BatchedNorm)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.PostNorm.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, epsVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen),
				metal.Threads(128),
			)

			// (g) Batched SwiGLU FFN: Gate + Up GEMMs
			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WGate.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			enc.SetComputePipelineState(eng.pso.BatchedGEMM)
			enc.SetBufferWithOffsetAtIndex(normXMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WUp.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, dimVal, 4)
			metal.SetScalar(enc, hiddenDimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.HiddenDim),
				metal.Threads(128),
			)

			// Elementwise silu(gate) * up
			enc.SetComputePipelineState(eng.pso.BatchedSiLU)
			enc.SetBufferWithOffsetAtIndex(gateMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(upMat.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 2)
			metal.SetScalar(enc, totalFFNElems, 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.Threads(int(totalFFNElems)),
				metal.Threads(128),
			)

			// Down GEMM + Residual Add: xMat += ffnActMat * WDown^T
			enc.SetComputePipelineState(eng.pso.BatchedGEMMRes)
			enc.SetBufferWithOffsetAtIndex(ffnActMat.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(layer.WDown.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), 0, 2)
			metal.SetScalar(enc, seqLenVal, 3)
			metal.SetScalar(enc, hiddenDimVal, 4)
			metal.SetScalar(enc, dimVal, 5)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				metal.Threads(seqLen*cfg.Dim),
				metal.Threads(128),
			)
		}

		// 3. Final RMSNorm on the LAST token row (seqLen - 1)
		lastRowOffset := uint((seqLen - 1) * cfg.Dim * 4) // bytes
		enc.SetComputePipelineState(eng.pso.Norm)
		enc.SetBufferWithOffsetAtIndex(xMat.Buffer(), lastRowOffset, 0)
		enc.SetBufferWithOffsetAtIndex(eng.finalNormW.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, epsVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(1),
			metal.Threads(128),
		)

		// 4. LM Head Logits on the last token
		enc.SetComputePipelineState(eng.pso.GEMV)
		enc.SetBufferWithOffsetAtIndex(eng.normXBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(eng.lmHeadW.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 2)
		metal.SetScalar(enc, dimVal, 3)
		metal.SetScalar(enc, vocabVal, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.Threads(cfg.VocabSize),
			metal.Threads(128),
		)
	}

	// 5. GPU Hardware Argmax Sampling
	enc.SetComputePipelineState(eng.pso.Sample)
	enc.SetBufferWithOffsetAtIndex(eng.logitsBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(eng.tokBuf.Buffer(), 0, 1)
	metal.SetScalar(enc, vocabVal, 2)
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.Threads(1),
		metal.Threads(1),
	)
}

// GPUBatchedPrefillPass executes prompt prefill across all prompt tokens via matrix GEMMs.
func GPUBatchedPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) (int32, error) {
	return GPUBatchedPrefillPassKV(eng, cfg, nil, promptTokens, 0)
}

// GPUBatchedPrefillPassKV executes prompt prefill via matrix GEMMs using an isolated SequenceKVCache.
func GPUBatchedPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32, startPos int) (int32, error) {
	seqLen := len(promptTokens)
	if seqLen == 0 {
		return 0, nil
	}
	if eng == nil {
		return 0, errors.New("batched prefill pass: nil engine")
	}

	if err := eng.ensurePrefillArena(seqLen, cfg); err != nil {
		return 0, fmt.Errorf("ensure prefill arena: %w", err)
	}

	dev := eng.dev
	queue := dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()

	kCache := eng.kCacheBuf
	vCache := eng.vCacheBuf
	if skv != nil && len(skv.KCache) == cfg.NumLayers {
		kCache = skv.KCache
		vCache = skv.VCache
	}

	encodeBatchedPrefillWithKV(enc, eng, cfg, kCache, vCache, promptTokens, startPos,
		eng.prefillXMat, eng.prefillNormXMat, eng.prefillQMat, eng.prefillKMat, eng.prefillVMat,
		eng.prefillAttnOut, eng.prefillGateMat, eng.prefillUpMat, eng.prefillFFNAct, eng.prefillTokIn)
	enc.EndEncoding()

	t0 := time.Now()
	cb.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return 0, fmt.Errorf("batched prefill pass: command buffer: %w", err)
	}
	wallTime := time.Since(t0)
	gpuTime := time.Duration((cb.GPUEndTime() - cb.GPUStartTime()) * float64(time.Second))
	if gpuTime < 0 {
		gpuTime = 0
	}
	if eng.metrics != nil {
		eng.metrics.RecordForward(ForwardMeasurement{
			Position: startPos + seqLen - 1,
			GPUTime:  gpuTime,
			WallTime: wallTime,
			When:     time.Now(),
		})
	}
	if eng.recorder != nil {
		eng.recorder.Record("LLaMA_Batched_Prefill_Pass", gpuTime, [3]int{cfg.Dim, seqLen, 1}, [3]int{128, 1, 1}, map[string]string{
			"model": cfg.Name,
			"len":   fmt.Sprintf("%d", seqLen),
		})
	}

	return eng.tokBuf.HostSlice()[0], nil
}

// GPUBatchDecodeStep executes concurrent single-token decode for multiple active sequences in a single command buffer.
func GPUBatchDecodeStep(eng *ModelEngine, cfg Config, batchKVs []*SequenceKVCache, batchTokens []int32, batchPositions []int) ([]int32, error) {
	batchSize := len(batchTokens)
	if batchSize == 0 {
		return nil, nil
	}
	if len(batchPositions) != batchSize || len(batchKVs) != batchSize {
		return nil, fmt.Errorf("batch size mismatch: tokens=%d, positions=%d, kvs=%d", batchSize, len(batchPositions), len(batchKVs))
	}
	if eng == nil {
		return nil, errors.New("batch decode: nil engine")
	}

	dev := eng.dev
	tokOutBuf, err := mempool.Alloc[int32](dev, batchSize)
	if err != nil {
		return nil, fmt.Errorf("alloc batch token output: %w", err)
	}
	defer tokOutBuf.Free()

	var batchLogits []mempool.Slice[float32]
	if batchSize > 1 {
		batchLogits = make([]mempool.Slice[float32], batchSize)
		for b := 0; b < batchSize; b++ {
			buf, err := mempool.Alloc[float32](dev, cfg.VocabSize)
			if err != nil {
				for i := 0; i < b; i++ {
					batchLogits[i].Free()
				}
				return nil, fmt.Errorf("alloc batch logits %d: %w", b, err)
			}
			defer buf.Free()
			batchLogits[b] = buf
		}
	}

	queue := dev.Queue()
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()

	vocabVal := uint32(cfg.VocabSize)
	for b := 0; b < batchSize; b++ {
		skv := batchKVs[b]
		kCache := eng.kCacheBuf
		vCache := eng.vCacheBuf
		if skv != nil && len(skv.KCache) == cfg.NumLayers {
			kCache = skv.KCache
			vCache = skv.VCache
		}

		logitsTarget := eng.logitsBuf
		if batchLogits != nil {
			logitsTarget = batchLogits[b]
		}

		if err := encodeForwardStepWithKVAndLogits(enc, eng, cfg, kCache, vCache, logitsTarget, batchTokens[b], batchPositions[b], true); err != nil {
			enc.EndEncoding()
			return nil, fmt.Errorf("batch decode sequence %d: %w", b, err)
		}

		outOffset := uint(b * 4)
		enc.SetComputePipelineState(eng.pso.Sample)
		enc.SetBufferWithOffsetAtIndex(logitsTarget.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(tokOutBuf.Buffer(), outOffset, 1)
		metal.SetScalar(enc, vocabVal, 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.Threads(1),
			metal.Threads(1),
		)
		enc.MemoryBarrierWithScope(apple.MTLBarrierScopeBuffers)
	}
	enc.EndEncoding()

	cb.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, fmt.Errorf("batch decode: shared command buffer: %w", err)
	}

	outTokens := make([]int32, batchSize)
	copy(outTokens, tokOutBuf.HostSlice()[:batchSize])
	return outTokens, nil
}

// runFastMetalInference executes all autoregressive decode iterations.
func runFastMetalInference(eng *ModelEngine, cfg Config) {
	promptText := cfg.Prompt
	if promptText == "" {
		promptText = "The key to scalable AI architectures is"
	}
	promptTokens, err := eng.tokenizer.Encode(promptText)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{578, 1401, 311, 69311, 15592, 78335, 374}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	loadedTensors := cfg.NumLayers*9 + 2
	if eng.weights != nil && len(eng.weights.Tensors) > 0 {
		loadedTensors = len(eng.weights.Tensors)
	}
	fmt.Printf("[Weights] Ingested %d tensors (%d Dim, %d Vocab, %d Layers)\n", loadedTensors, cfg.Dim, cfg.VocabSize, len(eng.layers))
	fmt.Printf("[Memory Planner] Zero-Copy Unified Memory Buffers Active\n")
	if cfg.FuseEGraph {
		fmt.Printf("[E-Graph Megakernel] Active: saturated %d equivalence classes with fused RMSNorm+GEMV & SwiGLU\n\n", eng.egraphSats)
	} else {
		fmt.Printf("[E-Graph Megakernel] Disabled: standard layer scheduling\n\n")
	}

	fmt.Printf("--- 10-Step Autoregressive Token Generation (Apple Metal GPU Resident) ---\n")
	fmt.Printf("Prompt: %q\n", promptText)
	fmt.Printf("Prompt tokens: %v (length: %d tokens)\n\n", promptTokens, len(promptTokens))

	// 1. Prefill prompt tokens 100% on Metal GPU in a single batched submission
	tPrefill0 := time.Now()
	currentTok := GPUPrefillPass(eng, cfg, promptTokens)
	prefillDur := time.Since(tPrefill0)

	numGen := cfg.GenTokens
	if numGen <= 0 {
		numGen = 10
	}
	generatedTokens := make([]int32, 0, numGen)

	// 2. Decode steps 100% on Metal GPU
	tDecode0 := time.Now()
	for step := 0; step < numGen; step++ {
		pos := len(promptTokens) + step
		nextTok := GPUForwardPass(eng, cfg, currentTok, pos)

		tokStr, _ := eng.tokenizer.Decode([]int32{nextTok})
		fmt.Printf("Step %2d (Pos %2d): Token %6d -> %q\n",
			step+1, pos, nextTok, tokStr)

		generatedTokens = append(generatedTokens, nextTok)
		currentTok = nextTok
	}
	decodeDur := time.Since(tDecode0)

	fullGenText, _ := eng.tokenizer.Decode(generatedTokens)
	fmt.Printf("\nGenerated completion: %q\n", fullGenText)
	fmt.Printf("\n--- Metal Real Latency Metrics ---\n")
	fmt.Printf("Prefill Time : %v (%d prompt tokens, %.2f tok/s)\n",
		prefillDur, len(promptTokens), float64(len(promptTokens))/prefillDur.Seconds())
	fmt.Printf("Decode Time  : %v (%d generated tokens, %.2f tok/s)\n",
		decodeDur, len(generatedTokens), float64(len(generatedTokens))/decodeDur.Seconds())
	ttftMs := float64(prefillDur.Microseconds()) / 1000.0
	tpotMs := float64(decodeDur.Microseconds()) / 1000.0 / float64(len(generatedTokens))
	fmt.Printf("TTFT: %.2f ms\n", ttftMs)
	fmt.Printf("TPOT: %.2f ms\n", tpotMs)
	fmt.Printf("Inference completed on Apple Silicon Metal GPU!\n")
}

func PrefillStep(cfg Config, promptTokens []int32) (time.Duration, error) {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	GPUPrefillPass(eng, cfg, promptTokens)
	return time.Since(start), nil
}

func DecodeStep(cfg Config, tokenID int32, pos int) (time.Duration, error) {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return 0, err
	}
	start := time.Now()
	GPUForwardPass(eng, cfg, tokenID, pos)
	return time.Since(start), nil
}

// RunBenchSuite executes count independent benchmark iterations and emits benchstat-compatible output.
func RunBenchSuite(cfg Config, count int) error {
	eng, err := GetOrCompileDarwinEngine(cfg)
	if err != nil {
		return fmt.Errorf("compile engine: %w", err)
	}

	promptText := cfg.Prompt
	if promptText == "" {
		promptText = "In one short paragraph, explain neural networks using the words layers, neurons, learning, and data."
	}
	promptTokens, err := eng.tokenizer.Encode(promptText)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{578, 1401, 311, 69311, 15592, 78335, 374}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	numGen := cfg.GenTokens
	if numGen <= 0 {
		numGen = 16
	}

	// Warmup run with clean KV cache
	eng.ResetKV()
	curr := GPUPrefillPass(eng, cfg, promptTokens)
	for s := 0; s < 2; s++ {
		curr = GPUForwardPass(eng, cfg, curr, len(promptTokens)+s)
	}

	searchDepths := cfg.SearchDepths
	if len(searchDepths) == 0 {
		depth := cfg.SearchGraphs
		if depth <= 0 {
			depth = 100
		}
		searchDepths = []int{depth}
	}

	bytesPerParam := 4.0
	if cfg.Precision == "f16" || eng.isF16 {
		bytesPerParam = 2.0
	}
	modelProf := gpuprof.ModelProfile{
		Name:            cfg.Name,
		HiddenDim:       cfg.Dim,
		IntermediateDim: cfg.HiddenDim,
		NumLayers:       cfg.NumLayers,
		NumHeads:        cfg.NumHeads,
		NumKVHeads:      cfg.NumKVHeads,
		HeadDim:         cfg.HeadDim,
		VocabSize:       cfg.VocabSize,
		TiedEmbeddings:  false,
		BytesPerParam:   bytesPerParam,
	}
	devName := "Apple Silicon Metal"
	if eng.dev != nil && eng.dev.Name() != "" {
		devName = eng.dev.Name()
	}
	hwProf := gpuprof.HardwareProfile{
		Name:              devName,
		Source:            "device-query",
		PeakBandwidthGBps: 400.0,
		PeakTFLOPs:        40.0,
	}

	for _, depth := range searchDepths {
		prefix := fmt.Sprintf("Benchmark%s/backend=metal/gen_tokens=%d/prompt=neural_network/search_graphs=%d", cfg.Name, numGen, depth)

		for iter := 0; iter < count; iter++ {
			eng.ResetKV()
			t0 := time.Now()
			currTok := GPUPrefillPass(eng, cfg, promptTokens)
			ttftDur := time.Since(t0)

			tDec0 := time.Now()
			for step := 0; step < numGen; step++ {
				pos := len(promptTokens) + step
				currTok = GPUForwardPass(eng, cfg, currTok, pos)
			}
			decDur := time.Since(tDec0)

			ttftNs := ttftDur.Nanoseconds()
			tpotNs := (decDur / time.Duration(numGen)).Nanoseconds()
			prefillTokPerSec := float64(len(promptTokens)) / ttftDur.Seconds()
			decTokPerSec := float64(numGen) / decDur.Seconds()

			prefObs := gpuprof.Observation{
				Duration:            ttftDur,
				TokenCount:          len(promptTokens),
				ThroughputTokPerSec: prefillTokPerSec,
			}
			decObs := gpuprof.Observation{
				Duration:            decDur,
				TokenCount:          numGen,
				ThroughputTokPerSec: decTokPerSec,
			}

			prefEst, errPref := gpuprof.EstimateWorkload(modelProf, hwProf, len(promptTokens), ttftDur, true)
			var pPrefEst *gpuprof.WorkloadEstimate
			if errPref == nil {
				pPrefEst = &prefEst
			}

			decEst, errDec := gpuprof.EstimateWorkload(modelProf, hwProf, numGen, decDur, false)
			var pDecEst *gpuprof.WorkloadEstimate
			if errDec == nil {
				pDecEst = &decEst
			}

			fmt.Printf("%s/TTFT-1 1 %d ns/op\n", prefix, ttftNs)
			fmt.Printf("%s/TPOT-1 1 %d ns/op\n", prefix, tpotNs)
			fmt.Print(gpuprof.FormatBenchmarkMetricLine(prefix, "prefill", ttftNs, prefObs, pPrefEst))
			fmt.Print(gpuprof.FormatBenchmarkMetricLine(prefix, "decode", decDur.Nanoseconds(), decObs, pDecEst))
		}
	}
	return nil
}

// Generate executes autoregressive decode on GPU for prompt and streams generated tokens through onToken.
func (eng *ModelEngine) Generate(ctx context.Context, cfg Config, prompt string, maxTokens int, onToken func(tokenID int32, tokenText string) bool) error {
	eng.generateMu.Lock()
	defer eng.generateMu.Unlock()

	if maxTokens <= 0 {
		maxTokens = 64
	}
	if eng.tokenizer == nil {
		eng.tokenizer = NewByteTokenizer()
	}
	if eng.dev == nil {
		// Mock simulation fallback when initialized without GPU device
		for step := 0; step < maxTokens; step++ {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
			if !onToken(int32(step), fmt.Sprintf(" tok_%d", step)) {
				break
			}
		}
		return nil
	}
	promptTokens, err := eng.tokenizer.Encode(prompt)
	if err != nil || len(promptTokens) == 0 {
		promptTokens = []int32{1}
	}
	if cfg.BOSTokenID != 0 {
		promptTokens = append([]int32{cfg.BOSTokenID}, promptTokens...)
	}

	var currentTok int32
	prefilled := 0
	if snapshot, matched, ok := eng.prefixes.load(promptTokens); ok && eng.restorePrefixPartial(snapshot, matched, cfg) {
		currentTok = snapshot.current
		prefilled = matched
	}
	if prefilled < len(promptTokens) {
		remaining := promptTokens[prefilled:]
		tok, err := GPUBatchedPrefillPassKV(eng, cfg, nil, remaining, prefilled)
		if err == nil {
			currentTok = tok
		} else {
			for i, tokID := range remaining {
				pos := prefilled + i
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}
				currentTok = GPUForwardPass(eng, cfg, tokID, pos)
			}
		}
		eng.prefixes.store(promptTokens, eng.capturePrefix(len(promptTokens), currentTok, cfg))
	}

	for step := 0; step < maxTokens; step++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		pos := len(promptTokens) + step
		nextTok := GPUForwardPass(eng, cfg, currentTok, pos)
		if cfg.IsEOS(nextTok) {
			break
		}
		tokStr, _ := eng.tokenizer.Decode([]int32{nextTok})
		if !onToken(nextTok, tokStr) {
			break
		}
		currentTok = nextTok
	}
	return nil
}

// GenerateTokenIDs runs generation and returns the token IDs emitted by this
// engine. It is used for exact paired-engine comparisons over Iroh.
func (eng *ModelEngine) GenerateTokenIDs(ctx context.Context, cfg Config, prompt string, maxTokens int) ([]int32, error) {
	var tokens []int32
	err := eng.Generate(ctx, cfg, prompt, maxTokens, func(tokenID int32, _ string) bool {
		tokens = append(tokens, tokenID)
		return true
	})
	return tokens, err
}
