//go:build linux

package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaRuntime owns one model mapping and one token timeline. All layer math
// runs on the device; only the final logits are copied to the host.
type cudaRuntime struct {
	power *powerControl
	*cudaFoundation
	model    *cudaModel
	arena    *cudamem.Arena
	batch    *cudaBatchResources
	state    cudaSessionState
	graphs   [cudaLayers]cudaLayerGraphs
	steering *cudaSteering

	// moeArtifacts holds the layers whose expert weights were repacked once at
	// load. A layer with one skips the per-group repack; see cudaMoEArtifact.
	moeArtifacts [cudaLayers]cudaMoEArtifact
	// denseArtifacts holds the layers whose dense Q8_0 weights were repacked
	// once at load. See cudaDenseArtifact.
	denseArtifacts [cudaLayers]cudaDenseArtifact
	// denseOutput is the output projection in the same aligned layout. It is
	// not part of cudaDenseArtifact because it is not per-layer: one 129280x4096
	// matrix, one launch per token, and the largest single weight read decode
	// still takes off the model mapping.
	denseOutput cudamem.Slice[byte]
	// ballast is device memory that is allocated and never read. It is the
	// control for the artifact residency question: if a run corrupts with
	// ballast in place, the artifacts are not what broke it.
	ballast                 []cudamem.Slice[byte]
	ballastGib              float64
	ballastSet              bool
	hcMath                  *cudaHC
	headObserver            outputHeadHistory
	ctx, pos                int
	prefillChunk            int
	logits                  []float32
	failed                  error
	batchProgress           func(int) error
	collectors              *profileCollectors
	dspark                  *cudaDSparkFeatures // active only inside its owning request Eval
	disableBlockDiagAttnOut bool
	disableSIMDExperts      bool
	disableGatherExperts    bool
	disableGraphs           bool
	disableAttentionHMMA    bool
	// borrowedFoundation is true for request-local runtimes created by the
	// dsv4 adapter. Such a runtime owns all mutable device state and its
	// stream, but shares immutable model resources with the backend runtime.
	borrowedFoundation bool
	requestOps         *cudaRequestOperations
	streamingRequest   *cudaStreamingRequest
	requestClosing     bool
	cleanupOps         *cudaChildCleanupOperations
}

type cudaLayerGraphs struct {
	qkv *cudagraph.Exec
	ffn *cudagraph.Exec
}

func (r *cudaRuntime) captureGraph(fn func()) (*cudagraph.Graph, error) {
	if r.streamingExperts != nil {
		return nil, errors.New("CUDA streaming graph capture unsupported")
	}
	mode := driver.StreamCaptureModeGlobal
	if r.borrowedFoundation {
		// Request-local resources are independent. Global capture would reject
		// another request's allocation or synchronous host copy, even though it
		// cannot access this request's nonblocking stream or mutable buffers.
		mode = driver.StreamCaptureModeThreadLocal
	}
	return cudagraph.CaptureWithMode(r.stream(), mode, fn)
}

type cudaSessionState struct {
	cudaTokenState
	layers [cudaLayers]cudaSessionLayer
}

// cudaTokenState holds one token row's transient layer activations. Layer
// history is session state and deliberately lives outside this type.
type cudaTokenState struct {
	hc                                                                [2]cudamem.Slice[float32]
	attnPre, attnNorm, qLora, qHeads, kvRaw, kvCur                    cudamem.Slice[float32]
	compressorKV, compressorGate, indexerKV, indexerGate              cudamem.Slice[float32]
	indexerQ, indexerWeights, indexerScratch, indexerScores           cudamem.Slice[float32]
	rawScores, rawOutput, mixedScores, mixedDenominators, mixedOutput cudamem.Slice[float32]
	allowed                                                           cudamem.Slice[uint32]
	attnLow, attnOut, ffnPre, ffnNorm, router, probs                  cudamem.Slice[float32]
	expertIDs                                                         cudamem.Slice[int32]
	expertW, sharedGate, sharedUp, sharedMid                          cudamem.Slice[float32]
	expertGate, expertUp, expertMid, expertDown                       cudamem.Slice[float32]
	ffnBlocks, sharedOut, routedOut, ffnOut, outputMixed, logits      cudamem.Slice[float32]
	q80Input, q81Input, q81Experts                                    cudamem.Slice[byte]
	q8KInput, q8KExperts                                              cudamem.Slice[byte]
}

type cudaSessionLayer struct {
	rawKV         cudamem.Slice[float32]
	attn, indexer cudaCompressedState
}

type cudaBatchDense struct {
	hc                                   [2]cudamem.Slice[float32]
	attnPre, qLora, qHeads, kvRaw, kvCur cudamem.Slice[float32]
	attnNorm                             cudamem.Slice[float32]
	compressorKV, compressorGate         cudamem.Slice[float32]
	indexerKV, indexerGate               cudamem.Slice[float32]
	pooled, indexerPooled                cudamem.Slice[float32]
	attnLow, attnOut                     cudamem.Slice[float32]
	attention                            cudamem.Slice[float32]
	ffnPre                               cudamem.Slice[float32]
	router, probs                        cudamem.Slice[float32]
	tokens                               cudamem.Slice[uint32]
	sharedGate, sharedUp, sharedOut      cudamem.Slice[float32]
	q81Attn, q81QLora, q81FFN            cudamem.Slice[byte]
	q80Heads, q80Low, q80Shared          cudamem.Slice[byte]
	expertIDs                            cudamem.Slice[int32]
	expertW                              cudamem.Slice[float32]
	expertGate, expertUp, expertDown     cudamem.Slice[float32]
	q4Mid                                cudamem.Slice[float32]
	q4Input, q4Experts                   cudamem.Slice[byte]
	d2rGate, d2rUp, d2rDown              cudamem.Slice[byte]
	d2rLayer                             int
	denseGate, denseUp, denseDown        cudamem.Slice[byte]
	denseAttnOutB, denseAttnOutA         cudamem.Slice[byte]
	denseAttnQA, denseAttnQB             cudamem.Slice[byte]
	denseAttnKV                          cudamem.Slice[byte]
	denseQ81Act                          cudamem.Slice[byte]
	// denseScratch holds the buffers the per-group dense repack writes. The
	// dense* fields above are rebound each layer to either these or the
	// layer's artifact, so a layer without one still repacks into scratch.
	denseScratch      cudaDenseArtifact
	denseSharedMid    cudamem.Slice[float32]
	profileMid        cudamem.Slice[float32]
	pairHalf          cudamem.Slice[uint16]
	denseD2RLayer     int
	q81MMQ            cudamem.Slice[byte]
	downQ8            cudamem.Slice[byte]
	d2rWork           cudamem.Slice[int32]
	d2rNItems         cudamem.Slice[int32]
	idsSrc            cudamem.Slice[int32]
	ffnOut            cudamem.Slice[float32]
	q81Experts        cudamem.Slice[byte]
	allowed           cudamem.Slice[uint32]
	tokenNComp        cudamem.Slice[uint32]
	mixedScores       cudamem.Slice[float32]
	mixedDenominators cudamem.Slice[float32]
	ttRecords         cudamem.Slice[byte]
	ttCounts          cudamem.Slice[uint32]
	ttRawMirror       cudamem.Slice[byte]
	ttCompMirror      cudamem.Slice[byte]
}

// cudaSharedFFNWidth is the shared expert's intermediate width, matching the
// per-token shared-gate and shared-up allocations in allocTokenState. A model
// whose shared FFN disagrees is rejected by encodeQ80MatvecQ81Batch's extent
// check rather than silently truncated.
const cudaSharedFFNWidth = 2048

type cudaBatchResources struct {
	closing    bool
	cleanupOps *cudaChildCleanupOperations
	arena      *cudamem.Arena
	denseArena *cudamem.Arena
	rows       []cudaTokenState
	hc         []*cudaHC
	hcBatch    *cudaHCBatch
	dense      cudaBatchDense
	grouping   *cudaMoEGrouping
	graphs     [cudaLayers]cudaLayerGraphs
}

// cudaPrefillGroupWidth is the number of tokens prefill holds resident at
// once. Batched projection kernels tile these rows internally, so the group
// width sets how often per-layer weights are re-read and repacked, not how the
// reductions inside a kernel associate: every width from 256 to 2048 produces
// the same logits, bit for bit. 2048 is the reference implementation's own
// single-chunk prefill width for a 2048-token prompt (ds4.c:14299), and it
// leaves 43 layer repacks where 256 left 344. Measured on GB10 against a fixed
// 2048-token prompt, one lock, canonical digest on every arm: 256 -> 90.9
// tok/s, 512 -> 109.2, 1024 -> 122.0, 2048 -> 122.7. The gain saturates by
// 1024; 2048 is kept because it costs nothing measured and matches the
// reference shape. A trailing group shorter than this runs without graph
// capture, which prefill's duty cycle makes a small price.
const cudaPrefillGroupWidth = 2048

// Attention kernels store rotated queries and keys before the head outputs.
const (
	cudaAttentionHeads   = 64 * cudaKVWidth
	cudaAttentionScratch = cudaAttentionHeads + cudaKVWidth
	cudaAttentionStride  = cudaAttentionScratch + cudaAttentionHeads
)

const (
	cudaPrefillQ81Chunk       = cudaPrefillGroupWidth
	cudaPrefillQ80Chunk       = cudaPrefillGroupWidth
	cudaPrefillBlockDiagChunk = cudaPrefillGroupWidth
	cudaPrefillF16PairChunk   = cudaPrefillGroupWidth
	cudaPrefillAttnOutChunk   = cudaPrefillGroupWidth
	cudaPrefillRouterChunk    = cudaPrefillGroupWidth
)

// Full groups tile exactly into chunks; the final group may be shorter.
// evalBatch routes single tokens to evalPosition.
const (
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillQ81Chunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillQ80Chunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillBlockDiagChunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillF16PairChunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillAttnOutChunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillRouterChunk)
	_ = uint(cudaPrefillGroupWidth - 2)
)

// chunkRows calls fn for each [lo, hi) span of n rows in steps of width. The
// final span may contain fewer than width rows.
func chunkRows(n, width int, fn func(lo, hi int) error) error {
	for lo := 0; lo < n; lo += width {
		if err := fn(lo, min(lo+width, n)); err != nil {
			return err
		}
	}
	return nil
}

var (
	_ sessionBackend = (*cudaRuntime)(nil)
	_ batchEvaluator = (*cudaRuntime)(nil)
)

// Shipped defaults for the three features that do not ship on yet:
// cudaShipD2RExperts sets disableD2RExperts, cudaShipDenseD2R sets
// disableDenseD2R, and cudaShipAttentionHMMA sets disableAttentionHMMA, all in
// newCUDARuntime. Flipping one constant to true ships that feature; nothing
// else changes. Every other behavior flag is a disable* field whose zero value
// means the feature is on, so all fourteen now read one way.
//
// These three were written enable* at first only because they arrived after
// the code they replace. The asymmetry recorded development order, not a
// decision, and it let a test measure a configuration no shipped binary could
// reach: setting an enable* field after construction selected a combination
// the constructor never produces.
//
// All three now ship on. Together they take prefill from 109 to 549 tok/s
// against the C oracle's 818, and they are what closes most of that gap: the
// two D2R arms carry 109 to 297, and the HMMA tokentile path carries 297 to
// 549. They ship with cudaMoEArtifactDefaultLayers at zero, which is the
// condition the safety measurement was taken under and not a separate choice;
// see that constant.
//
// Safety is measured against what shipped before, not against zero. In 30
// draws per arm, this configuration corrupted 1, the same as the previous
// shipped defaults, and it took 4 fewer stream-capture crashes because the
// D2R arms bypass most graph capture. The absolute rate is not established --
// 1 in 30 is [0.6%, 16.7%] with 95% confidence, which is a wide interval
// around a real defect that still has an open root cause. What is established
// is that turning these on does not make it worse.
const (
	cudaShipD2RExperts    = true
	cudaShipDenseD2R      = true
	cudaShipAttentionHMMA = true
)

type cudaRuntimeOptions struct {
	streaming      streamingOptions
	ballastGib     float64
	ballastSet     bool
	warmWeights    bool
	warmWeightsSet bool
}

func newCUDARuntime(path string, limit int) (*cudaRuntime, error) {
	return newCUDARuntimeWithOptions(path, limit, cudaRuntimeOptions{})
}

func newCUDARuntimeWithOptions(path string, limit int, options cudaRuntimeOptions) (_ *cudaRuntime, err error) {
	return newCUDARuntimeWithOperations(path, limit, options, defaultCUDARuntimeConstructorOperations())
}

func newCUDARuntimeWithOperations(path string, limit int, options cudaRuntimeOptions, ops cudaRuntimeConstructorOperations) (_ *cudaRuntime, err error) {
	if limit <= 1 || uint64(limit) > math.MaxUint32 {
		return nil, errors.New("cuda runtime: invalid context length")
	}
	r := &cudaRuntime{
		cleanupOps:    ops.cleanup,
		ctx:           limit,
		prefillChunk:  cudaPrefillGroupWidth,
		disableGraphs: os.Getenv("DS4_DISABLE_CUDA_GRAPH") == "1",
		ballastGib:    options.ballastGib,
		ballastSet:    options.ballastSet,
	}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, r)
		}
	}()
	r.cudaFoundation, err = ops.foundation(path, options)
	if err != nil {
		return nil, err
	}
	foundation := r.cudaFoundation
	if foundation.streamingExperts != nil {
		r.streamingRequest = foundation.streamingExperts.coordinator.request()
		r.disableGraphs = true
	}
	// Kept out of the literal above: two of the three are promoted from
	// cudaFoundation, and Go composite literals cannot set promoted fields.
	r.disableD2RExperts = !cudaShipD2RExperts
	r.disableDenseD2R = !cudaShipDenseD2R
	r.disableAttentionHMMA = !cudaShipAttentionHMMA
	r.disableRawAttentionBatch = os.Getenv("DS4_CUDA_DISABLE_RAW_ATTENTION_BATCH") == "1"
	r.model, err = ops.bind(foundation)
	if err != nil {
		return nil, err
	}
	if foundation.streamingExperts == nil {
		moeLayers := cudaMoEArtifactLayers()
		if foundation.resident.Len() > 0 {
			// Resident mode replaces every raw expert with an aligned artifact.
			moeLayers = cudaLayers
		}
		if err := ops.moe(r, moeLayers); err != nil {
			return nil, err
		}
	}
	if err := ops.dense(r, cudaDenseArtifactLayers()); err != nil {
		return nil, err
	}
	if err := ops.ballast(r); err != nil {
		return nil, err
	}
	if err := ops.allocate(r); err != nil {
		return nil, err
	}
	r.hcMath, err = ops.hc(foundation)
	if err != nil {
		return nil, err
	}
	r.logits = make([]float32, r.state.logits.Len())
	if err := ops.reset(r); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *cudaRuntime) allocState() {
	a, s := r.arena, &r.state
	r.allocTokenState(a, &s.cudaTokenState, "")
	r.allocLayerState(a, s)
	if a.Err() != nil {
		return
	}
	s.attnNorm = s.attnPre.Slice(cudaWidth, 2*cudaWidth)
	s.ffnNorm = s.ffnPre.Slice(cudaWidth, 2*cudaWidth)
	s.routedOut, s.sharedOut = s.ffnBlocks.Slice(0, cudaWidth), s.ffnBlocks.Slice(cudaWidth, 2*cudaWidth)
}

func (r *cudaRuntime) allocTokenState(a *cudamem.Arena, s *cudaTokenState, prefix string) {
	alloc := func(name string, n int) cudamem.Slice[float32] { return cudamem.AllocIn[float32](a, name, n) }
	s.hc[0], s.hc[1] = alloc(prefix+"hc-a", cudaHCWidth), alloc(prefix+"hc-b", cudaHCWidth)
	s.attnPre = alloc(prefix+"attention-pre", 2*cudaWidth)
	// Take views only after the allocation error has been checked below.
	s.qLora, s.qHeads = alloc(prefix+"query-lora", 1024), alloc(prefix+"query-heads", 64*cudaKVWidth)
	s.kvRaw, s.kvCur = alloc("kv-raw", cudaKVWidth), alloc("kv-current", cudaKVWidth)
	s.compressorKV, s.compressorGate = alloc("compressor-kv", 1024), alloc("compressor-gate", 1024)
	s.indexerKV, s.indexerGate = alloc("indexer-kv", 256), alloc("indexer-gate", 256)
	s.indexerQ, s.indexerWeights = alloc("indexer-query", 64*128), alloc("indexer-weights", 64)
	s.indexerScratch = alloc("indexer-query-scratch", 64*128)
	rawCap, maxComp := min(r.ctx, cudaRawCap), r.ctx/4+2
	s.indexerScores = alloc("indexer-scores", maxComp)
	s.rawScores, s.rawOutput = alloc("raw-scores", 64*rawCap), alloc("raw-output", 32768+512+32768)
	// mixedScores holds softmax weights between mixed-attention kernels.
	// mixedDenominators holds one denominator per head for the value kernel.
	s.mixedDenominators = alloc("mixed-denominators", 64)
	s.mixedScores, s.mixedOutput = alloc("mixed-scores", 64*(rawCap+maxComp)), alloc("mixed-output", 32768+512+32768)
	s.allowed = cudamem.AllocIn[uint32](a, "allowed-rows", maxComp)
	s.attnLow, s.attnOut = alloc("attention-low", 8192), alloc("attention-output", cudaWidth)
	s.ffnPre = alloc("ffn-pre", 2*cudaWidth)
	s.router, s.probs = alloc("router", 256), alloc("router-probabilities", 256)
	s.expertIDs = cudamem.AllocIn[int32](a, "expert-ids", 6)
	s.expertW = alloc("expert-weights", 6)
	s.sharedGate, s.sharedUp, s.sharedMid = alloc("shared-gate", 2048), alloc("shared-up", 2048), alloc("shared-mid", 2048)
	s.expertGate, s.expertUp, s.expertMid = alloc("expert-gate", 6*2048), alloc("expert-up", 6*2048), alloc("expert-mid", 6*2048)
	s.expertDown = alloc("expert-down", 6*cudaWidth)
	s.ffnBlocks, s.ffnOut = alloc("ffn-blocks", 2*cudaWidth), alloc("ffn-output", cudaWidth)
	s.outputMixed = alloc("output-mixed", 2*cudaWidth)
	s.logits = alloc("logits", int(r.model.output.weight.dimensions[1]))
	s.q80Input = cudamem.AllocIn[byte](a, "q80-input", 32768/32*36)
	s.q81Input = cudamem.AllocIn[byte](a, "q81-input", cudaWidth/32*36)
	s.q81Experts = cudamem.AllocIn[byte](a, "q81-experts", 6*2048/32*36)
	if cudaModelNeedsQ4(r.model) {
		s.q8KInput = cudamem.AllocIn[byte](a, "q8k-input", cudaWidth/256*292)
		s.q8KExperts = cudamem.AllocIn[byte](a, "q8k-experts", 6*2048/256*292)
	}
	if a.Err() != nil {
		return
	}
	s.attnNorm = s.attnPre.Slice(cudaWidth, 2*cudaWidth)
	s.ffnNorm = s.ffnPre.Slice(cudaWidth, 2*cudaWidth)
	s.routedOut, s.sharedOut = s.ffnBlocks.Slice(0, cudaWidth), s.ffnBlocks.Slice(cudaWidth, 2*cudaWidth)
}

func (r *cudaRuntime) allocLayerState(a *cudamem.Arena, s *cudaSessionState) {
	alloc := func(name string, n int) cudamem.Slice[float32] { return cudamem.AllocIn[float32](a, name, n) }
	rawCap := min(r.ctx, cudaRawCap)
	for i := range s.layers {
		l := &s.layers[i]
		l.rawKV = alloc(fmt.Sprintf("layer-%d-raw", i), rawCap*cudaKVWidth)
		if i < 2 {
			continue
		}
		ratio := 4
		if i%2 != 0 {
			ratio = 128
		}
		l.attn = r.allocCompressed(fmt.Sprintf("layer-%d-attention", i), ratio, 512)
		if ratio == 4 {
			l.indexer = r.allocCompressed(fmt.Sprintf("layer-%d-indexer", i), ratio, 128)
		}
	}
}

func (r *cudaRuntime) allocCompressed(name string, ratio, head int) cudaCompressedState {
	width, rows := head, ratio
	if ratio == 4 {
		width, rows = 2*head, 2*ratio
	}
	capacity := r.ctx/ratio + 2
	a := r.arena
	return cudaCompressedState{
		cache:      cudamem.AllocIn[float32](a, name+"-cache", capacity*head),
		stateKV:    cudamem.AllocIn[float32](a, name+"-kv", rows*width),
		stateScore: cudamem.AllocIn[float32](a, name+"-scores", rows*width),
		pooled:     cudamem.AllocIn[float32](a, name+"-pooled", head),
		headDim:    head, width: width, ratio: ratio, cap: capacity,
	}
}

func (r *cudaRuntime) Reset() error {
	if r != nil && errors.Is(r.failed, dsv4.ErrClosed) {
		return r.failed
	}
	if r == nil || r.arena == nil || r.device == nil {
		return errors.New("cuda runtime: closed")
	}
	r.headObserver.invalidate()
	r.failed = errors.New("cuda runtime: reset incomplete")
	if err := r.streamingRequest.reset(); err != nil {
		return fmt.Errorf("cuda runtime: streaming reset: %w", err)
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	for i := range r.state.layers {
		l := &r.state.layers[i]
		if err := l.rawKV.CopyFromHost(make([]float32, l.rawKV.Len())); err != nil {
			return err
		}
		for _, c := range []*cudaCompressedState{&l.attn, &l.indexer} {
			if c.cap == 0 {
				continue
			}
			for _, b := range []cudamem.Slice[float32]{c.cache, c.stateKV, c.pooled} {
				if err := b.CopyFromHost(make([]float32, b.Len())); err != nil {
					return err
				}
			}
			v := make([]float32, c.stateScore.Len())
			for j := range v {
				v[j] = float32(math.Inf(-1))
			}
			if err := c.stateScore.CopyFromHost(v); err != nil {
				return err
			}
			c.n = 0
		}
	}
	allowed := make([]uint32, r.state.allowed.Len())
	for i := range allowed {
		allowed[i] = 1
	}
	if err := r.state.allowed.CopyFromHost(allowed); err != nil {
		return err
	}
	if r.batch != nil {
		for i := range r.batch.rows {
			if err := r.batch.rows[i].allowed.CopyFromHost(allowed); err != nil {
				return err
			}
		}
		if err := r.batch.destroyGraphs(); err != nil {
			return err
		}
		r.batch.dense.denseD2RLayer = -1
		r.batch.dense.d2rLayer = -1
	}
	clear(r.logits)
	r.pos, r.failed = 0, nil
	return nil
}

func (r *cudaRuntime) Close() error {
	if r == nil {
		return nil
	}
	// Any partial teardown permanently closes execution admission.
	r.failed = errors.Join(r.failed, dsv4.ErrClosed, dsv4.ErrDesynchronized)
	if err := r.streamingRequest.Close(); err != nil {
		return fmt.Errorf("cuda runtime: streaming close: %w", err)
	}
	ops := r.cleanupOperations()
	if r.cudaFoundation != nil && r.device != nil && !r.requestClosing {
		if err := ops.synchronize(r); err != nil {
			return err
		}
		if !r.borrowedFoundation {
			r.reportDenseArtifactDigest("final")
		}
		// Completion permits retries after a terminal stream release.
		r.requestClosing = true
	}
	if r.steering != nil {
		if err := r.closeSteering(func(g *cudaSteering) error { return g.close() }); err != nil {
			return err
		}
	}
	// Request runtimes borrow the immutable artifact buffers and the output
	// artifact from their foundation.  They must release only their mutable
	// state; freeing these slices here would invalidate the backend for every
	// other request.  The owning runtime remains responsible for them.
	if !r.borrowedFoundation {
		for i := range r.moeArtifacts {
			if err := r.moeArtifacts[i].freeWith(ops.bytes); err != nil {
				return err
			}
		}
		for i := range r.denseArtifacts {
			if err := r.denseArtifacts[i].freeWith(ops.bytes); err != nil {
				return err
			}
		}
		if r.denseOutput.Len() > 0 {
			if err := ops.bytes(&r.denseOutput); err != nil {
				return err
			}
			r.denseOutput = cudamem.Slice[byte]{}
		}
		for i := range r.ballast {
			if err := freeCUDAByteSlices([]*cudamem.Slice[byte]{&r.ballast[i]}, ops.bytes); err != nil {
				return err
			}
		}
	}
	if r.ballast != nil {
		r.ballast = nil
	}
	for i := range r.graphs {
		if r.graphs[i].qkv != nil {
			if err := ops.graph(r.graphs[i].qkv); err != nil {
				return err
			}
			r.graphs[i].qkv = nil
		}
		if r.graphs[i].ffn != nil {
			if err := ops.graph(r.graphs[i].ffn); err != nil {
				return err
			}
			r.graphs[i].ffn = nil
		}
	}
	if r.hcMath != nil {
		if err := ops.hc(r.hcMath); err != nil {
			return err
		}
		r.hcMath = nil
	}
	if r.arena != nil {
		if err := ops.arena(r.arena); err != nil {
			return err
		}
		r.arena = nil
	}
	if r.batch != nil {
		if err := ops.batch(r.batch); err != nil {
			return err
		}
		r.batch = nil
	}
	if r.borrowedFoundation && r.cudaFoundation != nil && r.workStream != nil {
		if err := r.requestOperations().destroy(r); err != nil {
			return err
		}
		r.workStream = nil
	}
	if !r.borrowedFoundation && r.cudaFoundation != nil {
		if err := ops.foundation(r.cudaFoundation); err != nil {
			return err
		}
	}
	// A borrowed foundation belongs to the backend and must remain usable for
	// subsequent states.  Clear only this runtime's reference in either case.
	r.cudaFoundation = nil
	return nil
}

func (b *cudaBatchResources) destroyGraphs() error {
	if b == nil {
		return nil
	}
	ops := b.cleanupOperations()
	for i := range b.graphs {
		if b.graphs[i].qkv != nil {
			if err := ops.graph(b.graphs[i].qkv); err != nil {
				return err
			}
			b.graphs[i].qkv = nil
		}
		if b.graphs[i].ffn != nil {
			if err := ops.graph(b.graphs[i].ffn); err != nil {
				return err
			}
			b.graphs[i].ffn = nil
		}
	}
	return nil
}

func (b *cudaBatchResources) Close() error {
	if b == nil {
		return nil
	}
	b.closing = true
	ops := b.cleanupOperations()
	if err := b.destroyGraphs(); err != nil {
		return err
	}
	if b.grouping != nil {
		if err := ops.grouping(b.grouping); err != nil {
			return err
		}
		b.grouping = nil
	}
	if b.hcBatch != nil {
		if err := ops.hcBatch(b.hcBatch); err != nil {
			return err
		}
		b.hcBatch = nil
	}
	for i, h := range b.hc {
		if h == nil {
			continue
		}
		if err := ops.hc(h); err != nil {
			return err
		}
		b.hc[i] = nil
	}
	if err := freeCUDAByteSlices([]*cudamem.Slice[byte]{&b.dense.d2rGate, &b.dense.d2rUp, &b.dense.d2rDown}, ops.bytes); err != nil {
		return err
	}
	if b.arena != nil {
		if err := ops.arena(b.arena); err != nil {
			return err
		}
		b.arena = nil
	}
	if b.denseArena != nil {
		if err := ops.arena(b.denseArena); err != nil {
			return err
		}
		b.denseArena = nil
	}
	return nil
}

func (r *cudaRuntime) Logits() []float32 {
	if r == nil || r.arena == nil || r.pos == 0 || r.failed != nil {
		return nil
	}
	return r.logits
}

func (r *cudaRuntime) Eval(token, position int) error { return r.evalPosition(token, position, true) }

func (r *cudaRuntime) SetPrefillChunk(chunk int) error {
	if chunk < 256 || chunk > cudaPrefillGroupWidth || chunk%256 != 0 {
		return fmt.Errorf("cuda runtime: prefill chunk %d is not a multiple of 256 in [256,%d]", chunk, cudaPrefillGroupWidth)
	}
	r.prefillChunk = chunk
	return nil
}

func (r *cudaRuntime) EvalBatch(tokens []int, position int) error {
	if r == nil || r.arena == nil {
		return errors.New("cuda runtime: closed")
	}
	if r.failed != nil {
		return fmt.Errorf("cuda runtime: reset required: %w", r.failed)
	}
	if position != r.pos || position < 0 || position > r.ctx || len(tokens) > r.ctx-position {
		return fmt.Errorf("cuda runtime: batch at position %d exceeds timeline", position)
	}
	for _, token := range tokens {
		if token < 0 || uint64(token) >= r.model.output.embedding.dimensions[1] {
			return fmt.Errorf("cuda runtime: token %d out of range", token)
		}
	}
	if r.power != nil {
		previous := r.power.prefill
		r.power.prefill = true
		defer func() { r.power.prefill = previous }()
	}
	if r.streamingExperts != nil {
		for i, token := range tokens {
			if err := r.evalPosition(token, position+i, i == len(tokens)-1); err != nil {
				return err
			}
			if r.batchProgress != nil {
				if err := r.batchProgress(i + 1); err != nil {
					return err
				}
			}
		}
		return nil
	}
	width := r.prefillChunk
	if width == 0 {
		width = cudaPrefillGroupWidth
	}
	i := 0
	// Keep cache updates token ordered while batching the dense front end of
	// each chunk. Only the final chunk needs the vocabulary projection.
	for i < len(tokens) {
		n := min(width, len(tokens)-i)
		if n == 1 {
			if err := r.evalPosition(tokens[i], position+i, true); err != nil {
				return err
			}
			i++
			if r.batchProgress != nil {
				return r.batchProgress(i)
			}
			return nil
		}
		if err := r.evalBatch(tokens[i:i+n], position+i, i+n == len(tokens)); err != nil {
			return err
		}
		i += n
		if r.batchProgress != nil {
			if err := r.batchProgress(i); err != nil {
				return err
			}
		}
	}
	return nil
}

// scalarHCPath reports whether evalLayerBatch takes its per-token cudaHC
// fallbacks for b. Allocating those handles and choosing to use them read this
// one predicate, so the two cannot drift: a batch whose handles were skipped
// can never reach a branch that dereferences them.
func (r *cudaRuntime) scalarHCPath(b *cudaBatchResources) bool {
	return b == nil || b.hcBatch == nil || r.disableGatherExperts || r.disableSIMDExperts || r.disableBlockDiagAttnOut
}

// activeHCBatch keeps qualification fallbacks on their scalar HC path even
// when a previous evaluation allocated batch resources.
func (r *cudaRuntime) activeHCBatch() *cudaHCBatch {
	if r.scalarHCPath(r.batch) {
		return nil
	}
	return r.batch.hcBatch
}

// The three gates below ship on, so each one reads a DS4_CUDA_DISABLE_ name
// before anything else. The other eleven disable* fields need no such door,
// because a test ablates them by writing the field; these three are reached at
// artifact-build time, before a test can write anything, and they ship on over
// a corruption bug whose root cause is still open. An operator who hits it
// needs a way back to the previous behavior that does not involve a rebuild.
func (r *cudaRuntime) d2rEnabled() bool {
	if os.Getenv("DS4_CUDA_DISABLE_MOE_D2R") == "1" {
		return false
	}
	return !r.disableD2RExperts || os.Getenv("DS4_CUDA_MOE_D2R") == "1"
}

// moeD2R reports whether layer index's routed experts run the D2R GEMMs:
// they are enabled, and the layer has an artifact or the batch has scratch
// to repack into. The batch has none when every layer has an artifact.
func (r *cudaRuntime) moeD2R(d *cudaBatchDense, index int) bool {
	return r.model.layers[index].expertGate.typeID == typeIQ2XXS && r.d2rEnabled() && (r.moeArtifacts[index].built() || d.d2rGate.Len() > 0)
}

func (r *cudaRuntime) attentionHMMAEnabled() bool {
	if r == nil {
		return false
	}
	if os.Getenv("DS4_CUDA_DISABLE_ATTN_HMMA") == "1" {
		return false
	}
	return !r.disableAttentionHMMA || os.Getenv("DS4_CUDA_ATTENTION_HMMA") == "1"
}

// attnQKVD2REnabled reports whether the query and key/value projections run
// through the tensor-core GEMM. It follows the dense D2R gate, since it reuses
// that path's activation scratch and repack kernel.
func (r *cudaRuntime) attnQKVD2REnabled() bool {
	return r.denseD2REnabled() && !r.disableAttnQKVD2R
}

func (r *cudaRuntime) denseD2REnabled() bool {
	if r == nil {
		return false
	}
	if os.Getenv("DS4_CUDA_DISABLE_DENSE_D2R") == "1" {
		return false
	}
	return !r.disableDenseD2R || os.Getenv("DS4_CUDA_DENSE_D2R") == "1"
}

// denseD2REnvelopeShape reports whether a dense Q8_0 GEMM of this shape is one
// the reference routes to its own dense kernel. C's gate is a shape envelope,
// ds4_cuda.cu:14862-14863:
//
//	if (aligned && n_tok >= 512u && out_dim >= 2048u &&
//	    in_dim <= 4096u && cuda_use_mmq()) {
//
// The aligned conjunct is in_dim%1024 == 0 && out_dim%128 == 0 and a runtime
// enable (:14846-14853), both of which encodeDenseQ8D2R already demands of
// every caller here, so only the three inequalities discriminate between our
// sites. outDim is C's out_dim, nTok its n_tok, inDim its in_dim.
//
// Our own gate carries no envelope; cudaDenseD2REnvelope selects the arm that
// adds one.
func denseD2REnvelopeShape(outDim, nTok, inDim int) bool {
	return nTok >= 512 && outDim >= 2048 && inDim <= 4096
}

// cudaDenseD2REnvelope selects which dense call sites honour
// denseD2REnvelopeShape. Three arms, chosen so the two blocks that hold every
// envelope-failing site can be measured apart rather than merged:
//
//	unset     every site takes dense D2R (what we ship)
//	"attnout" only the attention-output block honours the envelope
//	"1"       every block honours it
//
// The attention-output block is the whole of its own effect: both of its
// shapes fail the envelope (attnOutA out_dim=1024, attnOutB in_dim=8192),
// where the query/key-value block is mixed (attnQA and attnKV fail, attnQB
// passes). Separating the two keeps the outcome classes disjoint.
func cudaDenseD2REnvelope() (qkv, attnOut bool) {
	switch os.Getenv("DS4_CUDA_DENSE_D2R_ENVELOPE") {
	case "1":
		return true, true
	case "attnout":
		return false, true
	}
	return false, false
}

// evalBatch spans the chunk rather than each token in it: the batched
// front end evaluates all of them together, so a per-token interval here
// would report the chunk's duration once per token.
func (r *cudaRuntime) evalBatch(tokens []int, position int, logits bool) (err error) {
	return r.evalBatchReadout(tokens, position, logits, nil)
}

// evalBatchReadout optionally reads every vocabulary row. The ordinary prefill
// caller still projects only the terminal row; speculative verification needs
// the preceding rows as well, without evaluating the transformer again.
func (r *cudaRuntime) evalBatchReadout(tokens []int, position int, logits bool, readout *[][]float32) (err error) {
	defer beginTraced(levelToken, "batch", nil).end()
	// The last phase in the group has no following mark to close it.
	defer closePhaseSpan()
	if len(tokens) < 2 || len(tokens) > cudaPrefillGroupWidth {
		return errors.New("cuda runtime: dense batch has invalid length")
	}
	if r.power != nil {
		previous := r.power.prefill
		r.power.prefill = true
		defer func() { r.power.prefill = previous }()
	}
	if os.Getenv("DS4_HEAD_OBSERVER_PREFIX") != "" {
		done, beginErr := r.headObserver.begin(position, tokens)
		if beginErr != nil {
			return beginErr
		}
		defer func() { done(err == nil) }()
	}
	defer func() {
		if err != nil {
			r.failed = err
		}
	}()
	if err := r.ensureBatch(func() (*cudaBatchResources, error) {
		return newCUDABatchResources(r, cudaPrefillGroupWidth)
	}); err != nil {
		return err
	}
	rows := make([]*cudaTokenState, len(tokens))
	hc := make([]*cudaHC, len(tokens))
	for i := 0; i < len(tokens)-1; i++ {
		rows[i], hc[i] = &r.batch.rows[i], r.batch.hc[i]
	}
	rows[len(tokens)-1], hc[len(tokens)-1] = &r.state.cudaTokenState, r.hcMath
	if r.activeHCBatch() != nil {
		// Rebind only local states. The persistent decode buffers retain their
		// addresses, including any addresses captured by a decode graph.
		local := make([]cudaTokenState, len(rows))
		for i := range rows {
			local[i] = *rows[i]
			for j := range 2 {
				local[i].hc[j] = r.batch.dense.hc[j].Slice(i*cudaHCWidth, (i+1)*cudaHCWidth)
			}
			rows[i] = &local[i]
		}
	}
	tokensU32 := make([]uint32, len(tokens))
	for i, token := range tokens {
		tokensU32[i] = uint32(token)
	}
	if err := r.batch.dense.tokens.Slice(0, len(tokens)).CopyFromHost(tokensU32); err != nil {
		return err
	}
	if r.activeHCBatch() != nil {
		if err := r.encodeEmbeddingRows(r.model.output.embedding, r.batch.dense.tokens, len(tokens), r.batch.dense.hc[0]); err != nil {
			return err
		}
	} else {
		for i, token := range tokensU32 {
			for stream := range 4 {
				if err := r.encodeEmbeddingRow(r.model.output.embedding, token, rows[i].hc[0].Slice(stream*cudaWidth, (stream+1)*cudaWidth)); err != nil {
					return err
				}
			}
		}
	}
	for layer := range r.model.layers {
		started := r.power.start()
		if err := r.evalLayerBatch(rows, hc, layer, tokens, position); err != nil {
			return fmt.Errorf("cuda runtime: layer %d: %w", layer, err)
		}
		if r.dspark != nil && r.dspark.grouped != nil {
			if err := r.dspark.captureGrouped(layer, position, rows); err != nil {
				return fmt.Errorf("cuda runtime: grouped support feature layer %d: %w", layer, err)
			}
		}
		if err := r.power.finishLayer(layer, started, func() error { return syncStream(r.stream()) }); err != nil {
			return fmt.Errorf("cuda runtime: power layer %d: %w", layer, err)
		}
	}
	last := rows[len(rows)-1]
	if r.activeHCBatch() != nil {
		// Subsequent scalar decode starts from the terminal token's HC state.
		for j := range 2 {
			if err := r.state.hc[j].CopyDeviceToDeviceAsync(last.hc[j], r.stream()); err != nil {
				return fmt.Errorf("cuda runtime: copy terminal HC state: %w", err)
			}
		}
	}
	if err := r.markCUDAPhase("group-tail", position+len(tokens)-1, -1); err != nil {
		return err
	}
	if logits {
		headRows := rows[len(rows)-1:]
		if readout != nil {
			headRows = rows
		}
		for headIndex, row := range headRows {
			if err := r.hcMath.pre(row.hc[0], r.model.output.hcFn, r.model.output.hcScale, r.model.output.hcBase, r.model.output.norm, row.outputMixed, true); err != nil {
				return err
			}
			headPosition := position + len(tokens) - 1
			if readout != nil {
				headPosition = position + headIndex
			}
			if err := r.encodeObservedOutputHead(row.outputMixed, row.logits, headPosition); err != nil {
				return err
			}
		}
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	if logits {
		if readout != nil {
			values := make([][]float32, len(rows))
			for i, row := range rows {
				values[i] = make([]float32, row.logits.Len())
				if err := row.logits.CopyToHost(values[i]); err != nil {
					return err
				}
				if err := checkFiniteLogits("cuda speculative row", values[i]); err != nil {
					return err
				}
			}
			*readout = values
		}
		if err := last.logits.CopyToHost(r.logits); err != nil {
			return err
		}
		if err := checkFiniteLogits("cuda runtime", r.logits); err != nil {
			return err
		}
	}
	r.pos += len(tokens)
	return nil
}

func newCUDABatchResources(r *cudaRuntime, width int) (_ *cudaBatchResources, err error) {
	if r == nil || r.device == nil || width < 2 {
		return nil, errors.New("cuda runtime: invalid batch allocation")
	}
	// The row states are about forty small buffers each, so a group of 2048
	// costs ninety thousand driver allocations -- six hundred milliseconds of
	// the first EvalBatch, and the whole of its gap over device time. They have
	// one lifetime, so one block allocation per quarter gigabyte serves them
	// all. The dense arena keeps the plain one: its buffers are few and large,
	// and packing them would only round each up.
	arena := cudamem.NewChunkedArena(r.device, cudaBatchArenaChunk)
	if r.disableChunkedArena {
		arena = cudamem.NewArena(r.device)
	}
	b := &cudaBatchResources{
		arena:      arena,
		denseArena: cudamem.NewArena(r.device),
		rows:       make([]cudaTokenState, width-1),
		hc:         make([]*cudaHC, width-1),
	}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, b)
		}
	}()
	for i := range b.rows {
		r.allocTokenState(b.arena, &b.rows[i], fmt.Sprintf("batch-%d-", i))
	}
	rawCap, maxComp := min(r.ctx, cudaRawCap), r.ctx/4+2
	scoreStride := 64 * (rawCap + maxComp)
	alloc := func(name string, n int) cudamem.Slice[float32] {
		return cudamem.AllocIn[float32](b.denseArena, name, n)
	}
	b.dense = cudaBatchDense{
		attnPre:           alloc("batch-dense-attention-pre", width*2*cudaWidth),
		qLora:             alloc("batch-dense-query-lora", width*1024),
		qHeads:            alloc("batch-dense-query-heads", width*64*cudaKVWidth),
		kvRaw:             alloc("batch-dense-kv-raw", width*cudaKVWidth),
		kvCur:             alloc("batch-dense-kv-current", width*cudaKVWidth),
		attnNorm:          alloc("batch-dense-attention-norm", width*cudaWidth),
		compressorKV:      alloc("batch-dense-compressor-kv", width*1024),
		compressorGate:    alloc("batch-dense-compressor-gate", width*1024),
		indexerKV:         alloc("batch-dense-indexer-kv", width*256),
		indexerGate:       alloc("batch-dense-indexer-gate", width*256),
		pooled:            alloc("batch-dense-pooled", width/4*512),
		indexerPooled:     alloc("batch-dense-indexer-pooled", width/4*128),
		attention:         alloc("batch-dense-attention", width*cudaAttentionStride),
		attnLow:           alloc("batch-dense-attention-low", width*8192),
		attnOut:           alloc("batch-dense-attention-output", width*cudaWidth),
		ffnPre:            alloc("batch-dense-ffn-pre", width*2*cudaWidth),
		router:            alloc("batch-dense-router", width*256),
		probs:             alloc("batch-dense-probs", width*256),
		tokens:            cudamem.AllocIn[uint32](b.denseArena, "batch-dense-tokens", width),
		sharedGate:        alloc("batch-dense-shared-gate", width*cudaSharedFFNWidth),
		sharedUp:          alloc("batch-dense-shared-up", width*cudaSharedFFNWidth),
		sharedOut:         alloc("batch-dense-shared-output", width*cudaWidth),
		q81Attn:           cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-attention", width*cudaWidth/32*36),
		q81QLora:          cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-query-lora", width*1024/32*36),
		q81FFN:            cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-ffn", width*cudaWidth/32*36),
		q80Heads:          cudamem.AllocIn[byte](b.denseArena, "batch-dense-q80-attention-heads", width*32768/32*36),
		q80Low:            cudamem.AllocIn[byte](b.denseArena, "batch-dense-q80-attention-low", width*8192/32*36),
		q80Shared:         cudamem.AllocIn[byte](b.denseArena, "batch-dense-q80-shared", width*cudaSharedFFNWidth/32*36),
		expertIDs:         cudamem.AllocIn[int32](b.denseArena, "batch-expert-ids", width*6),
		expertW:           alloc("batch-expert-weights", width*6),
		expertGate:        alloc("batch-expert-gate", width*6*2048),
		expertUp:          alloc("batch-expert-up", width*6*2048),
		expertDown:        alloc("batch-expert-down", width*6*cudaWidth),
		q81MMQ:            cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-mmq", (cudaWidth/128)*width*144),
		downQ8:            cudamem.AllocIn[byte](b.denseArena, "batch-dense-down-q8", (2048/128)*width*6*144),
		d2rWork:           cudamem.AllocIn[int32](b.denseArena, "batch-dense-d2r-work", (width*6+31)/32+256),
		d2rNItems:         cudamem.AllocIn[int32](b.denseArena, "batch-dense-d2r-nitems", 1),
		idsSrc:            cudamem.AllocIn[int32](b.denseArena, "batch-dense-ids-src", width*6),
		ffnOut:            alloc("batch-dense-ffn-output", width*cudaWidth),
		pairHalf:          cudamem.AllocIn[uint16](b.denseArena, "batch-dense-pair-half", width*cudaWidth),
		q81Experts:        cudamem.AllocIn[byte](b.denseArena, "batch-q81-experts", width*6*2048/32*36),
		allowed:           cudamem.AllocIn[uint32](b.denseArena, "batch-dense-allowed", width*maxComp),
		tokenNComp:        cudamem.AllocIn[uint32](b.denseArena, "batch-dense-token-ncomp", width),
		mixedScores:       alloc("batch-dense-mixed-scores", width*scoreStride),
		mixedDenominators: alloc("batch-dense-mixed-denominators", width*64),
		ttRecords:         cudamem.AllocIn[byte](b.denseArena, "batch-dense-tt-records", ((width+3)/4)*maxComp*8),
		ttCounts:          cudamem.AllocIn[uint32](b.denseArena, "batch-dense-tt-counts", (width+3)/4),
		ttRawMirror:       cudamem.AllocIn[byte](b.denseArena, "batch-dense-tt-raw-mirror", (width+128)*512*2),
		ttCompMirror:      cudamem.AllocIn[byte](b.denseArena, "batch-dense-tt-comp-mirror", maxComp*512*2),
	}
	if cudaModelNeedsQ4(r.model) {
		b.dense.q4Input = cudamem.AllocIn[byte](b.denseArena, "batch-q4-input", width*cudaWidth/256*292)
		b.dense.q4Experts = cudamem.AllocIn[byte](b.denseArena, "batch-q4-experts", width*6*2048/256*292)
		b.dense.q4Mid = alloc("batch-q4-mid", width*6*2048)
	}
	if r.collectors != nil && r.collectors.imatrix != nil {
		if uint64(width)*6*2048*4 > 256<<20 {
			return nil, errors.New("cuda profile mid exceeds byte limit")
		}
		b.dense.profileMid = cudamem.AllocIn[float32](b.denseArena, "batch-profile-mid", width*6*2048)
	}
	// Batch HC by default. DS4_CUDA_HC_BATCH=0 retains the scalar path for
	// diagnosis and rollback; diagnostic fallbacks below remain scalar.
	if os.Getenv("DS4_CUDA_HC_BATCH") != "0" {
		for j := range 2 {
			b.dense.hc[j] = alloc(fmt.Sprintf("batch-dense-hc-%d", j), width*cudaHCWidth)
		}
		b.hcBatch, err = newCUDAHCBatch(r.cudaFoundation, width)
		if err != nil {
			return nil, err
		}
	}
	b.dense.denseD2RLayer = -1
	if r.denseD2REnabled() {
		gateBytes := denseQ8AlignedBytes(2048, 4096)
		downBytes := denseQ8AlignedBytes(4096, 2048)
		bBytes := denseQ8AlignedBytes(4096, 8192)
		aBytes := 8 * denseQ8AlignedBytes(1024, 4096)
		// AttnOutA quantizes eight 4096-wide slices in one launch. Keep one
		// block-major output region per slice so the following eight D2R
		// launches can consume their bytes independently.
		actBytes := 8 * ((4096/128)*width + 256) * 144

		b.dense.denseGate = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-gate", gateBytes)
		b.dense.denseUp = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-up", gateBytes)
		b.dense.denseDown = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-down", downBytes)
		b.dense.denseAttnOutB = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-attnout-b", bBytes)
		b.dense.denseAttnOutA = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-attnout-a", aBytes)
		b.dense.denseQ81Act = cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-act", actBytes)
		b.dense.denseAttnQA = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-attn-qa", denseQ8AlignedBytes(cudaQLoraWidth, cudaWidth))
		b.dense.denseAttnQB = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-attn-qb", denseQ8AlignedBytes(cudaAttentionHeads, cudaQLoraWidth))
		b.dense.denseAttnKV = cudamem.AllocIn[byte](b.denseArena, "batch-dense-repack-attn-kv", denseQ8AlignedBytes(cudaKVWidth, cudaWidth))
		b.dense.denseSharedMid = cudamem.AllocIn[float32](b.denseArena, "batch-dense-shared-mid", width*2048)
		b.dense.denseScratch = cudaDenseArtifact{
			gate: b.dense.denseGate, up: b.dense.denseUp, down: b.dense.denseDown,
			attnOutB: b.dense.denseAttnOutB, attnOutA: b.dense.denseAttnOutA,
			attnQA: b.dense.denseAttnQA, attnQB: b.dense.denseAttnQB, attnKV: b.dense.denseAttnKV,
		}
	}
	if err := errors.Join(b.arena.Err(), b.denseArena.Err()); err != nil {
		return nil, fmt.Errorf("cuda runtime: allocate batch rows: %w", err)
	}
	b.grouping, err = newCUDAMoEGrouping(r.device, cudaMoEMaxExperts, width*6)
	if err != nil {
		return nil, fmt.Errorf("cuda runtime: allocate batch grouping: %w", err)
	}
	// One cuBLAS handle per token slot, each with its own workspace, and only
	// the scalar fallback path above dereferences them. Allocating width of
	// them ahead of a path that will not read them is what held the prefill
	// group at 256. Past roughly 1700 slots it failed outright with
	// CUBLAS_STATUS_ALLOC_FAILED, which was the harmless symptom; below that it
	// was worse. At 512 slots the same pressure changed the logits, and changed
	// them differently from run to run -- two digests from one build -- while
	// costing a factor of three in throughput. Both looked at first like
	// evidence that a wider group reassociates the reductions, and neither was:
	// widths 256 through 2048 are bit-identical once these are not allocated.
	if r.scalarHCPath(b) {
		for i := range b.hc {
			b.hc[i], err = newCUDAHC(r.cudaFoundation)
			if err != nil {
				return nil, fmt.Errorf("cuda runtime: allocate batch HC %d: %w", i, err)
			}
		}
	}
	allowed := make([]uint32, r.state.allowed.Len())
	for j := range allowed {
		allowed[j] = 1
	}
	for i := range b.rows {
		if err := b.rows[i].allowed.CopyFromHost(allowed); err != nil {
			return nil, fmt.Errorf("cuda runtime: initialize batch allowed %d: %w", i, err)
		}
	}
	b.dense.d2rLayer = -1
	if r.needsMoED2RScratch() {
		b.dense.d2rGate, err = cudamem.Alloc[byte](r.device, cudaIQ2XXSRepackedBytes)
		if err != nil {
			_ = b.Close()
			return nil, fmt.Errorf("cuda runtime: allocate d2rGate: %w", err)
		}
		b.dense.d2rUp, err = cudamem.Alloc[byte](r.device, cudaIQ2XXSRepackedBytes)
		if err != nil {
			_ = b.Close()
			return nil, fmt.Errorf("cuda runtime: allocate d2rUp: %w", err)
		}
		b.dense.d2rDown, err = cudamem.Alloc[byte](r.device, cudaQ2KRepackedBytes)
		if err != nil {
			_ = b.Close()
			return nil, fmt.Errorf("cuda runtime: allocate d2rDown: %w", err)
		}
	}
	// These buffers are allocated on the first EvalBatch and nothing writes
	// them before the first prefill reads them. That is invisible while the
	// allocator hands back memory whose last owner left zeroes in it, and it is
	// exactly what a first evaluation that returns wrong logits looks like when
	// the last owner was a process that held 72.56 GiB. DS4_CUDA_BATCH_ARENA_FILL
	// gives them a known value instead: "zero" says what the path assumes, and
	// any other value is its own control -- if a fill of ones moves the digest
	// somewhere new and a fill of zeroes fixes it, something is read before it
	// is written, and if neither changes anything the buffers are not the cause.
	if v := os.Getenv("DS4_CUDA_BATCH_ARENA_FILL"); v != "" {
		word := uint32(0)
		if v != "zero" {
			n, err := strconv.ParseUint(v, 0, 32)
			if err != nil {
				_ = b.Close()
				return nil, fmt.Errorf("cuda runtime: batch arena fill %q: %w", v, err)
			}
			word = uint32(n)
		}
		for _, a := range []*cudamem.Arena{b.arena, b.denseArena} {
			if err := a.Fill(word); err != nil {
				_ = b.Close()
				return nil, fmt.Errorf("cuda runtime: fill batch arena: %w", err)
			}
		}
	}
	return b, nil
}

func (r *cudaRuntime) ensureBatch(allocate func() (*cudaBatchResources, error)) error {
	if r.batch != nil {
		return nil
	}
	b, err := allocate()
	if err != nil {
		if b != nil {
			err = finishBackendCleanup(err, b)
		}
		return err
	}
	if b == nil || len(b.rows) < cudaPrefillGroupWidth-1 || len(b.hc) < cudaPrefillGroupWidth-1 {
		err = errors.New("cuda runtime: incomplete batch allocation")
		if b != nil {
			err = finishBackendCleanup(err, b)
		}
		return err
	}
	r.batch = b
	return nil
}

// evalLayerBatch carries the layer span for prefill. evalLayerState has
// the decode one, and the projected tail both paths share has none, so
// without this a prefill emits no layer intervals at all.
func (r *cudaRuntime) evalLayerBatch(base []*cudaTokenState, hc []*cudaHC, index int, tokens []int, position int) error {
	defer beginTracedLayer(index).end()
	if len(base) != len(tokens) || len(hc) != len(tokens) || len(tokens) < 2 || len(tokens) > cudaPrefillGroupWidth {
		return errors.New("cuda runtime: dense batch has invalid length")
	}
	l := &r.model.layers[index]
	state := &r.state.layers[index]
	d := &r.batch.dense
	gather := !r.disableGatherExperts && !r.disableSIMDExperts
	if err := r.markCUDAPhase("dense-front", position, index); err != nil {
		return err
	}
	rawCap, maxComp := min(r.ctx, cudaRawCap), r.ctx/4+2
	scoreStride := 64 * (rawCap + maxComp)
	rows := make([]cudaTokenState, len(tokens))
	for i := range rows {
		rows[i] = *base[i]
		rows[i].attnPre = d.attnPre.Slice(i*2*cudaWidth, (i+1)*2*cudaWidth)
		rows[i].attnNorm = rows[i].attnPre.Slice(cudaWidth, 2*cudaWidth)
		rows[i].qLora = d.qLora.Slice(i*1024, (i+1)*1024)
		rows[i].qHeads = d.qHeads.Slice(i*64*cudaKVWidth, (i+1)*64*cudaKVWidth)
		rows[i].kvRaw = d.kvRaw.Slice(i*cudaKVWidth, (i+1)*cudaKVWidth)
		rows[i].kvCur = d.kvCur.Slice(i*cudaKVWidth, (i+1)*cudaKVWidth)
		// The attention output, the shared FFN's activation and the shared
		// projections move into token-major group storage so the group projects
		// them with one launch per weight.
		// Raw and mixed attention run in different layers and can share this
		// storage. Keep only these local views rebound: decode still owns its
		// original buffers, including the terminal token's state.
		rows[i].rawOutput = d.attention.Slice(i*cudaAttentionStride, (i+1)*cudaAttentionStride)
		rows[i].mixedOutput = rows[i].rawOutput
		rows[i].allowed = d.allowed.Slice(i*maxComp, (i+1)*maxComp)
		rows[i].mixedScores = d.mixedScores.Slice(i*scoreStride, (i+1)*scoreStride)
		rows[i].mixedDenominators = d.mixedDenominators.Slice(i*64, (i+1)*64)
		rows[i].attnLow = d.attnLow.Slice(i*8192, (i+1)*8192)
		rows[i].attnOut = d.attnOut.Slice(i*cudaWidth, (i+1)*cudaWidth)
		rows[i].ffnPre = d.ffnPre.Slice(i*2*cudaWidth, (i+1)*2*cudaWidth)
		rows[i].ffnNorm = rows[i].ffnPre.Slice(cudaWidth, 2*cudaWidth)
		rows[i].router = d.router.Slice(i*256, (i+1)*256)
		rows[i].probs = d.probs.Slice(i*256, (i+1)*256)
		rows[i].expertIDs = d.expertIDs.Slice(i*6, (i+1)*6)
		rows[i].expertW = d.expertW.Slice(i*6, (i+1)*6)
		if r.collectors != nil && r.collectors.imatrix != nil {
			rows[i].expertMid = d.profileMid.Slice(i*6*2048, (i+1)*6*2048)
		}
		rows[i].q81Input = d.q81FFN.Slice(i*cudaWidth/32*36, (i+1)*cudaWidth/32*36)
		rows[i].sharedGate = d.sharedGate.Slice(i*cudaSharedFFNWidth, (i+1)*cudaSharedFFNWidth)
		rows[i].sharedUp = d.sharedUp.Slice(i*cudaSharedFFNWidth, (i+1)*cudaSharedFFNWidth)
		rows[i].sharedOut = d.sharedOut.Slice(i*cudaWidth, (i+1)*cudaWidth)
		if gather {
			rows[i].expertGate = d.expertGate.Slice(i*6*2048, (i+1)*6*2048)
			rows[i].expertUp = d.expertUp.Slice(i*6*2048, (i+1)*6*2048)
			rows[i].expertDown = d.expertDown.Slice(i*6*cudaWidth, (i+1)*6*cudaWidth)
			rows[i].ffnOut = d.ffnOut.Slice(i*cudaWidth, (i+1)*cudaWidth)
			rows[i].q81Experts = d.q81Experts.Slice(i*6*2048/32*36, (i+1)*6*2048/32*36)
		}
		// Compressor rows are adjacent at the layer's own compressor width, the
		// layout the batched pair kernel writes. Layers 0 and 1 have none.
		if index >= 2 {
			w := state.attn.width
			rows[i].compressorKV = d.compressorKV.Slice(i*w, (i+1)*w)
			rows[i].compressorGate = d.compressorGate.Slice(i*w, (i+1)*w)
			if state.indexer.cap != 0 {
				iw := state.indexer.width
				rows[i].indexerKV = d.indexerKV.Slice(i*iw, (i+1)*iw)
				rows[i].indexerGate = d.indexerGate.Slice(i*iw, (i+1)*iw)
			}
		}
	}
	n := len(rows)
	useGraph := !r.disableGraphs && r.phaseMark == nil && len(tokens) == cudaPrefillGroupWidth
	// Layers 0 and 1 bind no compressor, so their attention is the raw window
	// alone -- which is what the tokentile path computes once the compressed
	// record count is zero. Taking it here replaces two launches per token per
	// layer with the group's own, and costs nothing the mixed layers do not
	// already pay. The tokentile arm is the only one reachable: the tiled
	// fallback beside it reads the compressor cache these layers do not have.
	rawGrouped := index < 2 && r.attentionHMMAEnabled() && n >= 4 &&
		r.phaseMark == nil && !r.disableRawAttentionBatch
	perToken := (index < 2 && !rawGrouped) || r.disableMixedAttentionFused
	tokentile := !perToken && r.attentionHMMAEnabled() && n >= 4
	// The tokentile kernel reads the query heads in place, so they are
	// normalized and rotated in one pass there, after the graph below: the
	// rotation depends on the group's position, which a replayed graph cannot
	// carry. The layer-0 query dump wants the unrotated norm, so it keeps the
	// separate passes.
	fuseQRoPE := tokentile && !(index == 0 && os.Getenv("DS4_DUMP_LAYERS") != "")
	evalBatchQKV := func() error {
		if b := r.activeHCBatch(); b != nil {
			if err := b.pre(d.hc[0].Slice(0, n*cudaHCWidth), l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, d.attnPre.Slice(0, n*2*cudaWidth), n); err != nil {
				return err
			}
		} else {
			for i := range rows {
				if err := hc[i].pre(rows[i].hc[0], l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, rows[i].attnPre, false); err != nil {
					return fmt.Errorf("token %d attention pre: %w", i, err)
				}
			}
		}
		// The query and key/value projections are the same three matrices for
		// every token in the group, so repack them once per layer here rather
		// than streaming them once per 32 tokens through the matvec.
		qkvD2R := r.attnQKVD2REnabled() && d.denseScratch.attnQB.Len() > 0
		if qkvD2R && !r.bindDense(d, index) {
			if err := r.repackLayerAttnQ8(l, d); err != nil {
				return err
			}
		}
		if err := chunkRows(n, cudaPrefillQ81Chunk, func(lo, hi int) error {
			m := hi - lo
			attnNorm := d.attnPre.Slice(lo*2*cudaWidth+cudaWidth, hi*2*cudaWidth)
			q81Attn := d.q81Attn.Slice(lo*cudaWidth/32*36, hi*cudaWidth/32*36)
			qLora := d.qLora.Slice(lo*cudaQLoraWidth, hi*cudaQLoraWidth)
			q81QLora := d.q81QLora.Slice(lo*cudaQLoraWidth/32*36, hi*cudaQLoraWidth/32*36)
			qHeads := d.qHeads.Slice(lo*cudaAttentionHeads, hi*cudaAttentionHeads)
			kvRaw := d.kvRaw.Slice(lo*cudaKVWidth, hi*cudaKVWidth)
			if qkvD2R {
				// attnQA (out_dim 1024) and attnKV (out_dim 512) fail the
				// reference envelope; attnQB (out_dim 32768, in_dim 1024)
				// passes it, so the arm moves only the first two.
				qkvEnv, _ := cudaDenseD2REnvelope()
				offQA := qkvEnv && !denseD2REnvelopeShape(cudaQLoraWidth, m, cudaWidth)
				offKV := qkvEnv && !denseD2REnvelopeShape(cudaKVWidth, m, cudaWidth)
				if offQA || offKV {
					if err := r.encodeQuantQ81Batch(attnNorm, q81Attn, cudaWidth, 2*cudaWidth, m); err != nil {
						return fmt.Errorf("query lora envelope quantization: %w", err)
					}
				}
				if !offQA || !offKV {
					// attnQA and attnKV share one quantization of attnNorm, so both
					// run before qLora overwrites the activation scratch.
					if err := r.encodeQuantMMQ(attnNorm, d.denseQ81Act, 2*cudaWidth, cudaWidth, m); err != nil {
						return fmt.Errorf("query lora quantization: %w", err)
					}
				}
				if offQA {
					if err := r.encodeQ80MatvecQ81Batch(l.attnQA, q81Attn, qLora, m); err != nil {
						return fmt.Errorf("query lora envelope projection: %w", err)
					}
				} else if err := r.encodeDenseQ8D2R(d.denseAttnQA, d.denseQ81Act, qLora, cudaQLoraWidth, m, cudaWidth, cudaQLoraWidth); err != nil {
					return fmt.Errorf("query lora projection: %w", err)
				}
				if offKV {
					if err := r.encodeQ80MatvecQ81Batch(l.attnKV, q81Attn, kvRaw, m); err != nil {
						return fmt.Errorf("key value envelope projection: %w", err)
					}
				} else if err := r.encodeDenseQ8D2R(d.denseAttnKV, d.denseQ81Act, kvRaw, cudaKVWidth, m, cudaWidth, cudaKVWidth); err != nil {
					return fmt.Errorf("key value projection: %w", err)
				}
				if err := r.encodeRMS(qLora, l.attnQANorm, qLora, m, true); err != nil {
					return fmt.Errorf("query lora norm: %w", err)
				}
				if err := r.encodeQuantMMQ(qLora, d.denseQ81Act, cudaQLoraWidth, cudaQLoraWidth, m); err != nil {
					return fmt.Errorf("query head quantization: %w", err)
				}
				return r.encodeDenseQ8D2R(d.denseAttnQB, d.denseQ81Act, qHeads, cudaAttentionHeads, m, cudaQLoraWidth, cudaAttentionHeads)
			}
			if err := r.encodeQuantQ81Batch(attnNorm, q81Attn, cudaWidth, 2*cudaWidth, m); err != nil {
				return err
			}
			if err := r.encodeQ80MatvecQ81Batch(l.attnQA, q81Attn, qLora, m); err != nil {
				return err
			}
			if err := r.encodeRMS(qLora, l.attnQANorm, qLora, m, true); err != nil {
				return fmt.Errorf("query lora norm: %w", err)
			}
			if err := r.encodeQuantQ81Batch(qLora, q81QLora, cudaQLoraWidth, cudaQLoraWidth, m); err != nil {
				return err
			}
			if err := r.encodeQ80MatvecQ81Batch(l.attnQB, q81QLora, qHeads, m); err != nil {
				return err
			}
			return r.encodeQ80MatvecQ81Batch(l.attnKV, q81Attn, kvRaw, m)
		}); err != nil {
			return err
		}
		// The compressor projections read only each token's attnNorm, so the group
		// streams both weight pairs once; the pooling that consumes them stays in
		// the token-ordered loop below.
		if index >= 2 {
			if err := r.evalBatchCompressor(rows, index); err != nil {
				return err
			}
		}
		// Normalization for all query heads and key/value rows across the group
		// is grouped across all n tokens in one pass each.
		if !useGraph {
			if err := r.markCUDAPhase("qkv-normalization", position, index); err != nil {
				return err
			}
		}
		qHeadsAll := d.qHeads.Slice(0, n*64*cudaKVWidth)
		if err := r.dumpBatchQKVRows(qHeadsAll, d.kvRaw.Slice(0, n*cudaKVWidth), cudamem.Slice[float32]{}, index, position, n, "Qraw", "KVraw"); err != nil {
			return err
		}
		if !fuseQRoPE {
			if err := r.encodeRMS(qHeadsAll, cudaModelTensor{}, qHeadsAll, n*64, false); err != nil {
				return fmt.Errorf("batch query head norm: %w", err)
			}
		}
		kvRawAll := d.kvRaw.Slice(0, n*cudaKVWidth)
		kvCurAll := d.kvCur.Slice(0, n*cudaKVWidth)
		if err := r.encodeRMS(kvRawAll, l.attnKVNorm, kvCurAll, n, true); err != nil {
			return fmt.Errorf("batch key/value norm: %w", err)
		}
		if err := r.dumpBatchQKVRows(qHeadsAll, cudamem.Slice[float32]{}, kvCurAll, index, position, n, "Qnorm", "KVnorm"); err != nil {
			return err
		}
		return nil
	}
	if useGraph {
		if r.batch.graphs[index].qkv != nil {
			if err := r.batch.graphs[index].qkv.Launch(r.stream()); err != nil {
				return fmt.Errorf("cuda graph: launch batch qkv layer %d: %w", index, err)
			}
		} else {
			var bodyErr error
			g, err := r.captureGraph(func() {
				bodyErr = evalBatchQKV()
			})
			if err != nil {
				if bodyErr != nil {
					return fmt.Errorf("cuda graph: capture batch qkv layer %d: %w", index, errors.Join(bodyErr, err))
				}
				return fmt.Errorf("cuda graph: capture batch qkv layer %d: %w", index, err)
			}
			defer g.Destroy()
			if bodyErr != nil {
				return bodyErr
			}
			exec, err := g.Instantiate()
			if err != nil {
				return fmt.Errorf("cuda graph: instantiate batch qkv layer %d: %w", index, err)
			}
			r.batch.graphs[index].qkv = exec
			if err := exec.Launch(r.stream()); err != nil {
				return fmt.Errorf("cuda graph: launch initial batch qkv layer %d: %w", index, err)
			}
		}
	} else {
		if err := evalBatchQKV(); err != nil {
			return err
		}
	}
	// Attention is token ordered: compress mutates the compressor caches that the
	// following tokens read. Head outputs occupy strided rows of the dense
	// attention buffer, so quantization and projection can batch below.
	heads := make([]cudamem.Slice[float32], n)
	if perToken {
		for i := range rows {
			var err error
			if heads[i], err = r.evalLayerAttention(&rows[i], state, index, position+i, true); err != nil {
				return fmt.Errorf("token %d: %w", i, err)
			}
		}
	} else {
		nCompHost := make([]uint32, n)
		maxTotal := 0
		if err := r.encodeFillU32(d.allowed.Slice(0, n*maxComp), n*maxComp, 1); err != nil {
			return err
		}
		n0, grouped := state.attn.n, r.compressGroupEligible(state, position, n)
		if grouped {
			if err := r.compressGroup(l, state, d, position, n); err != nil {
				return err
			}
		}
		for i := range rows {
			if err := r.markCUDAPhase("compression", position+i, index); err != nil {
				return err
			}
			switch {
			case state.attn.cap == 0:
				nCompHost[i] = 0
			case grouped:
				nCompHost[i] = uint32(n0 + (i+1)/state.attn.ratio)
			default:
				if err := r.compress(l, state, &rows[i], position+i, index, true, true); err != nil {
					return fmt.Errorf("token %d: %w", i, err)
				}
				nCompHost[i] = uint32(state.attn.n)
			}
			tot := min(position+i+1, rawCap) + int(nCompHost[i])
			if tot > maxTotal {
				maxTotal = tot
			}
			if err := r.markCUDAPhase("attention", position+i, index); err != nil {
				return err
			}
			heads[i] = d.attention.Slice(i*cudaAttentionStride+cudaAttentionScratch, i*cudaAttentionStride+cudaAttentionScratch+cudaAttentionHeads)
		}
		// Keep this transfer on the inference stream. CopyFromHost performs a
		// synchronous HtoD operation and drains the legacy stream once per
		// layer; the C path carries this value as launch metadata. The async
		// copy preserves the existing device-buffer ABI while allowing the
		// following attention launches to remain queued on the same stream.
		if err := d.tokenNComp.Slice(0, n).CopyFromHostAsync(nCompHost, r.stream()); err != nil {
			return fmt.Errorf("copy token n_comp: %w", err)
		}

		qHeadsAll := d.qHeads.Slice(0, n*64*cudaKVWidth)
		kvCurAll := d.kvCur.Slice(0, n*cudaKVWidth)
		outAll := d.attention.Slice(0, n*cudaAttentionStride)
		// q is where the tokentile kernel reads the rotated query heads, at
		// qStride floats per token.
		q, qStride := outAll, cudaAttentionStride
		if fuseQRoPE {
			if err := r.headRMSRoPEBatch(qHeadsAll, position, n, index < 2); err != nil {
				return fmt.Errorf("batch query head norm: %w", err)
			}
			if err := r.mixedAttentionPrepareKVBatch(kvCurAll, outAll, position, n, index < 2); err != nil {
				return fmt.Errorf("batch mixed attention prepare: %w", err)
			}
			q, qStride = qHeadsAll, cudaAttentionHeads
		} else if err := r.mixedAttentionPrepareBatch(qHeadsAll, kvCurAll, outAll, position, rawCap, n, index < 2); err != nil {
			return fmt.Errorf("batch mixed attention prepare: %w", err)
		}
		if tokentile {
			recStride := max(int(state.attn.n), 1)
			if err := r.encodeTokentileDenseBuild(d.ttRecords, d.ttCounts, position, n, state.attn.ratio, state.attn.n, recStride); err != nil {
				return fmt.Errorf("batch tokentile dense build: %w", err)
			}
			availableBefore := min(position, 127)
			rawRowMin := 127 - availableBefore
			if err := r.encodeTokentileRawMirror(d.ttRawMirror, state.rawKV, outAll, position, n, rawCap, rawRowMin); err != nil {
				return fmt.Errorf("batch tokentile raw mirror: %w", err)
			}
			if state.attn.n > 0 {
				if err := r.encodeTokentileCompMirror(d.ttCompMirror, state.attn.cache.Slice(0, state.attn.n*512), state.attn.n); err != nil {
					return fmt.Errorf("batch tokentile comp mirror: %w", err)
				}
			}
			headsOut := outAll.Slice(cudaAttentionScratch, outAll.Len())
			if err := r.encodeTokentileHMMA(
				headsOut,
				l.attnSinks,
				q,
				d.ttRawMirror,
				d.ttCompMirror,
				d.ttRecords,
				d.ttCounts,
				position,
				qStride,
				cudaAttentionStride,
				recStride,
				n,
				64,
				rawRowMin,
				index < 2,
			); err != nil {
				return fmt.Errorf("batch tokentile hmma: %w", err)
			}
			if err := r.mixedAttentionCommitRaw(state.rawKV, outAll, position, rawCap, n); err != nil {
				return fmt.Errorf("batch mixed attention commit raw: %w", err)
			}
		} else {
			if err := r.mixedAttentionTileScoresBatch(
				state.rawKV,
				state.attn.cache,
				d.allowed.Slice(0, n*maxComp),
				d.mixedScores.Slice(0, n*scoreStride),
				outAll,
				d.tokenNComp.Slice(0, n),
				maxTotal,
				position,
				rawCap,
				maxComp,
				scoreStride,
				n,
			); err != nil {
				return fmt.Errorf("batch mixed attention tile scores: %w", err)
			}
			if err := r.mixedAttentionFinalizeBatch(
				l.attnSinks,
				state.rawKV,
				state.attn.cache,
				d.allowed.Slice(0, n*maxComp),
				d.mixedScores.Slice(0, n*scoreStride),
				d.mixedDenominators.Slice(0, n*64),
				outAll,
				d.tokenNComp.Slice(0, n),
				maxTotal,
				position,
				rawCap,
				maxComp,
				scoreStride,
				n,
			); err != nil {
				return fmt.Errorf("batch mixed attention finalize: %w", err)
			}
			if err := r.mixedAttentionCommitRaw(state.rawKV, outAll, position, rawCap, n); err != nil {
				return fmt.Errorf("batch mixed attention commit raw: %w", err)
			}
		}
	}
	evalBatchFFN := func() error {
		if r.denseD2REnabled() {
			if !r.bindDense(d, index) {
				if err := r.repackLayerDenseQ8(l, d); err != nil {
					return err
				}
			}
			d.denseD2RLayer = index
		}
		// The attention output weights do not depend on the token either. The
		// qualification fallback keeps its per-token kernels, which is the whole
		// point of that switch, so the group projects with it token by token.
		projected := !r.disableBlockDiagAttnOut
		if projected {
			if err := r.evalBatchAttnOut(rows, index, position); err != nil {
				return err
			}
			if err := r.dumpBatchLayerSlice(d.attnOut.Slice((n-1)*cudaWidth, n*cudaWidth), index, position+n-1, "attn_out"); err != nil {
				return err
			}
			if err := r.dumpBatchAttentionRows(d.attnOut, index, position, n); err != nil {
				return err
			}
		}
		if b := r.activeHCBatch(); b != nil {
			if r.steering != nil {
				if err := r.steering.projectRows(d.attnOut.Slice(0, n*cudaWidth), index, n, r.steering.attention); err != nil {
					return err
				}
			}
			if err := b.post(d.hc[0].Slice(0, n*cudaHCWidth), d.attnOut.Slice(0, n*cudaWidth), d.hc[1].Slice(0, n*cudaHCWidth), n); err != nil {
				return err
			}
			if err := r.dumpBatchLayerSlice(d.hc[1].Slice((n-1)*cudaHCWidth, n*cudaHCWidth), index, position+n-1, "hc_attn_post"); err != nil {
				return err
			}
			if !useGraph {
				if err := r.markCUDAPhase("ffn-pre", position, index); err != nil {
					return err
				}
			}
			if err := b.pre(d.hc[1].Slice(0, n*cudaHCWidth), l.hcFFNFn, l.hcFFNScale, l.hcFFNBase, l.ffnNorm, d.ffnPre.Slice(0, n*2*cudaWidth), n); err != nil {
				return err
			}
			if err := r.dumpBatchLayerSlice(d.ffnPre.Slice((n-1)*2*cudaWidth+cudaWidth, n*2*cudaWidth), index, position+n-1, "hc_ffn_pre"); err != nil {
				return err
			}
		} else {
			for i := range rows {
				if err := r.evalLayerFFNPre(hc[i], &rows[i], index, tokens[i], position+i, heads[i], projected); err != nil {
					return fmt.Errorf("token %d: %w", i, err)
				}
			}
		}
		if err := chunkRows(n, cudaPrefillRouterChunk, func(lo, hi int) error {
			m := hi - lo
			ffnNorm := d.ffnPre.Slice(lo*2*cudaWidth+cudaWidth, hi*2*cudaWidth)
			routerOut := d.router.Slice(lo*256, hi*256)
			// The per-token kernel re-reads the whole router matrix for every
			// token; one GEMM reads it once. Same operand precision -- that
			// kernel rounds its activations to half too -- so only the
			// summation order differs.
			if hb := r.activeHCBatch(); hb != nil && !r.disableLibraryGEMM {
				return r.encodeF16MatvecGEMM(hb.h.blas, l.router, ffnNorm, 2*cudaWidth,
					d.pairHalf.Slice(0, m*cudaWidth), routerOut, m)
			}
			return r.encodeF16MatvecDefaultBatch(l.router, ffnNorm, 2*cudaWidth, routerOut, m)
		}); err != nil {
			return err
		}
		if index < 3 {
			if err := r.encodeRouterHashBatch(d.router.Slice(0, n*256), l.expertMap, d.probs.Slice(0, n*256), d.expertIDs.Slice(0, n*6), d.expertW.Slice(0, n*6), d.tokens.Slice(0, n), n); err != nil {
				return err
			}
		} else {
			if err := r.encodeRouterBiasedBatch(d.router.Slice(0, n*256), l.expertBias, d.probs.Slice(0, n*256), d.expertIDs.Slice(0, n*6), d.expertW.Slice(0, n*6), n); err != nil {
				return err
			}
		}
		if !useGraph {
			for i := range rows {
				if err := r.markCUDAPhase("shared-ffn", position+i, index); err != nil {
					return err
				}
			}
		}
		if batchQ81FFNNeeded(gather, r.moeD2R(d, index), r.denseD2REnabled(), d.denseGate.Len()) {
			if err := chunkRows(n, cudaPrefillQ81Chunk, func(lo, hi int) error {
				m := hi - lo
				ffnNorm := d.ffnPre.Slice(lo*2*cudaWidth+cudaWidth, hi*2*cudaWidth)
				q81FFN := d.q81FFN.Slice(lo*cudaWidth/32*36, hi*cudaWidth/32*36)
				return r.encodeQuantQ81Batch(ffnNorm, q81FFN, cudaWidth, 2*cudaWidth, m)
			}); err != nil {
				return err
			}
		}
		if gather && r.moeD2R(d, index) {
			ffnNormAll := d.ffnPre.Slice(cudaWidth, n*2*cudaWidth)
			q81MMQAll := d.q81MMQ.Slice(0, (cudaWidth/128)*n*144)
			if err := r.encodeQuantMMQ(ffnNormAll, q81MMQAll, 2*cudaWidth, cudaWidth, n); err != nil {
				return err
			}
		}
		// The shared expert's weights do not depend on the token, and each token's
		// projection reads only its own activation row, so the group streams each
		// weight matrix once instead of n times. This is the same amortization the
		// dense front end above already gets for attnQA, attnQB and attnKV.
		if err := chunkRows(n, cudaPrefillQ81Chunk, func(lo, hi int) error {
			m := hi - lo
			if r.denseD2REnabled() && d.denseGate.Len() > 0 {
				ffnNorm := d.ffnPre.Slice(lo*2*cudaWidth+cudaWidth, hi*2*cudaWidth)
				if err := r.encodeQuantMMQ(ffnNorm, d.denseQ81Act, 2*cudaWidth, cudaWidth, m); err != nil {
					return fmt.Errorf("batch shared gate/up quant: %w", err)
				}
				if err := r.encodeDenseQ8D2R(d.denseGate, d.denseQ81Act, d.sharedGate.Slice(lo*cudaSharedFFNWidth, hi*cudaSharedFFNWidth), 2048, m, 4096, 2048); err != nil {
					return fmt.Errorf("batch shared gate d2r: %w", err)
				}
				return r.encodeDenseQ8D2R(d.denseUp, d.denseQ81Act, d.sharedUp.Slice(lo*cudaSharedFFNWidth, hi*cudaSharedFFNWidth), 2048, m, 4096, 2048)
			}
			q81FFN := d.q81FFN.Slice(lo*cudaWidth/32*36, hi*cudaWidth/32*36)
			if err := r.encodeQ80MatvecQ81Batch(l.sharedGate, q81FFN, d.sharedGate.Slice(lo*cudaSharedFFNWidth, hi*cudaSharedFFNWidth), m); err != nil {
				return err
			}
			return r.encodeQ80MatvecQ81Batch(l.sharedUp, q81FFN, d.sharedUp.Slice(lo*cudaSharedFFNWidth, hi*cudaSharedFFNWidth), m)
		}); err != nil {
			return err
		}
		// The shared expert's down projection reads only this token's activation for
		// the same reason, so it streams sharedDown once as well.
		if err := r.evalBatchSharedDown(rows, index, position); err != nil {
			return err
		}
		if gather {
			if err := r.evalBatchExperts(rows, index); err != nil {
				return err
			}
		}
		if gather {
			if err := r.encodeSumRoutedSharedBatch(d.expertDown.Slice(0, n*6*cudaWidth), d.sharedOut.Slice(0, n*cudaWidth), d.ffnOut.Slice(0, n*cudaWidth), cudaWidth, n); err != nil {
				return err
			}
			if r.steering != nil {
				if err := r.steering.projectRows(d.ffnOut.Slice(0, n*cudaWidth), index, n, r.steering.ffn); err != nil {
					return err
				}
			}
			if err := r.dumpBatchLayerSlice(d.ffnOut.Slice((n-1)*cudaWidth, n*cudaWidth), index, position+n-1, "ffn_out"); err != nil {
				return err
			}
			if !useGraph {
				for i := range rows {
					if err := r.markCUDAPhase("ffn-post", position+i, index); err != nil {
						return err
					}
				}
			}
			if r.activeHCBatch() == nil {
				for i := range rows {
					if err := hc[i].post(rows[i].hc[1], rows[i].ffnOut, rows[i].hc[0]); err != nil {
						return fmt.Errorf("token %d: %w", i, err)
					}
				}
			} else {
				b := r.activeHCBatch()
				if err := b.post(d.hc[1].Slice(0, n*cudaHCWidth), d.ffnOut.Slice(0, n*cudaWidth), d.hc[0].Slice(0, n*cudaHCWidth), n); err != nil {
					return err
				}
			}
		} else {
			for i := range rows {
				if err := r.evalLayerFFNPost(hc[i], &rows[i], index, position+i, true); err != nil {
					return fmt.Errorf("token %d: %w", i, err)
				}
			}
		}
		return nil
	}
	if useGraph {
		if r.batch.graphs[index].ffn != nil {
			if err := r.batch.graphs[index].ffn.Launch(r.stream()); err != nil {
				return fmt.Errorf("cuda graph: launch batch ffn layer %d: %w", index, err)
			}
		} else {
			var bodyErr error
			g, err := r.captureGraph(func() {
				bodyErr = evalBatchFFN()
			})
			if err != nil {
				if bodyErr != nil {
					return fmt.Errorf("cuda graph: capture batch ffn layer %d: %w", index, errors.Join(bodyErr, err))
				}
				return fmt.Errorf("cuda graph: capture batch ffn layer %d: %w", index, err)
			}
			defer g.Destroy()
			if bodyErr != nil {
				return bodyErr
			}
			exec, err := g.Instantiate()
			if err != nil {
				return fmt.Errorf("cuda graph: instantiate batch ffn layer %d: %w", index, err)
			}
			r.batch.graphs[index].ffn = exec
			if err := exec.Launch(r.stream()); err != nil {
				return fmt.Errorf("cuda graph: launch initial batch ffn layer %d: %w", index, err)
			}
		}
	} else {
		if err := evalBatchFFN(); err != nil {
			return err
		}
	}
	if err := r.dumpBatchLayerHC(rows, index, position+len(tokens)-1); err != nil {
		return err
	}
	if err := r.collectCUDAProfiles(rows, index, position); err != nil {
		return err
	}
	return nil
}

// dumpBatchLayerHC is an opt-in diagnostic for matching the oracle's terminal
// prefill row. It is deliberately after the layer completes and inactive unless
// DS4_DUMP_LAYERS is set, so normal execution has no host copy or file I/O.
func (r *cudaRuntime) dumpBatchLayerHC(rows []cudaTokenState, layer, position int) error {
	prefix := os.Getenv("DS4_DUMP_LAYERS")
	if prefix == "" || len(rows) == 0 {
		return nil
	}
	target, explicit, err := dumpPosition()
	if err != nil {
		return err
	}
	if explicit {
		start := position - len(rows) + 1
		if target < start || target > position {
			return nil
		}
		return r.dumpBatchLayerSlice(rows[target-start].hc[0], layer, target, "hc_ffn_post")
	}
	if position != 2047 {
		return nil
	}
	return r.dumpBatchLayerSlice(rows[len(rows)-1].hc[0], layer, position, "hc_ffn_post")
}

// dumpPosition returns an explicitly requested diagnostic row. An absent
// variable preserves the registered dump positions; a present malformed or
// negative value fails closed instead of silently selecting a different row.
func dumpPosition() (int, bool, error) {
	raw, ok := os.LookupEnv("DS4_DUMP_POSITION")
	if !ok {
		return 0, false, nil
	}
	position, err := strconv.Atoi(raw)
	if err != nil || position < 0 {
		return 0, false, fmt.Errorf("invalid DS4_DUMP_POSITION %q", raw)
	}
	return position, true, nil
}

func (r *cudaRuntime) dumpBatchLayerSlice(src cudamem.Slice[float32], layer, position int, name string) error {
	prefix := os.Getenv("DS4_DUMP_LAYERS")
	if prefix == "" || src.Len() == 0 {
		return nil
	}
	target, explicit, err := dumpPosition()
	if err != nil {
		return err
	}
	if explicit {
		if position != target {
			return nil
		}
	} else if position != 2047 && position != 2048 && !(layer == 0 &&
		(position == 0 || position == 64 || position == 512)) {
		return nil
	}
	if err := syncStream(r.stream()); err != nil {
		return fmt.Errorf("synchronize layer %d %s dump: %w", layer, name, err)
	}
	values := make([]float32, src.Len())
	if err := src.CopyToHost(values); err != nil {
		return fmt.Errorf("dump layer %d %s: %w", layer, name, err)
	}
	path := fmt.Sprintf("%s-L%02d-%s-pos%d.f32le", prefix, layer, name, position)
	if err := os.WriteFile(path, float32Bytes(values), 0o600); err != nil {
		return fmt.Errorf("write layer %d %s: %w", layer, name, err)
	}
	return nil
}

// dumpRawKVSlot records the cache row written for decode position 2048. It is
// opt-in and intentionally narrow: normal execution never copies the cache to
// the host, and the dump is used only to compare the step-1 write with the C
// reference before diagnosing the step-2 read.
func (r *cudaRuntime) dumpRawKVSlot(raw cudamem.Slice[float32], layer, position int) error {
	prefix := os.Getenv("DS4_DUMP_LAYERS")
	if prefix == "" || raw.Len() < cudaKVWidth {
		return nil
	}
	target, explicit, err := dumpPosition()
	if err != nil {
		return err
	}
	if explicit {
		if position != target {
			return nil
		}
	} else if position != 2048 {
		return nil
	}
	if err := syncStream(r.stream()); err != nil {
		return fmt.Errorf("synchronize layer %d raw KV dump: %w", layer, err)
	}
	rawCap := raw.Len() / cudaKVWidth
	slot := position % rawCap
	values := make([]float32, cudaKVWidth)
	if err := raw.Slice(slot*cudaKVWidth, (slot+1)*cudaKVWidth).CopyToHost(values); err != nil {
		return fmt.Errorf("dump layer %d raw KV slot %d: %w", layer, slot, err)
	}
	path := fmt.Sprintf("%s-L%02d-rawKVslot-pos%d.f32le", prefix, layer, position)
	if err := os.WriteFile(path, float32Bytes(values), 0o600); err != nil {
		return fmt.Errorf("write layer %d raw KV slot %d: %w", layer, slot, err)
	}
	return nil
}

// dumpBatchQKVRows records the host-visible query and key/value boundaries.
// Qraw/KVraw precede RMS; Qnorm/KVnorm follow RMS and precede fused RoPE.
func (r *cudaRuntime) dumpBatchQKVRows(q, kvRaw, kvCur cudamem.Slice[float32], layer, position, count int, qName, kvName string) error {
	if os.Getenv("DS4_DUMP_LAYERS") == "" || layer != 0 {
		return nil
	}
	target, explicit, err := dumpPosition()
	if err != nil {
		return err
	}
	if !explicit {
		target = 64
	}
	if target < position || target >= position+count {
		return nil
	}
	i := target - position
	if q.Len() > 0 {
		const width = cudaAttentionHeads
		if err := r.dumpBatchLayerSlice(q.Slice(i*width, (i+1)*width), layer, target, qName); err != nil {
			return err
		}
	}
	if kvRaw.Len() > 0 {
		if err := r.dumpBatchLayerSlice(kvRaw.Slice(i*cudaKVWidth, (i+1)*cudaKVWidth), layer, target, kvName); err != nil {
			return err
		}
	}
	if kvCur.Len() > 0 {
		if err := r.dumpBatchLayerSlice(kvCur.Slice(i*cudaKVWidth, (i+1)*cudaKVWidth), layer, target, "KVnorm"); err != nil {
			return err
		}
	}
	return nil
}

func (r *cudaRuntime) dumpBatchAttentionRows(src cudamem.Slice[float32], layer, position, count int) error {
	if os.Getenv("DS4_DUMP_LAYERS") == "" {
		return nil
	}
	target, explicit, err := dumpPosition()
	if err != nil {
		return err
	}
	targets := make([]int, 0, 34)
	if explicit {
		targets = append(targets, target)
	} else {
		if layer != 0 {
			return nil
		}
		for target := 0; target <= 32; target++ {
			targets = append(targets, target)
		}
		targets = append(targets, 64, 512, 2047)
	}
	for _, target := range targets {
		if target < position || target >= position+count {
			continue
		}
		start := (target - position) * cudaWidth
		if err := r.dumpBatchLayerSlice(src.Slice(start, start+cudaWidth), layer, target, "attn_out"); err != nil {
			return err
		}
	}
	return nil
}

func float32Bytes(values []float32) []byte {
	data := make([]byte, 4*len(values))
	for i, value := range values {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(value))
	}
	return data
}

// evalBatchExperts projects all routed experts while keeping the outputs in
// token and slot order for the weighted activation and residual kernels.
func (r *cudaRuntime) evalBatchExperts(rows []cudaTokenState, index int) error {
	l := &r.model.layers[index]
	if r.streamingExperts != nil {
		return errors.New("CUDA streaming grouped experts unsupported")
	}
	if l.expertGate.typeID == typeQ4K {
		return r.evalBatchQ4Experts(rows, index)
	}
	d := &r.batch.dense
	n := len(rows)
	// This view borrows the allocation; the batch owns and frees it.
	g := *r.batch.grouping
	g.npairs = 6 * n
	g.pairs = g.pairs.Slice(0, g.npairs)
	if r.moeD2R(d, index) {
		gate, up, down := d.d2rGate, d.d2rUp, d.d2rDown
		if a := &r.moeArtifacts[index]; a.built() {
			gate, up, down = a.gate, a.up, a.down
		} else if d.d2rLayer != index || !r.disableGraphs {
			if err := r.repackLayerMoE(l, gate, up, down); err != nil {
				return fmt.Errorf("repack layer %d: %w", index, err)
			}
			d.d2rLayer = index
		}
		if os.Getenv("DS4_CUDA_DISABLE_MOE_IDS_HELPER") == "1" {
			if err := r.encodeMoEGroupCSR(d.expertIDs.Slice(0, g.npairs), &g); err != nil {
				return err
			}
			if err := r.encodeD2RMakeIdsSrc(g.pairs, d.idsSrc.Slice(0, g.npairs), g.npairs); err != nil {
				return err
			}
		} else {
			if err := r.encodeMoEIDsHelper(d.expertIDs.Slice(0, g.npairs), d.idsSrc.Slice(0, g.npairs), g.pairs, g.offsets, n, 256, 6); err != nil {
				return err
			}
		}
		capacity32 := (g.npairs+31)/32 + 256
		if err := r.encodeD2RBuildWorklist32(g.offsets, d.d2rWork.Slice(0, capacity32), d.d2rNItems, 256); err != nil {
			return err
		}
		q81MMQ := d.q81MMQ.Slice(0, (cudaWidth/128)*n*144)
		downQ8 := d.downQ8.Slice(0, (2048/128)*g.npairs*144)
		gateup := func() error {
			return r.encodeGateUpD2R(
				gate, up,
				q81MMQ, d.idsSrc.Slice(0, g.npairs), g.pairs, g.offsets,
				d.expertW.Slice(0, g.npairs),
				d.d2rWork.Slice(0, capacity32), d.d2rNItems,
				downQ8,
				2048, cudaWidth, g.npairs, n, 256, swigluClampExp,
			)
		}
		if r.collectors != nil && r.collectors.imatrix != nil {
			gateup = func() error {
				return r.encodeGateUpD2RMid(gate, up, q81MMQ, d.idsSrc.Slice(0, g.npairs), g.pairs, g.offsets, d.expertW.Slice(0, g.npairs), d.d2rWork.Slice(0, capacity32), d.d2rNItems, downQ8, 2048, cudaWidth, g.npairs, n, 256, swigluClampExp, d.profileMid.Slice(0, n*6*2048))
			}
		}
		if err := gateup(); err != nil {
			return fmt.Errorf("d2r gateup: %w", err)
		}
		capacity64 := (g.npairs+63)/64 + 256
		if err := r.encodeD2RBuildWorklist64(g.offsets, d.d2rWork.Slice(0, capacity64), d.d2rNItems, 256); err != nil {
			return err
		}
		if err := r.encodeDownD2R(
			down, downQ8,
			g.pairs, g.offsets,
			d.d2rWork.Slice(0, capacity64), d.d2rNItems,
			d.expertDown.Slice(0, g.npairs*cudaWidth),
			cudaWidth, 2048, g.npairs, 256,
		); err != nil {
			return fmt.Errorf("d2r down: %w", err)
		}
		return nil
	}
	if err := r.encodeMoEGroupCSR(d.expertIDs.Slice(0, g.npairs), &g); err != nil {
		return err
	}
	if err := r.encodeMatvecIQ2SelectedQ81Gather(l.expertGate, l.expertUp, &g, 6,
		d.q81FFN.Slice(0, n*cudaWidth/32*36), d.expertGate.Slice(0, n*6*2048), d.expertUp.Slice(0, n*6*2048)); err != nil {
		return err
	}
	if err := r.encodeSwiGLUQuantQ81(
		d.expertGate.Slice(0, n*6*2048),
		d.expertUp.Slice(0, n*6*2048),
		d.expertW.Slice(0, n*6),
		d.q81Experts.Slice(0, n*6*2048/32*36),
		2048, n*6); err != nil {
		return fmt.Errorf("batch expert activation: %w", err)
	}
	if r.collectors != nil && r.collectors.imatrix != nil {
		if err := r.encodeSwiGLUSlots(d.expertGate.Slice(0, n*6*2048), d.expertUp.Slice(0, n*6*2048), d.expertW.Slice(0, n*6), d.profileMid.Slice(0, n*6*2048), n*6, 1); err != nil {
			return err
		}
	}
	return r.encodeMatvecQ2KSelectedQ81Gather(l.expertDown, &g,
		d.q81Experts.Slice(0, n*6*2048/32*36), d.expertDown.Slice(0, n*6*cudaWidth))
}

// evalBatchCompressor projects the group's compressor activations with one
// launch per weight pair. The batched kernel reads the group's activation rows
// adjacent, and pre writes each token's norm into the second half of its own
// attnPre block, so the rows are gathered first; the gather moves one token's
// activation where the launch it replaces reads the whole weight pair again.
func (r *cudaRuntime) evalBatchCompressor(rows []cudaTokenState, index int) error {
	l, state, d := &r.model.layers[index], &r.state.layers[index], &r.batch.dense
	b := r.activeHCBatch()
	return chunkRows(len(rows), cudaPrefillF16PairChunk, func(lo, hi int) error {
		m := hi - lo
		var in cudamem.Slice[float32]
		if b != nil {
			in = b.norm.Slice(lo*cudaWidth, hi*cudaWidth)
		} else {
			for i := lo; i < hi; i++ {
				if err := d.attnNorm.Slice(i*cudaWidth, (i+1)*cudaWidth).CopyDeviceToDeviceAsync(rows[i].attnNorm, r.stream()); err != nil {
					return fmt.Errorf("token %d compressor activation: %w", i, err)
				}
			}
			in = d.attnNorm.Slice(lo*cudaWidth, hi*cudaWidth)
		}
		// The GEMM path narrows the activations to half and hands both
		// projections to cuBLAS. That is what the reference implementation
		// does, and it is faster, but the half activations lose bits the
		// hand-written kernel keeps, so the two do not agree bit for bit.
		pair := r.encodeF16PairOrderedBatch
		if b != nil && !r.disableLibraryGEMM {
			pair = func(wa, wb cudaModelTensor, in cudamem.Slice[float32], outA, outB cudamem.Slice[float32], n int) error {
				return r.encodeF16PairGEMM(b.h.blas, wa, wb, in, d.pairHalf.Slice(0, n*cudaWidth), outA, outB, n)
			}
		}
		w := state.attn.width
		if err := pair(l.attnCompressorKV, l.attnCompressorGate, in,
			d.compressorKV.Slice(lo*w, hi*w), d.compressorGate.Slice(lo*w, hi*w), m); err != nil {
			return err
		}
		if state.indexer.cap == 0 {
			return nil
		}
		w = state.indexer.width
		return pair(l.indexerCompressorKV, l.indexerCompressorGate, in,
			d.indexerKV.Slice(lo*w, hi*w), d.indexerGate.Slice(lo*w, hi*w), m)
	})
}

// evalBatchAttnOut projects the group's attention outputs with one launch per
// weight tensor. Tokens within each kernel tile share the weight loads.
func (r *cudaRuntime) evalBatchAttnOut(rows []cudaTokenState, index, position int) error {
	l, d := &r.model.layers[index], &r.batch.dense
	// The mark covers the group's whole attention-output phase, however many
	// chunks it takes; it is host side and encodes nothing on the stream.
	if err := r.markCUDAPhase("attention-output", position, index); err != nil {
		return err
	}
	const width = cudaAttentionHeads
	const stride = width / 32 * 36
	lowWidth := rows[0].attnLow.Len()
	return chunkRows(len(rows), cudaPrefillAttnOutChunk, func(lo, hi int) error {
		m := hi - lo
		// Both shapes in this block fail the reference envelope: the eight
		// attnOutA blocks on out_dim=1024, attnOutB on in_dim=8192. The
		// envelope arm therefore takes the whole block to the fallback below,
		// which is the same code a runtime without dense artifacts runs.
		_, attnOutEnv := cudaDenseD2REnvelope()
		if r.denseD2REnabled() && !attnOutEnv && d.denseAttnOutB.Len() > 0 {
			low := d.attnLow.Slice(lo*lowWidth, hi*lowWidth)
			const blockCols = 4096
			headsIn := d.attention.Slice(lo*cudaAttentionStride+cudaAttentionScratch, d.attention.Len())
			if err := r.encodeQuantMMQBlocks(headsIn, d.denseQ81Act, cudaAttentionStride, blockCols, m, blockCols, 8); err != nil {
				return fmt.Errorf("batch attnOutA quant: %w", err)
			}
			if err := r.encodeDenseQ8D2RBlockDiag(d.denseAttnOutA, d.denseQ81Act, low, 1024, m, 4096, 8192, 8); err != nil {
				return fmt.Errorf("batch attnOutA blockdiag d2r: %w", err)
			}
			if err := r.encodeQuantMMQ(low, d.denseQ81Act, 8192, 8192, m); err != nil {
				return fmt.Errorf("batch attnOutB quant: %w", err)
			}
			return r.encodeDenseQ8D2R(d.denseAttnOutB, d.denseQ81Act, d.attnOut.Slice(lo*cudaWidth, hi*cudaWidth), 4096, m, 8192, 4096)
		}
		// Skip each row's scratch without copying the head outputs. The
		// quantizer retains the same independent 32-element blocks.
		heads := d.attention.Slice(lo*cudaAttentionStride+cudaAttentionScratch, hi*cudaAttentionStride)
		if err := r.encodeQuantQ80Strided(heads, d.q80Heads.Slice(lo*stride, hi*stride), width, cudaAttentionStride, m); err != nil {
			return fmt.Errorf("batch attention output quantization: %w", err)
		}
		low := d.attnLow.Slice(lo*lowWidth, hi*lowWidth)
		if err := r.encodeQ80MatvecQuantizedBlockDiagBatch(l.attnOutA, d.q80Heads.Slice(lo*stride, hi*stride), low, 8, m); err != nil {
			return err
		}
		// The pinned attention-output/HC fusion uses native float32 Q8_0
		// scales. These rows are adjacent and each 32-element quantization
		// block is independent, so one launch writes the bytes m per-token
		// launches would.
		q := d.q80Low.Slice(lo*lowWidth/32*36, hi*lowWidth/32*36)
		if err := r.encodeQuantQ80(low, q, lowWidth); err != nil {
			return err
		}
		return r.encodeQ80MatvecQuantizedBatch(l.attnOutB, q, d.attnOut.Slice(lo*cudaWidth, hi*cudaWidth), m)
	})
}

// evalBatchSharedDown projects the group through the shared expert's down
// matrix. The SwiGLU activation reads only each token's own shared gate and up
// rows, which the group already holds token major, so one quantization launch
// feeds one pass over sharedDown.
func (r *cudaRuntime) evalBatchSharedDown(rows []cudaTokenState, index, position int) error {
	l, d := &r.model.layers[index], &r.batch.dense
	// The mark opens the group's second half, where the per-token form opens
	// each token's; it precedes the projection it measures either way.
	if err := r.markCUDAPhase("expert-ffn", position, index); err != nil {
		return err
	}
	return chunkRows(len(rows), cudaPrefillQ80Chunk, func(lo, hi int) error {
		m := hi - lo
		from, to := lo*cudaSharedFFNWidth, hi*cudaSharedFFNWidth
		if r.denseD2REnabled() && d.denseDown.Len() > 0 {
			mid := d.denseSharedMid.Slice(0, to-from)
			if err := r.encodeSwiGLU(d.sharedGate.Slice(from, to), d.sharedUp.Slice(from, to), mid, 1.0); err != nil {
				return fmt.Errorf("batch shared down swiglu: %w", err)
			}
			if err := r.encodeQuantMMQ(mid, d.denseQ81Act, cudaSharedFFNWidth, cudaSharedFFNWidth, m); err != nil {
				return fmt.Errorf("batch shared down quant: %w", err)
			}
			return r.encodeDenseQ8D2R(d.denseDown, d.denseQ81Act, d.sharedOut.Slice(lo*cudaWidth, hi*cudaWidth), 4096, m, 2048, 4096)
		}
		// The SwiGLU quantization is elementwise over the chunk's rows, which
		// are adjacent, so it spans the whole chunk rather than one row.
		q := d.q80Shared.Slice(from/32*36, to/32*36)
		if err := r.encodeSwiGLUQuantQ80(d.sharedGate.Slice(from, to), d.sharedUp.Slice(from, to), q, to-from, 1); err != nil {
			return err
		}
		return r.encodeQ80MatvecQuantizedBatch(l.sharedDown, q, d.sharedOut.Slice(lo*cudaWidth, hi*cudaWidth), m)
	})
}

func (r *cudaRuntime) evalPosition(token, position int, logits bool) (err error) {
	defer beginTraced(levelToken, "token", nil).end()
	// The last phase in the group has no following mark to close it.
	defer closePhaseSpan()
	if r == nil || r.arena == nil {
		return errors.New("cuda runtime: closed")
	}
	if r.failed != nil {
		return fmt.Errorf("cuda runtime: reset required: %w", r.failed)
	}
	if position != r.pos || position < 0 || position >= r.ctx {
		return fmt.Errorf("cuda runtime: position %d does not follow %d", position, r.pos)
	}
	if token < 0 || uint64(token) >= r.model.output.embedding.dimensions[1] {
		return fmt.Errorf("cuda runtime: token %d out of range", token)
	}
	started := r.power.start()
	if os.Getenv("DS4_HEAD_OBSERVER_PREFIX") != "" {
		done, beginErr := r.headObserver.begin(position, []int{token})
		if beginErr != nil {
			return beginErr
		}
		defer func() { done(err == nil) }()
	}
	defer func() {
		if err != nil {
			r.failed = err
		}
	}()
	for stream := range 4 {
		if err := r.encodeEmbeddingRow(r.model.output.embedding, uint32(token), r.state.hc[0].Slice(stream*cudaWidth, (stream+1)*cudaWidth)); err != nil {
			return err
		}
	}
	for layer := range r.model.layers {
		layerStarted := r.power.start()
		if err := r.evalLayerState(&r.state.cudaTokenState, &r.state.layers[layer], layer, token, position); err != nil {
			return fmt.Errorf("cuda runtime: layer %d: %w", layer, err)
		}
		if err := r.power.finishLayer(layer, layerStarted, func() error { return syncStream(r.stream()) }); err != nil {
			return fmt.Errorf("cuda runtime: power layer %d: %w", layer, err)
		}
	}
	if logits {
		if err := r.hcMath.pre(r.state.hc[0], r.model.output.hcFn, r.model.output.hcScale, r.model.output.hcBase, r.model.output.norm, r.state.outputMixed, true); err != nil {
			return err
		}
	}
	if logits {
		if err := r.encodeObservedOutputHead(r.state.outputMixed, r.state.logits, position); err != nil {
			return err
		}
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	if logits {
		if err := r.state.logits.CopyToHost(r.logits); err != nil {
			return err
		}
		if err := checkFiniteLogits("cuda runtime", r.logits); err != nil {
			return err
		}
	}
	r.pos++
	r.power.finishToken(started)
	return nil
}

// denseArtifactMatvecEnabled reports whether decode projects its Q8_0 matvecs
// out of the dense artifact instead of the model mapping. It is on by default;
// DS4_CUDA_DENSE_ARTIFACT_MATVEC=0 is the arm that reads the mapping, kept so
// the comparison stays runnable from one binary.
//
// The flip was earned, not assumed. Both per-kernel gates agree bit for bit
// with the raw-block kernels on eight shapes with every positive control
// firing, and a 12-draw ABBA returned decode digest 665ac935 on every draw of
// both arms while generation went 14.233 -> 17.368 t/s.
func (r *cudaRuntime) denseArtifactMatvecEnabled() bool {
	return r != nil && os.Getenv("DS4_CUDA_DENSE_ARTIFACT_MATVEC") != "0"
}

// decodeQ81 projects t for a single token, out of layer index's dense artifact
// when there is one and out of the model mapping otherwise. aligned selects the
// plane within that layer's artifact; a zero-length slice means the layer has
// none, which is the state every layer is in when the build is bounded by
// DS4_CUDA_DENSE_ARTIFACT_LAYERS.
//
// The two paths compute the same bits -- see ds4_q8_0_matvec_q8_1_aligned --
// so which one runs is a residency choice and never an accuracy one.
func (r *cudaRuntime) decodeQ81(t cudaModelTensor, aligned cudamem.Slice[byte], in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if aligned.Len() > 0 && r.denseArtifactMatvecEnabled() && len(t.dimensions) == 2 {
		return r.encodeQ80MatvecQ81Aligned(aligned, t.dimensions[0], t.dimensions[1], in, out)
	}
	return r.encodeQ80MatvecQ81(t, in, out)
}

func (r *cudaRuntime) evalLayerQKV(s *cudaTokenState, index int) error {
	l := &r.model.layers[index]
	if err := r.hcMath.pre(s.hc[0], l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, s.attnPre, false); err != nil {
		return err
	}
	qAttn := r.state.q80Input.Slice(0, s.attnNorm.Len()/32*36)
	if err := r.encodeQuantQ81(s.attnNorm, qAttn, s.attnNorm.Len()); err != nil {
		return err
	}
	a := &r.denseArtifacts[index]
	if err := r.decodeQ81(l.attnQA, a.attnQA, qAttn, s.qLora); err != nil {
		return err
	}
	if err := r.decodeQ81(l.attnKV, a.attnKV, qAttn, s.kvRaw); err != nil {
		return err
	}
	if err := r.encodeRMSPair(s.qLora, l.attnQANorm, s.qLora, s.kvRaw, l.attnKVNorm, s.kvCur); err != nil {
		return err
	}
	if err := r.encodeQ80Matvec(l.attnQB, a.attnQB, s.qLora, s.qHeads); err != nil {
		return err
	}
	return r.encodeRMS(s.qHeads, cudaModelTensor{}, s.qHeads, 64, false)
}

func (r *cudaRuntime) evalLayerState(s *cudaTokenState, state *cudaSessionLayer, index, token, position int) error {
	defer beginTracedLayer(index).end()
	useGraph := !r.disableGraphs && r.phaseMark == nil && s == &r.state.cudaTokenState
	if useGraph {
		if r.graphs[index].qkv != nil {
			if err := r.graphs[index].qkv.Launch(r.stream()); err != nil {
				return fmt.Errorf("cuda graph: launch qkv layer %d: %w", index, err)
			}
			r.hcMath.ready = true
			r.hcMath.residual = s.hc[0].Ptr()
		} else {
			var bodyErr error
			g, err := r.captureGraph(func() {
				bodyErr = r.evalLayerQKV(s, index)
			})
			if err != nil {
				if bodyErr != nil {
					return fmt.Errorf("cuda graph: capture qkv layer %d: %w", index, errors.Join(bodyErr, err))
				}
				return fmt.Errorf("cuda graph: capture qkv layer %d: %w", index, err)
			}
			defer g.Destroy()
			if bodyErr != nil {
				return bodyErr
			}
			exec, err := g.Instantiate()
			if err != nil {
				return fmt.Errorf("cuda graph: instantiate qkv layer %d: %w", index, err)
			}
			r.graphs[index].qkv = exec
			if err := exec.Launch(r.stream()); err != nil {
				return fmt.Errorf("cuda graph: launch initial qkv layer %d: %w", index, err)
			}
			r.hcMath.ready = true
			r.hcMath.residual = s.hc[0].Ptr()
		}
	} else {
		if err := r.evalLayerQKV(s, index); err != nil {
			return err
		}
	}
	if err := r.evalLayerProjected(r.hcMath, s, state, index, token, position); err != nil {
		return err
	}
	if r.dspark != nil {
		if err := r.dspark.capture(index, position, s.hc[0]); err != nil {
			return err
		}
	}
	if err := r.collectCUDAProfiles([]cudaTokenState{*s}, index, position); err != nil {
		return err
	}
	// Decode dumps are opt-in and host-visible only. The dump helper applies
	// the registered/default position policy; calling it for every decode
	// position lets an explicit DS4_DUMP_POSITION observe step 2 and later
	// layers without changing the device route.
	if err := r.dumpBatchLayerSlice(s.hc[0], index, position, "hc_ffn_post"); err != nil {
		return err
	}
	return nil
}

// evalLayerFFN encodes one token's feed-forward block. It is the token-ordered
// composition of the three halves the batched front end interleaves: the shared
// projections in evalLayerFFNShared read only this token's ffnNorm, so a caller
// holding several tokens may hoist them into one batched launch instead.
func (r *cudaRuntime) evalLayerFFN(hc *cudaHC, s *cudaTokenState, index, token, position int, heads cudamem.Slice[float32]) error {
	if err := r.evalLayerFFNPre(hc, s, index, token, position, heads, false); err != nil {
		return err
	}
	l := &r.model.layers[index]
	if err := r.encodeF16MatvecDefault(l.router, s.ffnNorm, s.router); err != nil {
		return err
	}
	if index < 3 {
		if err := r.encodeRouterHash(s.router, l.expertMap, s.probs, s.expertIDs, s.expertW, uint32(token)); err != nil {
			return err
		}
	} else {
		if err := r.encodeRouterBiased(s.router, l.expertBias, s.probs, s.expertIDs, s.expertW); err != nil {
			return err
		}
	}
	if err := r.markCUDAPhase("shared-ffn", position, index); err != nil {
		return err
	}
	if err := r.encodeQuantQ81(s.ffnNorm, s.q81Input, cudaWidth); err != nil {
		return err
	}
	if err := r.evalLayerFFNShared(s, index); err != nil {
		return err
	}
	return r.evalLayerFFNPost(hc, s, index, position, false)
}

// evalLayerFFNPre encodes the attention output projection, the residual add and
// the FFN RMSNorm expand.
func (r *cudaRuntime) evalLayerFFNPre(hc *cudaHC, s *cudaTokenState, index, token, position int, heads cudamem.Slice[float32], projected bool) error {
	l, a := &r.model.layers[index], &r.denseArtifacts[index]
	if !projected {
		if err := r.markCUDAPhase("attention-output", position, index); err != nil {
			return err
		}
		if !r.disableBlockDiagAttnOut {
			if err := r.encodeQ80NativeBlockDiag(l.attnOutA, a.attnOutA, heads, s.attnLow, 8); err != nil {
				return err
			}
		} else {
			for group := range 8 {
				rows, err := cudaTensorRows(l.attnOutA, group*1024, 1024)
				if err != nil {
					return err
				}
				if err := r.encodeQ80Native(rows, cudamem.Slice[byte]{}, heads.Slice(group*4096, (group+1)*4096), s.attnLow.Slice(group*1024, (group+1)*1024)); err != nil {
					return err
				}
			}
		}
		// The pinned attention-output/HC fusion uses native float32 Q8_0 scales.
		if err := r.encodeQ80Native(l.attnOutB, a.attnOutB, s.attnLow, s.attnOut); err != nil {
			return err
		}
	}
	if r.steering != nil {
		if err := r.steering.project(s.attnOut, index, r.steering.attention); err != nil {
			return err
		}
	}
	if err := hc.post(s.hc[0], s.attnOut, s.hc[1]); err != nil {
		return err
	}
	if err := r.markCUDAPhase("ffn-pre", position, index); err != nil {
		return err
	}
	return hc.pre(s.hc[1], l.hcFFNFn, l.hcFFNScale, l.hcFFNBase, l.ffnNorm, s.ffnPre, false)
}

// evalLayerFFNShared projects one token through the shared expert. Each launch
// streams a whole weight matrix for a single output vector, so the batched front
// end routes these two through encodeQ80MatvecQ81Batch instead; this form serves
// the single-token path. Keep it the only unbatched caller in the FFN.
func (r *cudaRuntime) evalLayerFFNShared(s *cudaTokenState, index int) error {
	l, a := &r.model.layers[index], &r.denseArtifacts[index]
	if err := r.decodeQ81(l.sharedGate, a.gate, s.q81Input, s.sharedGate); err != nil {
		return err
	}
	return r.decodeQ81(l.sharedUp, a.up, s.q81Input, s.sharedUp)
}

// evalLayerFFNPost consumes the shared projections and encodes the routed
// experts and the residual. It reads only this token's state. projected reports
// that the caller already encoded this token's shared-down projection, which the
// batched front end hoists into one launch for the group.
func (r *cudaRuntime) evalLayerFFNPost(hc *cudaHC, s *cudaTokenState, index, position int, projected bool) error {
	l, a := &r.model.layers[index], &r.denseArtifacts[index]
	if !projected {
		// The mark opens this token's second half. It precedes the shared-down
		// projection so that no interval spans two tokens once a batched caller
		// interleaves the halves; the shared-down launch is measured here.
		if err := r.markCUDAPhase("expert-ffn", position, index); err != nil {
			return err
		}
		// Fused SwiGLU activation + Q8_0 quantization into q80Input for shared-down projection.
		q := r.state.q80Input.Slice(0, s.sharedGate.Len()/32*36)
		if err := r.encodeSwiGLUQuantQ80(s.sharedGate, s.sharedUp, q, s.sharedGate.Len(), 1); err != nil {
			return err
		}
		if err := r.decodeQ80Quantized(l.sharedDown, a.down, q, s.sharedOut); err != nil {
			return err
		}
	}
	if r.streamingExperts != nil {
		if err := r.evalStreamingExperts(s, index); err != nil {
			return err
		}
		return r.evalLayerFFNResidual(hc, s, index, position)
	}
	if l.expertGate.typeID == typeQ4K {
		if err := r.evalQ4Experts(s, index); err != nil {
			return err
		}
		return r.evalLayerFFNResidual(hc, s, index, position)
	}
	moe := &r.moeArtifacts[index]
	switch {
	case moe.built() && !r.disableSIMDExperts:
		if err := r.encodeMatvecIQ2SelectedQ81Aligned(l.expertGate, l.expertUp, moe, s.expertIDs, s.q81Input, s.expertGate, s.expertUp); err != nil {
			return err
		}
	case !r.disableSIMDExperts:
		if err := r.encodeMatvecIQ2SelectedQ81(l.expertGate, l.expertUp, s.expertIDs, s.q81Input, s.expertGate, s.expertUp); err != nil {
			return err
		}
	default:
		if err := r.encodeMatvecIQ2SelectedQ81Scalar(l.expertGate, l.expertUp, s.expertIDs, s.q81Input, s.expertGate, s.expertUp); err != nil {
			return err
		}
	}
	if err := r.encodeSwiGLUQuantQ81(s.expertGate, s.expertUp, s.expertW, s.q81Experts, 2048, 6); err != nil {
		return err
	}
	if r.collectors != nil && r.collectors.imatrix != nil {
		if err := r.encodeSwiGLUSlots(s.expertGate, s.expertUp, s.expertW, s.expertMid, 6, 1); err != nil {
			return err
		}
	}
	switch {
	case moe.built() && !r.disableSIMDExperts:
		if err := r.encodeMatvecQ2KSelectedQ81Aligned(l.expertDown, moe.down, s.expertIDs, s.q81Experts, s.expertDown); err != nil {
			return err
		}
	case !r.disableSIMDExperts:
		if err := r.encodeMatvecQ2KSelectedQ81(l.expertDown, s.expertIDs, s.q81Experts, s.expertDown); err != nil {
			return err
		}
	default:
		if err := r.encodeMatvecQ2KSelectedQ81Scalar(l.expertDown, s.expertIDs, s.q81Experts, s.expertDown); err != nil {
			return err
		}
	}
	return r.evalLayerFFNResidual(hc, s, index, position)
}

// evalLayerFFNResidual consumes the completed shared and routed projections.
func (r *cudaRuntime) evalLayerFFNResidual(hc *cudaHC, s *cudaTokenState, index, position int) error {
	if err := r.markCUDAPhase("ffn-post", position, index); err != nil {
		return err
	}
	if err := r.encodeSumRoutedShared(s.expertDown, s.sharedOut, s.ffnOut, cudaWidth); err != nil {
		return err
	}
	if r.steering != nil {
		if err := r.steering.project(s.ffnOut, index, r.steering.ffn); err != nil {
			return err
		}
	}
	return hc.post(s.hc[1], s.ffnOut, s.hc[0])
}

// evalLayerAttention encodes this token's compressor and attention and returns
// the head outputs the feed-forward block projects. compress mutates the layer's
// compressor caches that this token's own attention then reads, and that the
// next token's attention reads in turn, so callers MUST keep this token ordered;
// it is the sequential core a batched front end cannot hoist. Only the
// compressor projections can be, and projected says the caller already has.
func (r *cudaRuntime) evalLayerAttention(s *cudaTokenState, state *cudaSessionLayer, index, position int, projected bool) (cudamem.Slice[float32], error) {
	l := &r.model.layers[index]
	var heads cudamem.Slice[float32]
	if index < 2 {
		if err := r.markCUDAPhase("compression-raw", position, index); err != nil {
			return heads, err
		}
		if err := r.markCUDAPhase("attention", position, index); err != nil {
			return heads, err
		}
		if err := r.encodeRawAttention(s.qHeads, s.kvCur, l.attnSinks, state.rawKV, s.rawScores, s.rawOutput, position); err != nil {
			return heads, err
		}
		if err := r.dumpRawKVSlot(state.rawKV, index, position); err != nil {
			return heads, err
		}
		if err := r.dumpBatchLayerSlice(s.rawOutput.Slice(0, cudaAttentionHeads), index, position, "Qcur"); err != nil {
			return heads, err
		}
		if err := r.dumpBatchLayerSlice(s.rawOutput.Slice(cudaAttentionHeads, cudaAttentionHeads+cudaKVWidth), index, position, "KVcur"); err != nil {
			return heads, err
		}
		return s.rawOutput.Slice(32768+512, 32768+512+32768), nil
	}
	if err := r.markCUDAPhase("compression", position, index); err != nil {
		return heads, err
	}
	if err := r.compress(l, state, s, position, index, projected); err != nil {
		return heads, err
	}
	if err := r.markCUDAPhase("attention", position, index); err != nil {
		return heads, err
	}
	if err := r.encodeMixedAttentionParallel(s.qHeads, s.kvCur, l.attnSinks, state.rawKV, state.attn.cache, state.attn.n, s.allowed, s.mixedScores, s.mixedDenominators, s.mixedOutput, position, index); err != nil {
		return heads, err
	}
	return s.mixedOutput.Slice(32768+512, 32768+512+32768), nil
}

func (r *cudaRuntime) evalLayerProjected(hc *cudaHC, s *cudaTokenState, state *cudaSessionLayer, index, token, position int) error {
	heads, err := r.evalLayerAttention(s, state, index, position, false)
	if err != nil {
		return err
	}

	// Layers 0, 1 and 2 are excluded because their router takes the token id as
	// a kernel argument: they call encodeRouterHash(..., uint32(token)) at :1841
	// where later layers call encodeRouterBiased, which has no such argument. A
	// graph freezes its kernels' arguments at capture, so a captured layer 0 would
	// replay the first token's id for every token after it and route every one of
	// them to that token's experts. Nothing in the launch path would report it.
	useGraph := !r.disableGraphs && r.phaseMark == nil && s == &r.state.cudaTokenState && index >= 3
	if useGraph {
		if r.graphs[index].ffn != nil {
			if err := r.graphs[index].ffn.Launch(r.stream()); err != nil {
				return fmt.Errorf("cuda graph: launch ffn layer %d: %w", index, err)
			}
			r.hcMath.ready = false
			return nil
		}
		var bodyErr error
		g, err := r.captureGraph(func() {
			bodyErr = r.evalLayerFFN(hc, s, index, token, position, heads)
		})
		if err != nil {
			if bodyErr != nil {
				return fmt.Errorf("cuda graph: capture ffn layer %d: %w", index, errors.Join(bodyErr, err))
			}
			return fmt.Errorf("cuda graph: capture ffn layer %d: %w", index, err)
		}
		defer g.Destroy()
		if bodyErr != nil {
			return bodyErr
		}
		exec, err := g.Instantiate()
		if err != nil {
			return fmt.Errorf("cuda graph: instantiate ffn layer %d: %w", index, err)
		}
		r.graphs[index].ffn = exec
		if err := exec.Launch(r.stream()); err != nil {
			return fmt.Errorf("cuda graph: launch initial ffn layer %d: %w", index, err)
		}
		r.hcMath.ready = false
		return nil
	}
	return r.evalLayerFFN(hc, s, index, token, position, heads)
}

func (r *cudaRuntime) evalLayer(index, token, position int) error {
	return r.evalLayerState(&r.state.cudaTokenState, &r.state.layers[index], index, token, position)
}

// compressGroupEligible reports whether the whole group can be pooled at once.
// The batched encoder needs the group to open on a window boundary, and it does
// not carry the per-token indexer selection that compress falls into once the
// indexer cache passes the direct-selection range.
func (r *cudaRuntime) compressGroupEligible(state *cudaSessionLayer, position, n int) bool {
	if r.disableCompressorBatch || r.phaseMark != nil || state.attn.cap == 0 || position%state.attn.ratio != 0 {
		return false
	}
	if state.indexer.cap == 0 {
		return true
	}
	return position%state.indexer.ratio == 0 && state.indexer.n+n/state.indexer.ratio <= 512
}

// compressGroup pools every token of the group into the layer's caches, which
// is what the loop over compress does one token at a time.
func (r *cudaRuntime) compressGroup(l *cudaLayer, state *cudaSessionLayer, d *cudaBatchDense, position, n int) error {
	w := state.attn.width
	emits, err := r.encodeCompressorBatch(d.compressorKV.Slice(0, n*w), d.compressorGate.Slice(0, n*w), l.attnCompressorAPE, l.attnCompressorNorm, &state.attn, d.pooled, position, n, false)
	if err != nil {
		return err
	}
	state.attn.n += emits
	if state.indexer.cap == 0 {
		return nil
	}
	iw := state.indexer.width
	emits, err = r.encodeCompressorBatch(d.indexerKV.Slice(0, n*iw), d.indexerGate.Slice(0, n*iw), l.indexerCompressorAPE, l.indexerCompressorNorm, &state.indexer, d.indexerPooled, position, n, true)
	if err != nil {
		return err
	}
	state.indexer.n += emits
	return nil
}

// compress pools this token's compressor activations into the layer's caches.
// projected reports that the caller already projected them, which the batched
// front end does for the whole group: the projections read only attnNorm, while
// the pooling below is what forces the token order.
func (r *cudaRuntime) compress(l *cudaLayer, state *cudaSessionLayer, s *cudaTokenState, position, index int, projected bool, batchedAllowed ...bool) error {
	skipAllowed := len(batchedAllowed) > 0 && batchedAllowed[0]
	kv, gate := s.compressorKV.Slice(0, state.attn.width), s.compressorGate.Slice(0, state.attn.width)
	if !projected {
		if err := r.encodeF16PairOrdered(l.attnCompressorKV, l.attnCompressorGate, s.attnNorm, kv, gate); err != nil {
			return err
		}
	}
	emit, err := r.encodeCompressor(kv, gate, l.attnCompressorAPE, l.attnCompressorNorm, &state.attn, position, false)
	if err != nil {
		return err
	}
	indexEmit := false
	if state.indexer.cap != 0 {
		if !projected {
			if err := r.encodeF16PairOrdered(l.indexerCompressorKV, l.indexerCompressorGate, s.attnNorm, s.indexerKV, s.indexerGate); err != nil {
				return err
			}
		}
		indexEmit, err = r.encodeCompressor(s.indexerKV, s.indexerGate, l.indexerCompressorAPE, l.indexerCompressorNorm, &state.indexer, position, true)
		if err != nil {
			return err
		}
	}
	if emit {
		state.attn.n++
	}
	if indexEmit {
		state.indexer.n++
	}
	if state.indexer.cap != 0 {
		if state.indexer.n > 512 {
			if err := r.markCUDAPhase("indexer-qb-matvec", position, index); err != nil {
				return err
			}
			var err error
			switch l.indexerQB.typeID {
			case 1:
				err = r.encodeF16MatvecDefault(l.indexerQB, s.qLora, s.indexerQ)
			case 8:
				err = r.encodeQ80Matvec(l.indexerQB, cudamem.Slice[byte]{}, s.qLora, s.indexerQ)
			default:
				err = fmt.Errorf("cuda: indexer query: unsupported weight type %d", l.indexerQB.typeID)
			}
			if err != nil {
				return err
			}
			if err := r.markCUDAPhase("indexer-proj-matvec", position, index); err != nil {
				return err
			}
			if err := r.encodeF16MatvecDefault(l.indexerProj, s.attnNorm, s.indexerWeights); err != nil {
				return err
			}
			return r.encodeIndexer(s.indexerQ, s.indexerWeights, state.indexer.cache, state.indexer.n, s.indexerScratch, s.indexerScores, s.allowed, position, index)
		}
		if skipAllowed {
			return nil
		}
		return r.encodeFillU32(s.allowed, state.indexer.n, 1)
	}
	if skipAllowed {
		return nil
	}
	// An odd layer must not inherit the preceding even layer's selection mask.
	count := max(1, state.attn.n)
	if count > s.allowed.Len() {
		count = s.allowed.Len()
	}
	return r.encodeFillU32(s.allowed, count, 1)
}

func cudaTensorRows(t cudaModelTensor, start, count int) (cudaModelTensor, error) {
	if len(t.dimensions) != 2 || start < 0 || count <= 0 || uint64(start) > t.dimensions[1] || uint64(count) > t.dimensions[1]-uint64(start) {
		return cudaModelTensor{}, fmt.Errorf("cuda: tensor %q: invalid row range", t.name)
	}
	n, err := (gguf.Tensor{Name: t.name, TypeID: t.typeID, Dimensions: t.dimensions}).Bytes()
	if err != nil {
		return cudaModelTensor{}, err
	}
	if n > t.bytes || t.device == 0 {
		return cudaModelTensor{}, fmt.Errorf("cuda: tensor %q: invalid storage", t.name)
	}
	stride := n / t.dimensions[1]
	offset := uint64(start) * stride
	t.device += driver.DevicePtr(offset)
	t.fileOffset += offset
	t.bytes = uint64(count) * stride
	t.dimensions = []uint64{t.dimensions[0], uint64(count)}
	return t, nil
}

// decodeQ80Quantized routes an already-quantized activation to the aligned
// kernel when the artifact plane is there. The shared-down projection is the
// one caller that quantizes its own input, fused into the SwiGLU, so it cannot
// go through encodeQ80Native.
func (r *cudaRuntime) decodeQ80Quantized(t cudaModelTensor, aligned cudamem.Slice[byte], q cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if aligned.Len() > 0 && r.denseArtifactMatvecEnabled() && len(t.dimensions) == 2 {
		return r.encodeQ80MatvecQuantizedAligned(aligned, t.dimensions[0], t.dimensions[1], q, out)
	}
	return r.encodeQ80MatvecQuantized(t, q, out)
}

// encodeQ80Native uses the pinned CUDA single-token fallback quantization.
// aligned is the dense artifact plane for t, or a zero-length slice; it changes
// only where the weights are read from. See decodeQ81.
func (r *cudaRuntime) encodeQ80Native(t cudaModelTensor, aligned cudamem.Slice[byte], in, out cudamem.Slice[float32]) error {
	if in.Len() == 0 || in.Len()%32 != 0 || in.Len()/32*36 > r.state.q80Input.Len() {
		return errors.New("cuda runtime: invalid q8_0 input length")
	}
	q := r.state.q80Input.Slice(0, in.Len()/32*36)
	if err := r.encodeQuantQ80(in, q, in.Len()); err != nil {
		return err
	}
	if aligned.Len() > 0 && r.denseArtifactMatvecEnabled() && len(t.dimensions) == 2 {
		return r.encodeQ80MatvecQuantizedAligned(aligned, t.dimensions[0], t.dimensions[1], q, out)
	}
	return r.encodeQ80MatvecQuantized(t, q, out)
}

// encodeQ80NativeBlockDiag uses the pinned CUDA block-diagonal fallback quantization.
func (r *cudaRuntime) encodeQ80NativeBlockDiag(t cudaModelTensor, aligned cudamem.Slice[byte], in, out cudamem.Slice[float32], diag int) error {
	if in.Len() == 0 || in.Len()%32 != 0 || in.Len()/32*36 > r.state.q80Input.Len() {
		return errors.New("cuda runtime: invalid q8_0 input length")
	}
	q := r.state.q80Input.Slice(0, in.Len()/32*36)
	if err := r.encodeQuantQ80(in, q, in.Len()); err != nil {
		return err
	}
	if aligned.Len() > 0 && r.denseArtifactMatvecEnabled() && len(t.dimensions) == 2 {
		return r.encodeQ80MatvecQuantizedBlockDiagAligned(aligned, t.dimensions[0], t.dimensions[1], q, out, diag)
	}
	return r.encodeQ80MatvecQuantizedBlockDiag(t, q, out, diag)
}

// encodeQ80Matvec matches the default aligned-artifact route for eligible dense
// tensors. Grouped attention output uses encodeQ80NativeBlockDiag explicitly to
// project all 8 head groups in a single grid dispatch.
//
// aligned is the layer's dense artifact plane for t when there is one, and a
// zero-length slice otherwise. It changes only where the weights are read from
// -- see decodeQ81 -- so it is offered after the eligibility test rather than
// before it: a tensor the aligned kernel cannot serve is one the raw-block
// kernel cannot serve either, and both fall back to the same place.
func (r *cudaRuntime) encodeQ80Matvec(t cudaModelTensor, aligned cudamem.Slice[byte], in, out cudamem.Slice[float32]) error {
	if len(t.dimensions) != 2 || t.dimensions[0]%1024 != 0 || t.dimensions[1]%128 != 0 || t.bytes < 2<<20 {
		return r.encodeQ80Native(t, aligned, in, out)
	}
	if in.Len() == 0 || in.Len()%32 != 0 || in.Len()/32*36 > r.state.q80Input.Len() {
		return errors.New("cuda runtime: invalid q8_1 input length")
	}
	q := r.state.q80Input.Slice(0, in.Len()/32*36)
	if err := r.encodeQuantQ81(in, q, in.Len()); err != nil {
		return err
	}
	return r.decodeQ81(t, aligned, q, out)
}

// cudaMoEArtifactLayers is how many layers repack their expert weights once at
// load rather than once per prefill group, from DS4_CUDA_MOE_ARTIFACT_LAYERS.
//
// It defaults to cudaMoEArtifactDefaultLayers. Each layer costs 1.6875 GiB of
// device memory, so two is 3.375 GiB on top of the dense artifacts' 5.62, and
// that pairing is +1.8% over the dense arm alone -- 557.8 against 548.0 tok/s,
// fifteen ABBA rounds with thirty draws per arm, faster in every round, all
// sixty at the same digest -- for +8.8% over neither.
//
// Two is the measured optimum of the curve, not a floor to raise later. The
// non-expert weights are still addressed through host page tables, and holding
// tens of gigabytes of artifacts leaves the page cache no room to serve them.
// Eight layers beside the dense artifacts is 19.12 GiB and about a tenth of
// its draws collapse to a fifth of the throughput; all 43 is 72.56 GiB and
// returns non-finite logits. That is not a fact about the artifacts, which
// hash identical run to run across every one of those failures. Allocating the
// same bytes as ballast, which no kernel reads, breaks the run the same way;
// see allocBallast. The limit is a working set, not a footprint, and it sits
// at 20-24 GiB dormant and near 12 GiB touched.
//
// So this gate and cudaDenseArtifactLayers spend one budget between them until
// every tensor has an artifact and the mapping goes unread. That is also the
// only way past the cliff rather than up to it: the raw mapping is roughly the
// size of the artifacts that would replace it, so repacking everything is
// close to memory-neutral, but only once the mapping is actually released.
// Every run that has failed here held both.
func cudaMoEArtifactLayers() int {
	s := os.Getenv("DS4_CUDA_MOE_ARTIFACT_LAYERS")
	if s == "" {
		if os.Getenv("DS4_CUDA_LOAD_TIME_ARTIFACTS") == "1" {
			return cudaLayers
		}
		return cudaMoEArtifactDefaultLayers
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return cudaMoEArtifactDefaultLayers
	}
	return min(n, cudaLayers)
}

// cudaMoEArtifactDefaultLayers is zero: the expert artifacts are bought back
// for safety, not given up for speed.
//
// Two layers is the optimum of the throughput curve, which rises to two and
// falls past it, and it is worth about +1.8% of prefill -- 557.8 against 548.0
// tok/s over fifteen ABBA rounds at thirty draws per arm, faster in every
// round. A six-draw rerun beside the shipped flags agrees on the sign in all
// three of its rounds and puts the size near +0.2%, which is too little
// evidence to move the number and enough to say the effect is not larger than
// recorded.
//
// The 3.375 GiB those two layers hold is what makes them expensive. Measured
// over 30 draws per arm with the flags on, corruption runs 1/30 at zero layers
// (5.62 GiB of dense artifacts alone), 5/30 at two, and 9/30 at eight --
// monotone in bytes held, trend p=0.0077. The same ABBA rerun shows where the
// bytes go: availGiB at the dense-artifact build is 107.7-108.3 at zero layers
// against 104.1-104.3 at two, a 3.5 GiB difference that matches the artifacts
// to within measurement. That headroom is the currency, because the non-expert
// weights are still addressed through host page tables and a draw that starts
// with less of it is the draw that corrupts.
//
// So this is a trade at a known price: at most 1.8% of prefill against a
// fivefold difference in how often the answer is wrong. It stops being the
// right trade once the mapping goes unread, which is the only way past the
// cliff rather than up to it; see cudaMoEArtifactLayers.
const cudaMoEArtifactDefaultLayers = 0

// buildMoEArtifacts repacks the first n layers' expert weights. It runs before
// the first EvalBatch so the cost lands at load, where it is paid once, rather
// than in every prefill.
func (r *cudaRuntime) buildMoEArtifacts(n int) (err error) {
	n = min(n, cudaLayers)
	if n <= 0 {
		return nil
	}
	start := time.Now()
	stage, err := newCUDAMoEStage(r.device)
	if err != nil {
		return err
	}
	defer func() { err = finishBackendCleanup(err, stage) }()
	built := 0
	for i := range n {
		if r.model.layers[i].expertGate.typeID == typeQ4K {
			continue
		}
		if err := r.buildMoEArtifact(&r.model.layers[i], &r.moeArtifacts[i], stage); err != nil {
			return fmt.Errorf("cuda runtime: moe artifact layer %d: %w", i, err)
		}
		built++
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	relT, relB := r.releasedMapping()
	dirB, bufB := r.stagedBytes()
	fmt.Fprintf(os.Stderr, "ds4: moe artifacts built: layers=%d bytes=%.2fGiB secs=%.2f availGiB=%.1f released=%d/%.2fGiB staged=%.2f/%.2fGiB direct/buffered\n",
		built, float64(built)*cudaMoEArtifactBytes/(1<<30), time.Since(start).Seconds(), memAvailableGiB(),
		relT, float64(relB)/(1<<30), float64(dirB)/(1<<30), float64(bufB)/(1<<30))
	if os.Getenv("DS4_CUDA_MOE_ARTIFACT_VERIFY") != "1" {
		return nil
	}
	buf := make([]byte, 4096)
	for i := range n {
		if !r.moeArtifacts[i].built() {
			continue
		}
		d, err := r.moeArtifacts[i].digest(buf)
		if err != nil {
			return fmt.Errorf("cuda runtime: digest artifact %d: %w", i, err)
		}
		fmt.Fprintf(os.Stderr, "ds4: artifact %d digest %016x\n", i, d)
	}
	return nil
}

// bindDense points the batch's dense weight slices at layer index's artifact
// when it has one and at the per-group scratch buffers otherwise, and reports
// whether an artifact was bound. A false return means the caller still has to
// repack, into the scratch it just bound.
func (r *cudaRuntime) bindDense(d *cudaBatchDense, index int) bool {
	a, scratch := &r.denseArtifacts[index], &d.denseScratch
	if !a.built() {
		a = scratch
	}
	d.denseGate, d.denseUp, d.denseDown = a.gate, a.up, a.down
	d.denseAttnOutB, d.denseAttnOutA = a.attnOutB, a.attnOutA
	d.denseAttnQA, d.denseAttnQB, d.denseAttnKV = a.attnQA, a.attnQB, a.attnKV
	return a != scratch
}

// cudaDenseArtifactLayers is how many layers repack their dense Q8_0 weights
// once at load rather than once per prefill group, from
// DS4_CUDA_DENSE_ARTIFACT_LAYERS.
//
// A layer's dense artifact is 134 MiB, two orders of magnitude below its
// expert artifact, so all 43 cost 5.62 GiB. That is +7.2% -- 549.5 against
// 512.5 tok/s, fifteen ABBA rounds with thirty draws per arm, the arms
// disjoint and the dense arm faster in every round -- with the logit digest
// and the continuation tokens unchanged on all sixty runs. Hence the default.
//
// It defaulted to none for a long time, on a failure rate measured with no
// dense=0 arm beside it. Two controls retired that. In the clean sequence
// above both arms failed zero in thirty draws, so the rate does not reproduce
// and is not distinguishable from the machine's own. Behind a process that
// had committed 72.56 GiB -- the position where the artifact was supposed to
// be dangerous -- dense=43 failed 0 of 8 where dense=0 failed 4 of 8, and the
// artifact's clean draws were the faster ones. It is protective, and the
// mechanism fits: 5.62 GiB of repacked weights on the device is 5.62 GiB the
// kernel does not read back through the model's HMM mapping, which is exactly
// what a large predecessor makes the driver reclaim.
//
// The failure it does not prevent is worth stating, because the artifacts are
// never the corrupted thing. Throughput collapses to a fifth, or holds, and
// the logits come back wrong or non-finite while the continuation tokens stay
// right -- so the digest is the gate, not the tokens. It is not the build,
// which is byte-identical across processes and across a failing run's own
// start and end, and it is not the footprint: DS4_CUDA_BALLAST_GIB holding
// bytes that no kernel reads runs at full speed to 20 GiB. The limit is a
// working set, not a footprint.
//
// DS4_CUDA_DENSE_ARTIFACT_VERIFY prints the artifact digest after the build
// and again at Close; equal digests across a failing run say the bytes were
// intact and move the search to the consumer, which is where it now is. See
// cudaMoEArtifactLayers for the expert side, which pairs with this one for
// +9.3% but has not had its own matched control and so still defaults to none.
func cudaDenseArtifactLayers() int {
	s := os.Getenv("DS4_CUDA_DENSE_ARTIFACT_LAYERS")
	if s == "" {
		return cudaLayers
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return cudaLayers
	}
	return min(n, cudaLayers)
}

// buildDenseArtifacts repacks the first n layers' dense Q8_0 weights before the
// first EvalBatch.
func (r *cudaRuntime) buildDenseArtifacts(n int) (err error) {
	n = min(n, cudaLayers)
	if n <= 0 {
		return nil
	}
	// No denseD2REnabled check here: DS4_CUDA_DENSE_D2R can be set after the
	// runtime is constructed, so the answer at this point does not bind the
	// answer at use. bindDense is reached only from the paths D2R guards
	// anyway, so repacking unconditionally costs a gate nobody reads.
	start := time.Now()
	stage, err := newCUDAMoEStage(r.device)
	if err != nil {
		return err
	}
	defer func() { err = finishBackendCleanup(err, stage) }()
	for i := range n {
		if err := r.buildDenseArtifact(&r.model.layers[i], &r.denseArtifacts[i], stage); err != nil {
			return fmt.Errorf("cuda runtime: dense artifact layer %d: %w", i, err)
		}
	}
	if err := r.buildDenseTensor(&r.denseOutput, &r.model.output.weight, stage); err != nil {
		return fmt.Errorf("cuda runtime: dense artifact output: %w", err)
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	dir, buf := r.stagedBytes()
	fmt.Fprintf(os.Stderr, "ds4: dense artifacts built: layers=%d bytes=%.2fGiB output=%.2fGiB secs=%.2f availGiB=%.1f staged=%.2f/%.2fGiB direct/buffered\n",
		n, float64(n)*cudaDenseArtifactBytes/(1<<30), float64(r.denseOutput.Len())/(1<<30), time.Since(start).Seconds(), memAvailableGiB(),
		float64(dir)/(1<<30), float64(buf)/(1<<30))
	r.reportDenseArtifactDigest("built")
	return nil
}

// reportDenseArtifactDigest prints one digest over every built dense artifact
// under DS4_CUDA_DENSE_ARTIFACT_VERIFY. Called once after the build and once at
// Close, it separates an artifact that was built wrong from one that was
// overwritten afterwards: only the second changes the digest between the two.
func (r *cudaRuntime) reportDenseArtifactDigest(when string) {
	if os.Getenv("DS4_CUDA_DENSE_ARTIFACT_VERIFY") != "1" {
		return
	}
	h, buf, n := uint64(14695981039346656037), make([]byte, 1024), 0
	for i := range r.denseArtifacts {
		if !r.denseArtifacts[i].built() {
			continue
		}
		v, err := r.denseArtifacts[i].digest(h, buf)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ds4: dense artifact digest %s: layer %d: %v\n", when, i, err)
			return
		}
		h, n = v, n+1
	}
	fmt.Fprintf(os.Stderr, "ds4: dense artifacts digest %s: layers=%d digest=%016x\n", when, n, h)
}

// allocBallast allocates DS4_CUDA_BALLAST_GIB gibibytes of device memory that
// nothing ever reads or writes.
//
// The artifact levers hold gigabytes of device memory next to a model that is
// still addressed through host page tables, and past a few gigabytes the
// prefill answer stops being reproducible. Two mechanisms explain that equally
// well: the artifacts could be consumed wrongly, or the allocation alone could
// be enough to break reads of the mapping. Ballast tells them apart, because it
// allocates without giving any kernel something new to read.
func (r *cudaRuntime) allocBallast() error {
	gib := r.ballastGib
	if !r.ballastSet {
		s := os.Getenv("DS4_CUDA_BALLAST_GIB")
		if s == "" {
			return nil
		}
		var err error
		gib, err = strconv.ParseFloat(s, 64)
		if err != nil || gib <= 0 {
			return nil
		}
	}
	n := int(gib * (1 << 30))
	// One block by default, or DS4_CUDA_BALLAST_CHUNKS of them. The chunk
	// count is the shape control: the dense artifact is 344 separate
	// allocations interleaved with the stage buffer, not one contiguous
	// block, so a single-block ballast holds the same bytes in a shape the
	// allocator never sees in the case under test.
	chunks := 1
	if v, err := strconv.Atoi(os.Getenv("DS4_CUDA_BALLAST_CHUNKS")); err == nil && v > 0 {
		chunks = v
	}
	for i := range chunks {
		lo, hi := i*n/chunks, (i+1)*n/chunks
		b, err := cudamem.Alloc[byte](r.device, hi-lo)
		if err != nil {
			return fmt.Errorf("cuda runtime: ballast %.2fGiB chunk %d of %d: %w", gib, i, chunks, err)
		}
		r.ballast = append(r.ballast, b)
	}
	// Write it once. cudaMalloc on an integrated device need not commit
	// physical pages until they are touched, and ballast that occupies no
	// memory measures nothing -- the first run of this control was fast at
	// 19.1 GiB where the same bytes of artifact ran at a quarter the speed,
	// which is what an uncommitted allocation looks like. Nothing reads it
	// after this; the point is the footprint, not the contents.
	const words = 1 << 20
	if err := r.withContext(func() error {
		for _, b := range r.ballast {
			for off := 0; off < b.Len(); off += words * 4 {
				w := uint((min(off+words*4, b.Len()) - off) / 4)
				if err := (b.Ptr() + driver.DevicePtr(off)).MemsetD32(0, w); err != nil {
					return err
				}
			}
		}
		return nil
	}); err != nil {
		return fmt.Errorf("cuda runtime: commit ballast: %w", err)
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "ds4: ballast allocated: bytes=%.2fGiB chunks=%d availGiB=%.1f\n", gib, chunks, memAvailableGiB())
	return nil
}

// memAvailableGiB reports MemAvailable from /proc/meminfo, which on an
// integrated device is the budget the artifacts and the model's page cache
// share. It returns 0 when the file cannot be read.
func memAvailableGiB() float64 {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		f := strings.Fields(line)
		if len(f) < 2 {
			return 0
		}
		kb, err := strconv.ParseFloat(f[1], 64)
		if err != nil {
			return 0
		}
		return kb / (1 << 20)
	}
	return 0
}
