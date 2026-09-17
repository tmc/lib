//go:build linux

package main

import (
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaRuntime owns one model mapping and one token timeline. All layer math
// runs on the device; only the final logits are copied to the host.
type cudaRuntime struct {
	*cudaFoundation
	model                   *cudaModel
	arena                   *cudamem.Arena
	batch                   *cudaBatchResources
	state                   cudaSessionState
	graphs                  [cudaLayers]cudaLayerGraphs
	hcMath                  *cudaHC
	ctx, pos                int
	logits                  []float32
	failed                  error
	batchProgress           func(int) error
	disableBlockDiagAttnOut bool
	disableSIMDExperts      bool
	disableGatherExperts    bool
	disableGraphs           bool
	enableAttentionHMMA     bool
}

type cudaLayerGraphs struct {
	qkv *cudagraph.Exec
	ffn *cudagraph.Exec
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
	d2rGate, d2rUp, d2rDown              cudamem.Slice[byte]
	d2rLayer                             int
	q81MMQ                               cudamem.Slice[byte]
	downQ8                               cudamem.Slice[byte]
	d2rWork                              cudamem.Slice[int32]
	d2rNItems                            cudamem.Slice[int32]
	idsSrc                               cudamem.Slice[int32]
	ffnOut                               cudamem.Slice[float32]
	q81Experts                           cudamem.Slice[byte]
	allowed                              cudamem.Slice[uint32]
	tokenNComp                           cudamem.Slice[uint32]
	mixedScores                          cudamem.Slice[float32]
	mixedDenominators                    cudamem.Slice[float32]
	ttRecords                            cudamem.Slice[byte]
	ttCounts                             cudamem.Slice[uint32]
	ttRawMirror                          cudamem.Slice[byte]
	ttCompMirror                         cudamem.Slice[byte]
}

// cudaSharedFFNWidth is the shared expert's intermediate width, matching the
// per-token shared-gate and shared-up allocations in allocTokenState. A model
// whose shared FFN disagrees is rejected by encodeQ80MatvecQ81Batch's extent
// check rather than silently truncated.
const cudaSharedFFNWidth = 2048

type cudaBatchResources struct {
	arena      *cudamem.Arena
	denseArena *cudamem.Arena
	rows       []cudaTokenState
	hc         []*cudaHC
	hcBatch    *cudaHCBatch
	dense      cudaBatchDense
	grouping   *cudaMoEGrouping
}

// cudaPrefillGroupWidth is the number of tokens prefill holds resident at
// once. Batched projection kernels tile these rows internally.
const cudaPrefillGroupWidth = 256

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
)

// Full groups tile exactly into chunks; the final group may be shorter.
// evalBatch routes single tokens to evalPosition.
const (
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillQ81Chunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillQ80Chunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillBlockDiagChunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillF16PairChunk)
	_ = uint(0 - cudaPrefillGroupWidth%cudaPrefillAttnOutChunk)
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

func newCUDARuntime(path string, limit int) (_ *cudaRuntime, err error) {
	if limit <= 1 || uint64(limit) > math.MaxUint32 {
		return nil, errors.New("cuda runtime: invalid context length")
	}
	foundation, err := newCUDAFoundation(path)
	if err != nil {
		return nil, err
	}
	r := &cudaRuntime{
		cudaFoundation: foundation,
		ctx:            limit,
		disableGraphs:  os.Getenv("DS4_DISABLE_CUDA_GRAPH") == "1",
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, r.Close())
		}
	}()
	r.model, err = foundation.bindModel()
	if err != nil {
		return nil, err
	}
	r.arena = cudamem.NewArena(foundation.device)
	r.allocState()
	if err := r.arena.Err(); err != nil {
		return nil, err
	}
	r.hcMath, err = newCUDAHC(foundation)
	if err != nil {
		return nil, err
	}
	r.logits = make([]float32, r.state.logits.Len())
	if err := r.Reset(); err != nil {
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
	if r == nil || r.arena == nil || r.device == nil {
		return errors.New("cuda runtime: closed")
	}
	r.failed = errors.New("cuda runtime: reset incomplete")
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
	}
	clear(r.logits)
	r.pos, r.failed = 0, nil
	return nil
}

func (r *cudaRuntime) Close() error {
	if r == nil {
		return nil
	}
	var errs []error
	if r.cudaFoundation != nil && r.device != nil {
		errs = append(errs, syncStream(r.stream()))
	}
	for i := range r.graphs {
		if r.graphs[i].qkv != nil {
			errs = append(errs, r.graphs[i].qkv.Destroy())
			r.graphs[i].qkv = nil
		}
		if r.graphs[i].ffn != nil {
			errs = append(errs, r.graphs[i].ffn.Destroy())
			r.graphs[i].ffn = nil
		}
	}
	if r.hcMath != nil {
		errs = append(errs, r.hcMath.Close())
		r.hcMath = nil
	}
	if r.arena != nil {
		errs = append(errs, r.arena.Free())
		r.arena = nil
	}
	if r.batch != nil {
		errs = append(errs, r.batch.Close())
		r.batch = nil
	}
	if err := r.cudaFoundation.Close(); err != nil {
		errs = append(errs, err)
	} else {
		r.cudaFoundation = nil
	}
	return errors.Join(errs...)
}

func (b *cudaBatchResources) Close() error {
	if b == nil {
		return nil
	}
	var errs []error
	if b.grouping != nil {
		errs = append(errs, b.grouping.Free())
		b.grouping = nil
	}
	if b.hcBatch != nil {
		errs = append(errs, b.hcBatch.Close())
		b.hcBatch = nil
	}
	for _, h := range b.hc {
		errs = append(errs, h.Close())
	}
	if b.dense.d2rGate.Len() > 0 {
		errs = append(errs, b.dense.d2rGate.Free())
		b.dense.d2rGate = cudamem.Slice[byte]{}
	}
	if b.dense.d2rUp.Len() > 0 {
		errs = append(errs, b.dense.d2rUp.Free())
		b.dense.d2rUp = cudamem.Slice[byte]{}
	}
	if b.dense.d2rDown.Len() > 0 {
		errs = append(errs, b.dense.d2rDown.Free())
		b.dense.d2rDown = cudamem.Slice[byte]{}
	}
	if b.arena != nil {
		errs = append(errs, b.arena.Free())
		b.arena = nil
	}
	if b.denseArena != nil {
		errs = append(errs, b.denseArena.Free())
		b.denseArena = nil
	}
	return errors.Join(errs...)
}

func (r *cudaRuntime) Logits() []float32 {
	if r == nil || r.arena == nil || r.pos == 0 || r.failed != nil {
		return nil
	}
	return r.logits
}

func (r *cudaRuntime) Eval(token, position int) error { return r.evalPosition(token, position, true) }
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
	i := 0
	// Keep cache updates token ordered while batching the dense front end of
	// each chunk. Only the final chunk needs the vocabulary projection.
	for i < len(tokens) {
		n := min(cudaPrefillGroupWidth, len(tokens)-i)
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

// activeHCBatch keeps qualification fallbacks on their scalar HC path even
// when a previous evaluation allocated batch resources.
func (r *cudaRuntime) activeHCBatch() *cudaHCBatch {
	if r.batch == nil || r.disableGatherExperts || r.disableSIMDExperts || r.disableBlockDiagAttnOut {
		return nil
	}
	return r.batch.hcBatch
}

func (r *cudaRuntime) d2rEnabled() bool {
	return r.enableD2RExperts || os.Getenv("DS4_CUDA_MOE_D2R") == "1"
}

func (r *cudaRuntime) attentionHMMAEnabled() bool {
	if r == nil {
		return false
	}
	return r.enableAttentionHMMA || os.Getenv("DS4_CUDA_ATTENTION_HMMA") == "1"
}

// evalBatch spans the chunk rather than each token in it: the batched
// front end evaluates all of them together, so a per-token interval here
// would report the chunk's duration once per token.
func (r *cudaRuntime) evalBatch(tokens []int, position int, logits bool) (err error) {
	defer beginTraced(levelToken, "batch", nil).end()
	// The last phase in the group has no following mark to close it.
	defer closePhaseSpan()
	if len(tokens) < 2 || len(tokens) > cudaPrefillGroupWidth {
		return errors.New("cuda runtime: dense batch has invalid length")
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
		for stream := range 4 {
			if err := r.encodeEmbeddingRow(r.model.output.embedding, uint32(token), rows[i].hc[0].Slice(stream*cudaWidth, (stream+1)*cudaWidth)); err != nil {
				return err
			}
		}
	}
	if err := r.batch.dense.tokens.Slice(0, len(tokens)).CopyFromHost(tokensU32); err != nil {
		return err
	}
	for layer := range r.model.layers {
		if err := r.evalLayerBatch(rows, hc, layer, tokens, position); err != nil {
			return fmt.Errorf("cuda runtime: layer %d: %w", layer, err)
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
		if err := r.hcMath.pre(last.hc[0], r.model.output.hcFn, r.model.output.hcScale, r.model.output.hcBase, r.model.output.norm, last.outputMixed, true); err != nil {
			return err
		}
		if err := r.encodeQ80Matvec(r.model.output.weight, last.outputMixed.Slice(4096, 8192), last.logits); err != nil {
			return err
		}
	}
	if err := syncStream(r.stream()); err != nil {
		return err
	}
	if logits {
		if err := last.logits.CopyToHost(r.logits); err != nil {
			return err
		}
		for i, v := range r.logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("cuda runtime: non-finite logit %d", i)
			}
		}
	}
	r.pos += len(tokens)
	return nil
}

func newCUDABatchResources(r *cudaRuntime, width int) (_ *cudaBatchResources, err error) {
	if r == nil || r.device == nil || width < 2 {
		return nil, errors.New("cuda runtime: invalid batch allocation")
	}
	b := &cudaBatchResources{
		arena:      cudamem.NewArena(r.device),
		denseArena: cudamem.NewArena(r.device),
		rows:       make([]cudaTokenState, width-1),
		hc:         make([]*cudaHC, width-1),
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, b.Close())
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
	if err := errors.Join(b.arena.Err(), b.denseArena.Err()); err != nil {
		return nil, fmt.Errorf("cuda runtime: allocate batch rows: %w", err)
	}
	b.grouping, err = newCUDAMoEGrouping(r.device, cudaMoEMaxExperts, width*6)
	if err != nil {
		return nil, fmt.Errorf("cuda runtime: allocate batch grouping: %w", err)
	}
	for i := range b.hc {
		b.hc[i], err = newCUDAHC(r.cudaFoundation)
		if err != nil {
			return nil, fmt.Errorf("cuda runtime: allocate batch HC %d: %w", i, err)
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
	if r.d2rEnabled() {
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
	return b, nil
}

func (r *cudaRuntime) ensureBatch(allocate func() (*cudaBatchResources, error)) error {
	if r.batch != nil {
		return nil
	}
	b, err := allocate()
	if err != nil {
		if b != nil {
			err = errors.Join(err, b.Close())
		}
		return err
	}
	if b == nil || len(b.rows) < cudaPrefillGroupWidth-1 || len(b.hc) < cudaPrefillGroupWidth-1 {
		if b != nil {
			err = b.Close()
		}
		return errors.Join(errors.New("cuda runtime: incomplete batch allocation"), err)
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
		if r.activeHCBatch() == nil {
			if err := hc[i].pre(rows[i].hc[0], l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, rows[i].attnPre, false); err != nil {
				return fmt.Errorf("token %d attention pre: %w", i, err)
			}
		}
	}
	n := len(rows)
	if b := r.activeHCBatch(); b != nil {
		if err := b.pre(d.hc[0].Slice(0, n*cudaHCWidth), l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, d.attnPre.Slice(0, n*2*cudaWidth), n); err != nil {
			return err
		}
	}
	if err := chunkRows(n, cudaPrefillQ81Chunk, func(lo, hi int) error {
		m := hi - lo
		attnNorm := d.attnPre.Slice(lo*2*cudaWidth+cudaWidth, hi*2*cudaWidth)
		q81Attn := d.q81Attn.Slice(lo*cudaWidth/32*36, hi*cudaWidth/32*36)
		qLora := d.qLora.Slice(lo*1024, hi*1024)
		q81QLora := d.q81QLora.Slice(lo*1024/32*36, hi*1024/32*36)
		qHeads := d.qHeads.Slice(lo*64*cudaKVWidth, hi*64*cudaKVWidth)
		kvRaw := d.kvRaw.Slice(lo*cudaKVWidth, hi*cudaKVWidth)
		if err := r.encodeQuantQ81Batch(attnNorm, q81Attn, cudaWidth, 2*cudaWidth, m); err != nil {
			return err
		}
		if err := r.encodeQ80MatvecQ81Batch(l.attnQA, q81Attn, qLora, m); err != nil {
			return err
		}
		if err := r.encodeRMS(qLora, l.attnQANorm, qLora, m, true); err != nil {
			return fmt.Errorf("query lora norm: %w", err)
		}
		if err := r.encodeQuantQ81Batch(qLora, q81QLora, 1024, 1024, m); err != nil {
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
	if err := r.markCUDAPhase("qkv-normalization", position, index); err != nil {
		return err
	}
	qHeadsAll := d.qHeads.Slice(0, n*64*cudaKVWidth)
	if err := r.encodeRMS(qHeadsAll, cudaModelTensor{}, qHeadsAll, n*64, false); err != nil {
		return fmt.Errorf("batch query head norm: %w", err)
	}
	kvRawAll := d.kvRaw.Slice(0, n*cudaKVWidth)
	kvCurAll := d.kvCur.Slice(0, n*cudaKVWidth)
	if err := r.encodeRMS(kvRawAll, l.attnKVNorm, kvCurAll, n, true); err != nil {
		return fmt.Errorf("batch key/value norm: %w", err)
	}
	// Attention is token ordered: compress mutates the compressor caches that the
	// following tokens read. Head outputs occupy strided rows of the dense
	// attention buffer, so quantization and projection can batch below.
	heads := make([]cudamem.Slice[float32], n)
	if index < 2 || r.disableMixedAttentionFused {
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
		for i := range rows {
			if err := r.markCUDAPhase("compression", position+i, index); err != nil {
				return err
			}
			if err := r.compress(l, state, &rows[i], position+i, index, true, true); err != nil {
				return fmt.Errorf("token %d: %w", i, err)
			}
			nCompHost[i] = uint32(state.attn.n)
			tot := min(position+i+1, rawCap) + int(nCompHost[i])
			if tot > maxTotal {
				maxTotal = tot
			}
			if err := r.markCUDAPhase("attention", position+i, index); err != nil {
				return err
			}
			heads[i] = d.attention.Slice(i*cudaAttentionStride+cudaAttentionScratch, i*cudaAttentionStride+cudaAttentionScratch+cudaAttentionHeads)
		}
		if err := d.tokenNComp.Slice(0, n).CopyFromHost(nCompHost); err != nil {
			return fmt.Errorf("copy token n_comp: %w", err)
		}

		qHeadsAll := d.qHeads.Slice(0, n*64*cudaKVWidth)
		kvCurAll := d.kvCur.Slice(0, n*cudaKVWidth)
		outAll := d.attention.Slice(0, n*cudaAttentionStride)
		if err := r.mixedAttentionPrepareBatch(qHeadsAll, kvCurAll, outAll, position, rawCap, n); err != nil {
			return fmt.Errorf("batch mixed attention prepare: %w", err)
		}
		if r.attentionHMMAEnabled() && n >= 4 {
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
				outAll,
				d.ttRawMirror,
				d.ttCompMirror,
				d.ttRecords,
				d.ttCounts,
				position,
				cudaAttentionStride,
				cudaAttentionStride,
				recStride,
				n,
				64,
				rawRowMin,
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
	// The attention output weights do not depend on the token either. The
	// qualification fallback keeps its per-token kernels, which is the whole
	// point of that switch, so the group projects with it token by token.
	projected := !r.disableBlockDiagAttnOut
	if projected {
		if err := r.evalBatchAttnOut(rows, index, position); err != nil {
			return err
		}
	}
	if b := r.activeHCBatch(); b != nil {
		if err := b.post(d.hc[0].Slice(0, n*cudaHCWidth), d.attnOut.Slice(0, n*cudaWidth), d.hc[1].Slice(0, n*cudaHCWidth), n); err != nil {
			return err
		}
		if err := r.markCUDAPhase("ffn-pre", position, index); err != nil {
			return err
		}
		if err := b.pre(d.hc[1].Slice(0, n*cudaHCWidth), l.hcFFNFn, l.hcFFNScale, l.hcFFNBase, l.ffnNorm, d.ffnPre.Slice(0, n*2*cudaWidth), n); err != nil {
			return err
		}
	} else {
		for i := range rows {
			if err := r.evalLayerFFNPre(hc[i], &rows[i], index, tokens[i], position+i, heads[i], projected); err != nil {
				return fmt.Errorf("token %d: %w", i, err)
			}
		}
	}
	if err := chunkRows(n, cudaPrefillGroupWidth, func(lo, hi int) error {
		m := hi - lo
		ffnNorm := d.ffnPre.Slice(lo*2*cudaWidth+cudaWidth, hi*2*cudaWidth)
		routerOut := d.router.Slice(lo*256, hi*256)
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
	for i := range rows {
		if err := r.markCUDAPhase("shared-ffn", position+i, index); err != nil {
			return err
		}
	}
	if err := chunkRows(n, cudaPrefillQ81Chunk, func(lo, hi int) error {
		m := hi - lo
		ffnNorm := d.ffnPre.Slice(lo*2*cudaWidth+cudaWidth, hi*2*cudaWidth)
		q81FFN := d.q81FFN.Slice(lo*cudaWidth/32*36, hi*cudaWidth/32*36)
		return r.encodeQuantQ81Batch(ffnNorm, q81FFN, cudaWidth, 2*cudaWidth, m)
	}); err != nil {
		return err
	}
	if gather && r.d2rEnabled() && d.d2rGate.Len() > 0 {
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
		for i := range rows {
			if err := r.markCUDAPhase("ffn-post", position+i, index); err != nil {
				return err
			}
			if r.activeHCBatch() == nil {
				if err := hc[i].post(rows[i].hc[1], rows[i].ffnOut, rows[i].hc[0]); err != nil {
					return fmt.Errorf("token %d: %w", i, err)
				}
			}
		}
		if b := r.activeHCBatch(); b != nil {
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

// evalBatchExperts projects all routed experts while keeping the outputs in
// token and slot order for the weighted activation and residual kernels.
func (r *cudaRuntime) evalBatchExperts(rows []cudaTokenState, index int) error {
	l := &r.model.layers[index]
	d := &r.batch.dense
	n := len(rows)
	// This view borrows the allocation; the batch owns and frees it.
	g := *r.batch.grouping
	g.npairs = 6 * n
	g.pairs = g.pairs.Slice(0, g.npairs)
	if err := r.encodeMoEGroupCSR(d.expertIDs.Slice(0, g.npairs), &g); err != nil {
		return err
	}
	if r.d2rEnabled() && d.d2rGate.Len() > 0 {
		if d.d2rLayer != index {
			if err := r.repackLayerMoE(l, d.d2rGate, d.d2rUp, d.d2rDown); err != nil {
				return fmt.Errorf("repack layer %d: %w", index, err)
			}
			d.d2rLayer = index
		}
		if err := r.encodeD2RMakeIdsSrc(g.pairs, d.idsSrc.Slice(0, g.npairs), g.npairs); err != nil {
			return err
		}
		capacity32 := (g.npairs+31)/32 + 256
		if err := r.encodeD2RBuildWorklist32(g.offsets, d.d2rWork.Slice(0, capacity32), d.d2rNItems, 256); err != nil {
			return err
		}
		q81MMQ := d.q81MMQ.Slice(0, (cudaWidth/128)*n*144)
		downQ8 := d.downQ8.Slice(0, (2048/128)*g.npairs*144)
		if err := r.encodeGateUpD2R(
			d.d2rGate, d.d2rUp,
			q81MMQ, d.idsSrc.Slice(0, g.npairs), g.pairs, g.offsets,
			d.expertW.Slice(0, g.npairs),
			d.d2rWork.Slice(0, capacity32), d.d2rNItems,
			downQ8,
			2048, cudaWidth, g.npairs, n, 256, 10.0,
		); err != nil {
			return fmt.Errorf("d2r gateup: %w", err)
		}
		capacity64 := (g.npairs+63)/64 + 256
		if err := r.encodeD2RBuildWorklist64(g.offsets, d.d2rWork.Slice(0, capacity64), d.d2rNItems, 256); err != nil {
			return err
		}
		if err := r.encodeDownD2R(
			d.d2rDown, downQ8,
			g.pairs, g.offsets,
			d.d2rWork.Slice(0, capacity64), d.d2rNItems,
			d.expertDown.Slice(0, g.npairs*cudaWidth),
			cudaWidth, 2048, g.npairs, 256,
		); err != nil {
			return fmt.Errorf("d2r down: %w", err)
		}
		return nil
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
		w := state.attn.width
		if err := r.encodeF16PairOrderedBatch(l.attnCompressorKV, l.attnCompressorGate, in,
			d.compressorKV.Slice(lo*w, hi*w), d.compressorGate.Slice(lo*w, hi*w), m); err != nil {
			return err
		}
		if state.indexer.cap == 0 {
			return nil
		}
		w = state.indexer.width
		return r.encodeF16PairOrderedBatch(l.indexerCompressorKV, l.indexerCompressorGate, in,
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
		// The SwiGLU quantization is elementwise over the chunk's rows, which
		// are adjacent, so it spans the whole chunk rather than one row.
		from, to := lo*cudaSharedFFNWidth, hi*cudaSharedFFNWidth
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
		if err := r.evalLayerState(&r.state.cudaTokenState, &r.state.layers[layer], layer, token, position); err != nil {
			return fmt.Errorf("cuda runtime: layer %d: %w", layer, err)
		}
	}
	if logits {
		if err := r.hcMath.pre(r.state.hc[0], r.model.output.hcFn, r.model.output.hcScale, r.model.output.hcBase, r.model.output.norm, r.state.outputMixed, true); err != nil {
			return err
		}
	}
	if logits {
		if err := r.encodeQ80Matvec(r.model.output.weight, r.state.outputMixed.Slice(4096, 8192), r.state.logits); err != nil {
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
		for i, v := range r.logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("cuda runtime: non-finite logit %d", i)
			}
		}
	}
	r.pos++
	return nil
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
	if err := r.encodeQ80MatvecQ81(l.attnQA, qAttn, s.qLora); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81(l.attnKV, qAttn, s.kvRaw); err != nil {
		return err
	}
	if err := r.encodeRMSPair(s.qLora, l.attnQANorm, s.qLora, s.kvRaw, l.attnKVNorm, s.kvCur); err != nil {
		return err
	}
	if err := r.encodeQ80Matvec(l.attnQB, s.qLora, s.qHeads); err != nil {
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
			g, err := cudagraph.Capture(r.stream(), func() {
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
	return r.evalLayerProjected(r.hcMath, s, state, index, token, position)
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
	l := &r.model.layers[index]
	if !projected {
		if err := r.markCUDAPhase("attention-output", position, index); err != nil {
			return err
		}
		if !r.disableBlockDiagAttnOut {
			if err := r.encodeQ80NativeBlockDiag(l.attnOutA, heads, s.attnLow, 8); err != nil {
				return err
			}
		} else {
			for group := range 8 {
				rows, err := cudaTensorRows(l.attnOutA, group*1024, 1024)
				if err != nil {
					return err
				}
				if err := r.encodeQ80Native(rows, heads.Slice(group*4096, (group+1)*4096), s.attnLow.Slice(group*1024, (group+1)*1024)); err != nil {
					return err
				}
			}
		}
		// The pinned attention-output/HC fusion uses native float32 Q8_0 scales.
		if err := r.encodeQ80Native(l.attnOutB, s.attnLow, s.attnOut); err != nil {
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
	l := &r.model.layers[index]
	if err := r.encodeQ80MatvecQ81(l.sharedGate, s.q81Input, s.sharedGate); err != nil {
		return err
	}
	return r.encodeQ80MatvecQ81(l.sharedUp, s.q81Input, s.sharedUp)
}

// evalLayerFFNPost consumes the shared projections and encodes the routed
// experts and the residual. It reads only this token's state. projected reports
// that the caller already encoded this token's shared-down projection, which the
// batched front end hoists into one launch for the group.
func (r *cudaRuntime) evalLayerFFNPost(hc *cudaHC, s *cudaTokenState, index, position int, projected bool) error {
	l := &r.model.layers[index]
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
		if err := r.encodeQ80MatvecQuantized(l.sharedDown, q, s.sharedOut); err != nil {
			return err
		}
	}
	if !r.disableSIMDExperts {
		if err := r.encodeMatvecIQ2SelectedQ81(l.expertGate, l.expertUp, s.expertIDs, s.q81Input, s.expertGate, s.expertUp); err != nil {
			return err
		}
	} else {
		if err := r.encodeMatvecIQ2SelectedQ81Scalar(l.expertGate, l.expertUp, s.expertIDs, s.q81Input, s.expertGate, s.expertUp); err != nil {
			return err
		}
	}
	if err := r.encodeSwiGLUQuantQ81(s.expertGate, s.expertUp, s.expertW, s.q81Experts, 2048, 6); err != nil {
		return err
	}
	if !r.disableSIMDExperts {
		if err := r.encodeMatvecQ2KSelectedQ81(l.expertDown, s.expertIDs, s.q81Experts, s.expertDown); err != nil {
			return err
		}
	} else {
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
		g, err := cudagraph.Capture(r.stream(), func() {
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
				err = r.encodeQ80Matvec(l.indexerQB, s.qLora, s.indexerQ)
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

// encodeQ80Native uses the pinned CUDA single-token fallback quantization.
func (r *cudaRuntime) encodeQ80Native(t cudaModelTensor, in, out cudamem.Slice[float32]) error {
	if in.Len() == 0 || in.Len()%32 != 0 || in.Len()/32*36 > r.state.q80Input.Len() {
		return errors.New("cuda runtime: invalid q8_0 input length")
	}
	q := r.state.q80Input.Slice(0, in.Len()/32*36)
	if err := r.encodeQuantQ80(in, q, in.Len()); err != nil {
		return err
	}
	return r.encodeQ80MatvecQuantized(t, q, out)
}

// encodeQ80NativeBlockDiag uses the pinned CUDA block-diagonal fallback quantization.
func (r *cudaRuntime) encodeQ80NativeBlockDiag(t cudaModelTensor, in, out cudamem.Slice[float32], diag int) error {
	if in.Len() == 0 || in.Len()%32 != 0 || in.Len()/32*36 > r.state.q80Input.Len() {
		return errors.New("cuda runtime: invalid q8_0 input length")
	}
	q := r.state.q80Input.Slice(0, in.Len()/32*36)
	if err := r.encodeQuantQ80(in, q, in.Len()); err != nil {
		return err
	}
	return r.encodeQ80MatvecQuantizedBlockDiag(t, q, out, diag)
}

// encodeQ80Matvec matches the default aligned-artifact route for eligible dense
// tensors. Grouped attention output uses encodeQ80NativeBlockDiag explicitly to
// project all 8 head groups in a single grid dispatch.
func (r *cudaRuntime) encodeQ80Matvec(t cudaModelTensor, in, out cudamem.Slice[float32]) error {
	if len(t.dimensions) != 2 || t.dimensions[0]%1024 != 0 || t.dimensions[1]%128 != 0 || t.bytes < 2<<20 {
		return r.encodeQ80Native(t, in, out)
	}
	if in.Len() == 0 || in.Len()%32 != 0 || in.Len()/32*36 > r.state.q80Input.Len() {
		return errors.New("cuda runtime: invalid q8_1 input length")
	}
	q := r.state.q80Input.Slice(0, in.Len()/32*36)
	if err := r.encodeQuantQ81(in, q, in.Len()); err != nil {
		return err
	}
	return r.encodeQ80MatvecQ81(t, q, out)
}
