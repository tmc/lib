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
	disableGraphs           bool
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
	attnPre, qLora, qHeads, kvRaw, kvCur cudamem.Slice[float32]
	attnNorm                             cudamem.Slice[float32]
	compressorKV, compressorGate         cudamem.Slice[float32]
	indexerKV, indexerGate               cudamem.Slice[float32]
	attnLow, attnOut                     cudamem.Slice[float32]
	sharedGate, sharedUp, sharedOut      cudamem.Slice[float32]
	q81Attn, q81QLora, q81FFN            cudamem.Slice[byte]
	q80Heads, q80Low, q80Shared          cudamem.Slice[byte]
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
	dense      cudaBatchDense
}

// cudaPrefillBatchWidth matches the compile-time maximum batch width of the
// batched matvec kernel (cudaQ81MaxBatch).
const cudaPrefillBatchWidth = cudaQ81MaxBatch

// The group also runs the Q8_0, block-diagonal and F16 pair batch kernels at
// this width. Each declaration below fails to compile if that kernel's maximum
// ever falls below it, which a wider group would otherwise truncate.
const (
	_ = uint(cudaQ80MaxBatch - cudaPrefillBatchWidth)
	_ = uint(cudaQ80BlockDiagMaxBatch - cudaPrefillBatchWidth)
	_ = uint(cudaF16PairMaxBatch - cudaPrefillBatchWidth)
)

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
	for _, h := range b.hc {
		errs = append(errs, h.Close())
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
		n := min(cudaPrefillBatchWidth, len(tokens)-i)
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

// evalBatch spans the chunk rather than each token in it: the batched
// front end evaluates all of them together, so a per-token interval here
// would report the chunk's duration once per token.
func (r *cudaRuntime) evalBatch(tokens []int, position int, logits bool) (err error) {
	defer beginTraced(levelToken, "batch", nil).end()
	// The last phase in the group has no following mark to close it.
	defer closePhaseSpan()
	if len(tokens) < 2 || len(tokens) > cudaPrefillBatchWidth {
		return errors.New("cuda runtime: dense batch has invalid length")
	}
	defer func() {
		if err != nil {
			r.failed = err
		}
	}()
	if err := r.ensureBatch(func() (*cudaBatchResources, error) {
		return newCUDABatchResources(r, cudaPrefillBatchWidth)
	}); err != nil {
		return err
	}
	rows := make([]*cudaTokenState, len(tokens))
	hc := make([]*cudaHC, len(tokens))
	for i := 0; i < len(tokens)-1; i++ {
		rows[i], hc[i] = &r.batch.rows[i], r.batch.hc[i]
	}
	rows[len(tokens)-1], hc[len(tokens)-1] = &r.state.cudaTokenState, r.hcMath
	for i, token := range tokens {
		for stream := range 4 {
			if err := r.encodeEmbeddingRow(r.model.output.embedding, uint32(token), rows[i].hc[0].Slice(stream*cudaWidth, (stream+1)*cudaWidth)); err != nil {
				return err
			}
		}
	}
	for layer := range r.model.layers {
		if err := r.evalLayerBatch(rows, hc, layer, tokens, position); err != nil {
			return fmt.Errorf("cuda runtime: layer %d: %w", layer, err)
		}
	}
	last := rows[len(rows)-1]
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
	alloc := func(name string, n int) cudamem.Slice[float32] {
		return cudamem.AllocIn[float32](b.denseArena, name, n)
	}
	b.dense = cudaBatchDense{
		attnPre:        alloc("batch-dense-attention-pre", width*2*cudaWidth),
		qLora:          alloc("batch-dense-query-lora", width*1024),
		qHeads:         alloc("batch-dense-query-heads", width*64*cudaKVWidth),
		kvRaw:          alloc("batch-dense-kv-raw", width*cudaKVWidth),
		kvCur:          alloc("batch-dense-kv-current", width*cudaKVWidth),
		attnNorm:       alloc("batch-dense-attention-norm", width*cudaWidth),
		compressorKV:   alloc("batch-dense-compressor-kv", width*1024),
		compressorGate: alloc("batch-dense-compressor-gate", width*1024),
		indexerKV:      alloc("batch-dense-indexer-kv", width*256),
		indexerGate:    alloc("batch-dense-indexer-gate", width*256),
		attnLow:        alloc("batch-dense-attention-low", width*8192),
		attnOut:        alloc("batch-dense-attention-output", width*cudaWidth),
		sharedGate:     alloc("batch-dense-shared-gate", width*cudaSharedFFNWidth),
		sharedUp:       alloc("batch-dense-shared-up", width*cudaSharedFFNWidth),
		sharedOut:      alloc("batch-dense-shared-output", width*cudaWidth),
		q81Attn:        cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-attention", width*cudaWidth/32*36),
		q81QLora:       cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-query-lora", width*1024/32*36),
		q81FFN:         cudamem.AllocIn[byte](b.denseArena, "batch-dense-q81-ffn", width*cudaWidth/32*36),
		q80Heads:       cudamem.AllocIn[byte](b.denseArena, "batch-dense-q80-attention-heads", width*32768/32*36),
		q80Low:         cudamem.AllocIn[byte](b.denseArena, "batch-dense-q80-attention-low", width*8192/32*36),
		q80Shared:      cudamem.AllocIn[byte](b.denseArena, "batch-dense-q80-shared", width*cudaSharedFFNWidth/32*36),
	}
	if err := errors.Join(b.arena.Err(), b.denseArena.Err()); err != nil {
		return nil, fmt.Errorf("cuda runtime: allocate batch rows: %w", err)
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
	if b == nil || len(b.rows) < cudaPrefillBatchWidth-1 || len(b.hc) < cudaPrefillBatchWidth-1 {
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
	if len(base) != len(tokens) || len(hc) != len(tokens) || len(tokens) < 2 || len(tokens) > cudaPrefillBatchWidth {
		return errors.New("cuda runtime: dense batch has invalid length")
	}
	l := &r.model.layers[index]
	state := &r.state.layers[index]
	d := &r.batch.dense
	if err := r.markCUDAPhase("dense-front", position, index); err != nil {
		return err
	}
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
		rows[i].attnLow = d.attnLow.Slice(i*8192, (i+1)*8192)
		rows[i].attnOut = d.attnOut.Slice(i*cudaWidth, (i+1)*cudaWidth)
		rows[i].q81Input = d.q81FFN.Slice(i*cudaWidth/32*36, (i+1)*cudaWidth/32*36)
		rows[i].sharedGate = d.sharedGate.Slice(i*cudaSharedFFNWidth, (i+1)*cudaSharedFFNWidth)
		rows[i].sharedUp = d.sharedUp.Slice(i*cudaSharedFFNWidth, (i+1)*cudaSharedFFNWidth)
		rows[i].sharedOut = d.sharedOut.Slice(i*cudaWidth, (i+1)*cudaWidth)
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
		if err := hc[i].pre(rows[i].hc[0], l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, rows[i].attnPre, false); err != nil {
			return fmt.Errorf("token %d attention pre: %w", i, err)
		}
	}
	n := len(rows)
	attnNorm := d.attnPre.Slice(cudaWidth, n*2*cudaWidth)
	q81Attn := d.q81Attn.Slice(0, n*cudaWidth/32*36)
	qLora := d.qLora.Slice(0, n*1024)
	q81QLora := d.q81QLora.Slice(0, n*1024/32*36)
	qHeads := d.qHeads.Slice(0, n*64*cudaKVWidth)
	kvRaw := d.kvRaw.Slice(0, n*cudaKVWidth)
	if err := r.encodeQuantQ81Batch(attnNorm, q81Attn, cudaWidth, 2*cudaWidth, n); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81Batch(l.attnQA, q81Attn, qLora, n); err != nil {
		return err
	}
	for i := range rows {
		if err := r.encodeRMS(rows[i].qLora, l.attnQANorm, rows[i].qLora, 1, true); err != nil {
			return fmt.Errorf("token %d query lora norm: %w", i, err)
		}
	}
	if err := r.encodeQuantQ81Batch(qLora, q81QLora, 1024, 1024, n); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81Batch(l.attnQB, q81QLora, qHeads, n); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81Batch(l.attnKV, q81Attn, kvRaw, n); err != nil {
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
	// Attention is token ordered: compress mutates the compressor caches that the
	// following tokens read. Only the head outputs leave this loop, each in its
	// own token's buffer, so the projections that consume them batch below.
	heads := make([]cudamem.Slice[float32], n)
	for i := range rows {
		if err := r.markCUDAPhase("qkv-normalization", position+i, index); err != nil {
			return err
		}
		if err := r.encodeRMS(rows[i].qHeads, cudaModelTensor{}, rows[i].qHeads, 64, false); err != nil {
			return fmt.Errorf("token %d query head norm: %w", i, err)
		}
		if err := r.encodeRMS(rows[i].kvRaw, l.attnKVNorm, rows[i].kvCur, 1, true); err != nil {
			return fmt.Errorf("token %d key/value norm: %w", i, err)
		}
		var err error
		if heads[i], err = r.evalLayerAttention(&rows[i], state, index, position+i, true); err != nil {
			return fmt.Errorf("token %d: %w", i, err)
		}
	}
	// The attention output weights do not depend on the token either. The
	// qualification fallback keeps its per-token kernels, which is the whole
	// point of that switch, so the group projects with it token by token.
	projected := !r.disableBlockDiagAttnOut
	if projected {
		if err := r.evalBatchAttnOut(rows, heads, index, position); err != nil {
			return err
		}
	}
	for i := range rows {
		if err := r.evalLayerFFNPre(hc[i], &rows[i], index, tokens[i], position+i, heads[i], projected); err != nil {
			return fmt.Errorf("token %d: %w", i, err)
		}
	}
	// The shared expert's weights do not depend on the token, and each token's
	// projection reads only its own activation row, so the group streams each
	// weight matrix once instead of n times. This is the same amortization the
	// dense front end above already gets for attnQA, attnQB and attnKV.
	q81FFN := d.q81FFN.Slice(0, n*cudaWidth/32*36)
	if err := r.encodeQ80MatvecQ81Batch(l.sharedGate, q81FFN, d.sharedGate.Slice(0, n*cudaSharedFFNWidth), n); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81Batch(l.sharedUp, q81FFN, d.sharedUp.Slice(0, n*cudaSharedFFNWidth), n); err != nil {
		return err
	}
	// The shared expert's down projection reads only this token's activation for
	// the same reason, so it streams sharedDown once as well.
	if err := r.evalBatchSharedDown(rows, index, position); err != nil {
		return err
	}
	for i := range rows {
		if err := r.evalLayerFFNPost(hc[i], &rows[i], index, position+i, true); err != nil {
			return fmt.Errorf("token %d: %w", i, err)
		}
	}
	return nil
}

// evalBatchCompressor projects the group's compressor activations with one
// launch per weight pair. The batched kernel reads the group's activation rows
// adjacent, and pre writes each token's norm into the second half of its own
// attnPre block, so the rows are gathered first; the gather moves one token's
// activation where the launch it replaces reads the whole weight pair again.
func (r *cudaRuntime) evalBatchCompressor(rows []cudaTokenState, index int) error {
	l, state, d := &r.model.layers[index], &r.state.layers[index], &r.batch.dense
	n := len(rows)
	for i := range rows {
		if err := d.attnNorm.Slice(i*cudaWidth, (i+1)*cudaWidth).CopyDeviceToDeviceAsync(rows[i].attnNorm, r.stream()); err != nil {
			return fmt.Errorf("token %d compressor activation: %w", i, err)
		}
	}
	in := d.attnNorm.Slice(0, n*cudaWidth)
	w := state.attn.width
	if err := r.encodeF16PairOrderedBatch(l.attnCompressorKV, l.attnCompressorGate, in,
		d.compressorKV.Slice(0, n*w), d.compressorGate.Slice(0, n*w), n); err != nil {
		return err
	}
	if state.indexer.cap == 0 {
		return nil
	}
	w = state.indexer.width
	return r.encodeF16PairOrderedBatch(l.indexerCompressorKV, l.indexerCompressorGate, in,
		d.indexerKV.Slice(0, n*w), d.indexerGate.Slice(0, n*w), n)
}

// evalBatchAttnOut projects the group's attention outputs. Each token reads only
// its own head outputs, so the group streams attnOutA and attnOutB once instead
// of once per token.
func (r *cudaRuntime) evalBatchAttnOut(rows []cudaTokenState, heads []cudamem.Slice[float32], index, position int) error {
	l, d := &r.model.layers[index], &r.batch.dense
	n := len(rows)
	if err := r.markCUDAPhase("attention-output", position, index); err != nil {
		return err
	}
	// The head outputs stay in each token's own attention buffer, so they are
	// quantized one token at a time, as the per-token path does. A token whose
	// width disagrees fails here rather than shifting the rows that follow.
	width := heads[0].Len()
	stride := width / 32 * 36
	for i := range rows {
		if err := r.encodeQuantQ80(heads[i], d.q80Heads.Slice(i*stride, (i+1)*stride), width); err != nil {
			return fmt.Errorf("token %d attention output quantization: %w", i, err)
		}
	}
	lowWidth := rows[0].attnLow.Len()
	low := d.attnLow.Slice(0, n*lowWidth)
	if err := r.encodeQ80MatvecQuantizedBlockDiagBatch(l.attnOutA, d.q80Heads.Slice(0, n*stride), low, 8, n); err != nil {
		return err
	}
	// The pinned attention-output/HC fusion uses native float32 Q8_0 scales.
	// These rows are adjacent and each 32-element quantization block is
	// independent, so one launch writes the bytes n per-token launches would.
	q := d.q80Low.Slice(0, n*lowWidth/32*36)
	if err := r.encodeQuantQ80(low, q, lowWidth); err != nil {
		return err
	}
	return r.encodeQ80MatvecQuantizedBatch(l.attnOutB, q, d.attnOut.Slice(0, n*cudaWidth), n)
}

// evalBatchSharedDown projects the group through the shared expert's down
// matrix. The SwiGLU activation reads only each token's own shared gate and up
// rows, which the group already holds token major, so one quantization launch
// feeds one pass over sharedDown.
func (r *cudaRuntime) evalBatchSharedDown(rows []cudaTokenState, index, position int) error {
	l, d := &r.model.layers[index], &r.batch.dense
	n := len(rows)
	// The mark opens the group's second half, where the per-token form opens
	// each token's; it precedes the projection it measures either way.
	if err := r.markCUDAPhase("expert-ffn", position, index); err != nil {
		return err
	}
	width := n * cudaSharedFFNWidth
	q := d.q80Shared.Slice(0, width/32*36)
	if err := r.encodeSwiGLUQuantQ80(d.sharedGate.Slice(0, width), d.sharedUp.Slice(0, width), q, width, 1); err != nil {
		return err
	}
	return r.encodeQ80MatvecQuantizedBatch(l.sharedDown, q, d.sharedOut.Slice(0, n*cudaWidth), n)
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
	if err := r.evalLayerFFNShared(s, index); err != nil {
		return err
	}
	return r.evalLayerFFNPost(hc, s, index, position, false)
}

// evalLayerFFNPre encodes the attention output projection, the FFN norm and the
// router, ending with this token's Q8_1 activation quantization. It writes only
// this token's state; nothing here reads another token in the group. projected
// reports that the caller already encoded this token's attention output, which
// the batched front end hoists into one launch per weight matrix.
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
	if err := hc.pre(s.hc[1], l.hcFFNFn, l.hcFFNScale, l.hcFFNBase, l.ffnNorm, s.ffnPre, false); err != nil {
		return err
	}
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
	return r.encodeQuantQ81(s.ffnNorm, s.q81Input, cudaWidth)
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
func (r *cudaRuntime) compress(l *cudaLayer, state *cudaSessionLayer, s *cudaTokenState, position, index int, projected bool) error {
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
		}
		return r.encodeIndexer(s.indexerQ, s.indexerWeights, state.indexer.cache, state.indexer.n, s.indexerScratch, s.indexerScores, s.allowed, position, index)
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
