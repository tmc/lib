//go:build linux

package main

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaRuntime owns one model mapping and one token timeline. All layer math
// runs on the device; only the final logits are copied to the host.
type cudaRuntime struct {
	*cudaFoundation
	model           *cudaModel
	arena           *cudamem.Arena
	batchArena      *cudamem.Arena
	batchDenseArena *cudamem.Arena
	state           cudaSessionState
	batchFirst      cudaTokenState
	batchDense      cudaBatchDense
	hcMath          *cudaHC
	batchHC         *cudaHC
	ctx, pos        int
	logits          []float32
	failed          error
}

type cudaSessionState struct {
	cudaTokenState
	layers [cudaLayers]cudaSessionLayer
}

// cudaTokenState holds one token row's transient layer activations. Layer
// history is session state and deliberately lives outside this type.
type cudaTokenState struct {
	hc                                                           [2]cudamem.Slice[float32]
	attnPre, attnNorm, qLora, qHeads, kvRaw, kvCur               cudamem.Slice[float32]
	compressorKV, compressorGate, indexerKV, indexerGate         cudamem.Slice[float32]
	indexerQ, indexerWeights, indexerScratch, indexerScores      cudamem.Slice[float32]
	rawScores, rawOutput, mixedScores, mixedOutput               cudamem.Slice[float32]
	allowed                                                      cudamem.Slice[uint32]
	attnLow, attnOut, ffnPre, ffnNorm, router, probs             cudamem.Slice[float32]
	expertIDs                                                    cudamem.Slice[int32]
	expertW, sharedGate, sharedUp, sharedMid                     cudamem.Slice[float32]
	expertGate, expertUp, expertMid, expertDown                  cudamem.Slice[float32]
	ffnBlocks, sharedOut, routedOut, ffnOut, outputMixed, logits cudamem.Slice[float32]
	q80Input, q81Input, q81Experts                               cudamem.Slice[byte]
}

type cudaSessionLayer struct {
	rawKV         cudamem.Slice[float32]
	attn, indexer cudaCompressedState
}

type cudaBatchDense struct {
	attnPre, qLora, qHeads, kvRaw, kvCur cudamem.Slice[float32]
	q81Attn, q81QLora                    cudamem.Slice[byte]
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
	r := &cudaRuntime{cudaFoundation: foundation, ctx: limit}
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
	s.q80Input = cudamem.AllocIn[byte](a, "q80-input", 8192/32*36)
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
	if err := syncStream(r.device.DefaultStream()); err != nil {
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
		errs = append(errs, syncStream(r.device.DefaultStream()))
	}
	if r.hcMath != nil {
		errs = append(errs, r.hcMath.Close())
		r.hcMath = nil
	}
	if r.batchHC != nil {
		errs = append(errs, r.batchHC.Close())
		r.batchHC = nil
	}
	if r.arena != nil {
		errs = append(errs, r.arena.Free())
		r.arena = nil
	}
	if r.batchArena != nil {
		errs = append(errs, r.batchArena.Free())
		r.batchArena = nil
	}
	if r.batchDenseArena != nil {
		errs = append(errs, r.batchDenseArena.Free())
		r.batchDenseArena = nil
	}
	if err := r.cudaFoundation.Close(); err != nil {
		errs = append(errs, err)
	} else {
		r.cudaFoundation = nil
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
	// each adjacent pair. Only the final pair needs the vocabulary projection.
	for ; i+1 < len(tokens); i += 2 {
		if err := r.evalBatch2(tokens[i:i+2], position+i, i+2 == len(tokens)); err != nil {
			return err
		}
	}
	if i < len(tokens) {
		return r.evalPosition(tokens[i], position+i, true)
	}
	return nil
}

func (r *cudaRuntime) evalBatch2(tokens []int, position int, logits bool) (err error) {
	if len(tokens) != 2 {
		return errors.New("cuda runtime: two-token batch has invalid length")
	}
	defer func() {
		if err != nil {
			r.failed = err
		}
	}()
	if err := r.ensureBatchFirst(func() (*cudamem.Arena, cudaTokenState, error) {
		a := cudamem.NewArena(r.device)
		var row cudaTokenState
		r.allocTokenState(a, &row, "batch-first-")
		return a, row, a.Err()
	}); err != nil {
		return err
	}
	if err := r.ensureBatchDense(); err != nil {
		return err
	}
	if err := r.ensureBatchHC(); err != nil {
		return err
	}
	rows := []*cudaTokenState{&r.batchFirst, &r.state.cudaTokenState}
	for i, token := range tokens {
		for stream := range 4 {
			if err := r.encodeEmbeddingRow(r.model.output.embedding, uint32(token), rows[i].hc[0].Slice(stream*cudaWidth, (stream+1)*cudaWidth)); err != nil {
				return err
			}
		}
	}
	for layer := range r.model.layers {
		if err := r.evalLayerBatch2(rows, layer, tokens, position); err != nil {
			return fmt.Errorf("cuda runtime: layer %d: %w", layer, err)
		}
	}
	last := rows[len(rows)-1]
	if logits {
		if err := r.hcMath.pre(last.hc[0], r.model.output.hcFn, r.model.output.hcScale, r.model.output.hcBase, r.model.output.norm, last.outputMixed, true); err != nil {
			return err
		}
		if err := r.encodeQ80Matvec(r.model.output.weight, last.outputMixed.Slice(4096, 8192), last.logits); err != nil {
			return err
		}
	}
	if err := syncStream(r.device.DefaultStream()); err != nil {
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

func (r *cudaRuntime) ensureBatchDense() error {
	if r.batchDenseArena != nil {
		return nil
	}
	a := cudamem.NewArena(r.device)
	alloc := func(name string, n int) cudamem.Slice[float32] {
		return cudamem.AllocIn[float32](a, name, n)
	}
	d := cudaBatchDense{
		attnPre:  alloc("batch-dense-attention-pre", 2*2*cudaWidth),
		qLora:    alloc("batch-dense-query-lora", 2*1024),
		qHeads:   alloc("batch-dense-query-heads", 2*64*cudaKVWidth),
		kvRaw:    alloc("batch-dense-kv-raw", 2*cudaKVWidth),
		kvCur:    alloc("batch-dense-kv-current", 2*cudaKVWidth),
		q81Attn:  cudamem.AllocIn[byte](a, "batch-dense-q81-attention", 2*cudaWidth/32*36),
		q81QLora: cudamem.AllocIn[byte](a, "batch-dense-q81-query-lora", 2*1024/32*36),
	}
	if err := a.Err(); err != nil {
		return errors.Join(fmt.Errorf("cuda runtime: allocate dense batch rows: %w", err), a.Free())
	}
	r.batchDenseArena, r.batchDense = a, d
	return nil
}

func (r *cudaRuntime) ensureBatchHC() error {
	if r.batchHC != nil {
		return nil
	}
	h, err := newCUDAHC(r.cudaFoundation)
	if err != nil {
		return fmt.Errorf("cuda runtime: allocate batch HC: %w", err)
	}
	r.batchHC = h
	return nil
}

func (r *cudaRuntime) evalLayerBatch2(base []*cudaTokenState, index int, tokens []int, position int) error {
	if len(base) != 2 || len(tokens) != 2 {
		return errors.New("cuda runtime: dense two-token batch has invalid length")
	}
	l := &r.model.layers[index]
	d := &r.batchDense
	rows := [2]cudaTokenState{*base[0], *base[1]}
	hc := [2]*cudaHC{r.batchHC, r.hcMath}
	for i := range rows {
		rows[i].attnPre = d.attnPre.Slice(i*2*cudaWidth, (i+1)*2*cudaWidth)
		rows[i].attnNorm = rows[i].attnPre.Slice(cudaWidth, 2*cudaWidth)
		rows[i].qLora = d.qLora.Slice(i*1024, (i+1)*1024)
		rows[i].qHeads = d.qHeads.Slice(i*64*cudaKVWidth, (i+1)*64*cudaKVWidth)
		rows[i].kvRaw = d.kvRaw.Slice(i*cudaKVWidth, (i+1)*cudaKVWidth)
		rows[i].kvCur = d.kvCur.Slice(i*cudaKVWidth, (i+1)*cudaKVWidth)
		if err := hc[i].pre(rows[i].hc[0], l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, rows[i].attnPre, false); err != nil {
			return fmt.Errorf("token %d attention pre: %w", i, err)
		}
	}
	attnNorm := d.attnPre.Slice(cudaWidth, 4*cudaWidth)
	if err := r.encodeQuantQ81Batch(attnNorm, d.q81Attn, cudaWidth, 2*cudaWidth, 2); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81Batch(l.attnQA, d.q81Attn, d.qLora, 2); err != nil {
		return err
	}
	for i := range rows {
		if err := r.encodeRMS(rows[i].qLora, l.attnQANorm, rows[i].qLora, 1, true); err != nil {
			return fmt.Errorf("token %d query lora norm: %w", i, err)
		}
	}
	if err := r.encodeQuantQ81Batch(d.qLora, d.q81QLora, 1024, 1024, 2); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81Batch(l.attnQB, d.q81QLora, d.qHeads, 2); err != nil {
		return err
	}
	if err := r.encodeQ80MatvecQ81Batch(l.attnKV, d.q81Attn, d.kvRaw, 2); err != nil {
		return err
	}
	for i := range rows {
		if err := r.encodeRMS(rows[i].qHeads, cudaModelTensor{}, rows[i].qHeads, 64, false); err != nil {
			return fmt.Errorf("token %d query head norm: %w", i, err)
		}
		if err := r.encodeRMS(rows[i].kvRaw, l.attnKVNorm, rows[i].kvCur, 1, true); err != nil {
			return fmt.Errorf("token %d key/value norm: %w", i, err)
		}
		if err := r.evalLayerProjected(hc[i], &rows[i], &r.state.layers[index], index, tokens[i], position+i); err != nil {
			return fmt.Errorf("token %d: %w", i, err)
		}
	}
	return nil
}

func (r *cudaRuntime) ensureBatchFirst(allocate func() (*cudamem.Arena, cudaTokenState, error)) error {
	if r.batchArena != nil {
		return nil
	}
	a, row, err := allocate()
	if err != nil {
		var freeErr error
		if a != nil {
			freeErr = a.Free()
		}
		return errors.Join(fmt.Errorf("cuda runtime: allocate batch rows: %w", err), freeErr)
	}
	if a == nil {
		return errors.New("cuda runtime: allocate batch rows: missing arena")
	}
	r.batchArena, r.batchFirst = a, row
	return nil
}

func (r *cudaRuntime) evalPosition(token, position int, logits bool) (err error) {
	defer begin(levelToken, "token", nil).end()
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
	if err := syncStream(r.device.DefaultStream()); err != nil {
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

func (r *cudaRuntime) evalLayerState(s *cudaTokenState, state *cudaSessionLayer, index, token, position int) error {
	defer beginLayer(index).end()
	l := &r.model.layers[index]
	if err := r.hcMath.pre(s.hc[0], l.hcAttnFn, l.hcAttnScale, l.hcAttnBase, l.attnNorm, s.attnPre, false); err != nil {
		return err
	}
	if err := r.encodeQ80Matvec(l.attnQA, s.attnNorm, s.qLora); err != nil {
		return err
	}
	if err := r.encodeRMS(s.qLora, l.attnQANorm, s.qLora, 1, true); err != nil {
		return err
	}
	if err := r.encodeQ80Matvec(l.attnQB, s.qLora, s.qHeads); err != nil {
		return err
	}
	if err := r.encodeRMS(s.qHeads, cudaModelTensor{}, s.qHeads, 64, false); err != nil {
		return err
	}
	if err := r.encodeQ80Matvec(l.attnKV, s.attnNorm, s.kvRaw); err != nil {
		return err
	}
	if err := r.encodeRMS(s.kvRaw, l.attnKVNorm, s.kvCur, 1, true); err != nil {
		return err
	}
	return r.evalLayerProjected(r.hcMath, s, state, index, token, position)
}

func (r *cudaRuntime) evalLayerProjected(hc *cudaHC, s *cudaTokenState, state *cudaSessionLayer, index, token, position int) error {
	l := &r.model.layers[index]
	var heads cudamem.Slice[float32]
	if index < 2 {
		if err := r.encodeRawAttention(s.qHeads, s.kvCur, l.attnSinks, state.rawKV, s.rawScores, s.rawOutput, position); err != nil {
			return err
		}
		heads = s.rawOutput.Slice(32768+512, 32768+512+32768)
	} else {
		if err := r.compress(l, state, s, position); err != nil {
			return err
		}
		if err := r.encodeMixedAttention(s.qHeads, s.kvCur, l.attnSinks, state.rawKV, state.attn.cache, state.attn.n, s.allowed, s.mixedScores, s.mixedOutput, position, index); err != nil {
			return err
		}
		heads = s.mixedOutput.Slice(32768+512, 32768+512+32768)
	}
	for group := range 8 {
		rows, err := cudaTensorRows(l.attnOutA, group*1024, 1024)
		if err != nil {
			return err
		}
		if err := r.encodeQ80Native(rows, heads.Slice(group*4096, (group+1)*4096), s.attnLow.Slice(group*1024, (group+1)*1024)); err != nil {
			return err
		}
	}
	// The pinned attention-output/HC fusion uses native float32 Q8_0 scales.
	if err := r.encodeQ80Native(l.attnOutB, s.attnLow, s.attnOut); err != nil {
		return err
	}
	if err := hc.post(s.hc[0], s.attnOut, s.hc[1]); err != nil {
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
	if err := r.encodeQ80Matvec(l.sharedGate, s.ffnNorm, s.sharedGate); err != nil {
		return err
	}
	if err := r.encodeQ80Matvec(l.sharedUp, s.ffnNorm, s.sharedUp); err != nil {
		return err
	}
	if err := r.encodeSwiGLU(s.sharedGate, s.sharedUp, s.sharedMid, 1); err != nil {
		return err
	}
	// The pinned fused shared-down/HC path uses native float32 Q8_0 scales.
	if err := r.encodeQ80Native(l.sharedDown, s.sharedMid, s.sharedOut); err != nil {
		return err
	}
	if err := r.encodeQuantQ81(s.ffnNorm, s.q81Input, cudaWidth); err != nil {
		return err
	}
	if err := r.encodeMatvecIQ2SelectedQ81(l.expertGate, l.expertUp, s.expertIDs, s.q81Input, s.expertGate, s.expertUp); err != nil {
		return err
	}
	if err := r.encodeSwiGLUSlots(s.expertGate, s.expertUp, s.expertW, s.expertMid, 6, 1); err != nil {
		return err
	}
	if err := r.encodeQuantQ81(s.expertMid, s.q81Experts, 2048); err != nil {
		return err
	}
	if err := r.encodeMatvecQ2KSelectedQ81(l.expertDown, s.expertIDs, s.q81Experts, s.expertDown); err != nil {
		return err
	}
	if err := r.encodeSumVectors(s.expertDown, s.routedOut, 6); err != nil {
		return err
	}
	if err := r.encodeSumVectors(s.ffnBlocks, s.ffnOut, 2); err != nil {
		return err
	}
	if err := hc.post(s.hc[1], s.ffnOut, s.hc[0]); err != nil {
		return err
	}
	return nil
}

func (r *cudaRuntime) evalLayer(index, token, position int) error {
	return r.evalLayerState(&r.state.cudaTokenState, &r.state.layers[index], index, token, position)
}

func (r *cudaRuntime) compress(l *cudaLayer, state *cudaSessionLayer, s *cudaTokenState, position int) error {
	kv, gate := s.compressorKV.Slice(0, state.attn.width), s.compressorGate.Slice(0, state.attn.width)
	if err := r.encodeF16PairOrdered(l.attnCompressorKV, l.attnCompressorGate, s.attnNorm, kv, gate); err != nil {
		return err
	}
	emit, err := r.encodeCompressor(kv, gate, l.attnCompressorAPE, l.attnCompressorNorm, &state.attn, position, false)
	if err != nil {
		return err
	}
	indexEmit := false
	if state.indexer.cap != 0 {
		if err := r.encodeF16PairOrdered(l.indexerCompressorKV, l.indexerCompressorGate, s.attnNorm, s.indexerKV, s.indexerGate); err != nil {
			return err
		}
		indexEmit, err = r.encodeCompressor(s.indexerKV, s.indexerGate, l.indexerCompressorAPE, l.indexerCompressorNorm, &state.indexer, position, true)
		if err != nil {
			return err
		}
	}
	if err := syncStream(r.device.DefaultStream()); err != nil {
		return err
	}
	if emit {
		state.attn.n++
	}
	if indexEmit {
		state.indexer.n++
	}
	if state.indexer.cap != 0 {
		if state.indexer.n > 512 {
			if err := r.encodeQ80Matvec(l.indexerQB, s.qLora, s.indexerQ); err != nil {
				return err
			}
			if err := r.encodeF16MatvecDefault(l.indexerProj, s.attnNorm, s.indexerWeights); err != nil {
				return err
			}
		}
		return r.encodeIndexer(s.indexerQ, s.indexerWeights, state.indexer.cache, state.indexer.n, s.indexerScratch, s.indexerScores, s.allowed, position)
	}
	// An odd layer must not inherit the preceding even layer's selection mask.
	allowed := make([]uint32, s.allowed.Len())
	for i := range allowed {
		allowed[i] = 1
	}
	return s.allowed.CopyFromHost(allowed)
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

// encodeQ80Matvec matches the default aligned-artifact route for eligible dense
// tensors. Grouped attention output uses encodeQ80Native explicitly.
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
