//go:build darwin && arm64

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

const (
	ds4Layers  = 43
	ds4Width   = 4096
	ds4HCWidth = 4 * ds4Width
	ds4KVWidth = 512
	ds4RawCap  = 128
)

// metalRuntime owns the immutable model bindings and mutable buffers for one
// DS4 session.
type metalRuntime struct {
	file                *os.File
	dev                 *libmetal.Device
	views               *metalModelViews
	ops                 *metalPinnedOps
	arena               *metalmem.Arena
	model               *ggufFile
	layers              [ds4Layers]metalLayer
	output              metalOutput
	state               metalSessionState
	ctx                 int
	pos                 int
	lastOpeningBatch    metalBatchStats
	lastRawOpeningBatch metalBatchStats
	lastAttentionBatch  metalBatchStats
	lastFFNBatches      metalBatchStats
	lastRoutedBatch     metalBatchStats
}

var _ sessionBackend = (*metalRuntime)(nil)

type metalLayer struct {
	attnSinks boundMetalTensor

	attnCompressorAPE  boundMetalTensor
	attnCompressorNorm boundMetalTensor
	attnCompressorGate boundMetalTensor
	attnCompressorKV   boundMetalTensor

	indexerCompressorAPE  boundMetalTensor
	indexerCompressorNorm boundMetalTensor
	indexerCompressorGate boundMetalTensor
	indexerCompressorKV   boundMetalTensor
	indexerProj           boundMetalTensor
	indexerQB             boundMetalTensor

	attnKV     boundMetalTensor
	attnKVNorm boundMetalTensor
	attnNorm   boundMetalTensor
	attnOutA   boundMetalTensor
	attnOutB   boundMetalTensor
	attnQA     boundMetalTensor
	attnQANorm boundMetalTensor
	attnQB     boundMetalTensor

	expertBias  boundMetalTensor
	expertMap   boundMetalTensor
	expertGate  boundMetalTensor
	expertUp    boundMetalTensor
	expertDown  boundMetalTensor
	router      boundMetalTensor
	sharedGate  boundMetalTensor
	sharedUp    boundMetalTensor
	sharedDown  boundMetalTensor
	ffnNorm     boundMetalTensor
	hcAttnBase  boundMetalTensor
	hcAttnFn    boundMetalTensor
	hcAttnScale boundMetalTensor
	hcFFNBase   boundMetalTensor
	hcFFNFn     boundMetalTensor
	hcFFNScale  boundMetalTensor
}

type metalOutput struct {
	embedding boundMetalTensor
	hcBase    boundMetalTensor
	hcFn      boundMetalTensor
	hcScale   boundMetalTensor
	norm      boundMetalTensor
	weight    boundMetalTensor
}

type metalCompressedState struct {
	cache      metalmem.Slice[float32]
	stateKV    metalmem.Slice[float32]
	stateScore metalmem.Slice[float32]
	pooled     metalmem.Slice[float32]
	headDim    int
	width      int
	ratio      int
	cap        int
	n          int
}

type metalLayerState struct {
	rawKV   metalmem.Slice[float32]
	attn    metalCompressedState
	indexer metalCompressedState
}

type metalSessionState struct {
	hc            [2]metalmem.Slice[float32]
	attnPre       metalmem.Slice[float32]
	attnCur       metalmem.Slice[float32]
	attnNorm      metalmem.Slice[float32]
	qLora         metalmem.Slice[float32]
	qHeads        metalmem.Slice[float32]
	kvRaw         metalmem.Slice[float32]
	kvCur         metalmem.Slice[float32]
	heads         metalmem.Slice[float32]
	rawScores     metalmem.Slice[float32]
	rawOutput     metalmem.Slice[float32]
	attnLow       metalmem.Slice[float32]
	attnOut       metalmem.Slice[float32]
	ffnPre        metalmem.Slice[float32]
	ffnCur        metalmem.Slice[float32]
	ffnNorm       metalmem.Slice[float32]
	router        metalmem.Slice[float32]
	probs         metalmem.Slice[float32]
	expertIDs     metalmem.Slice[int32]
	expertW       metalmem.Slice[float32]
	sharedGate    metalmem.Slice[float32]
	sharedUp      metalmem.Slice[float32]
	sharedMid     metalmem.Slice[float32]
	sharedOut     metalmem.Slice[float32]
	expertGate    metalmem.Slice[float32]
	expertUp      metalmem.Slice[float32]
	expertGateRaw metalmem.Slice[float32]
	expertUpRaw   metalmem.Slice[float32]
	expertMid     metalmem.Slice[float32]
	expertDown    metalmem.Slice[float32]
	routedOut     metalmem.Slice[float32]
	ffnBlocks     metalmem.Slice[float32]
	ffnOut        metalmem.Slice[float32]
	q8            metalmem.Slice[byte]
	outputNorm    metalmem.Slice[float32]
	logits        metalmem.Slice[float32]
	layers        [ds4Layers]metalLayerState
}

func newMetalRuntime(path string, context int) (_ *metalRuntime, err error) {
	if context <= 1 {
		return nil, errors.New("Metal runtime: context must exceed one token")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: open model: %w", err)
	}
	r := &metalRuntime{file: f, ctx: context}
	defer func() {
		if err != nil {
			_ = r.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: stat model: %w", err)
	}
	r.model, err = readGGUF(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: read model: %w", err)
	}
	if err := validatePinnedProfile(r.model); err != nil {
		return nil, fmt.Errorf("Metal runtime: model profile: %w", err)
	}
	var maxTensor uint64
	for _, tensor := range r.model.tensors {
		n, err := tensorBytes(tensor)
		if err != nil {
			return nil, err
		}
		maxTensor = max(maxTensor, n)
	}
	r.dev, err = libmetal.NewDevice()
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: create device: %w", err)
	}
	r.views, err = mapMetalModel(r.dev, f, uint64(info.Size()), maxTensor)
	if err != nil {
		return nil, err
	}
	r.ops, err = newMetalPinnedOps(r.dev)
	if err != nil {
		return nil, err
	}
	if err := r.bindModel(); err != nil {
		return nil, err
	}
	r.arena = metalmem.NewArena(r.dev)
	r.allocState()
	if err := r.arena.Err(); err != nil {
		return nil, err
	}
	if err := r.Reset(); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *metalRuntime) bindModel() error {
	bind := func(name string) (boundMetalTensor, error) {
		t, err := r.views.Bind(r.model, name)
		if err != nil {
			return boundMetalTensor{}, fmt.Errorf("Metal runtime: %w", err)
		}
		return t, nil
	}
	var err error
	if r.output.embedding, err = bind("token_embd.weight"); err != nil {
		return err
	}
	for i := range r.layers {
		if err := r.bindLayer(i, bind); err != nil {
			return err
		}
	}
	for dst, name := range map[*boundMetalTensor]string{
		&r.output.hcBase:  "output_hc_base.weight",
		&r.output.hcFn:    "output_hc_fn.weight",
		&r.output.hcScale: "output_hc_scale.weight",
		&r.output.norm:    "output_norm.weight",
		&r.output.weight:  "output.weight",
	} {
		if *dst, err = bind(name); err != nil {
			return err
		}
	}
	return nil
}

func (r *metalRuntime) bindLayer(i int, bind func(string) (boundMetalTensor, error)) error {
	l := &r.layers[i]
	prefix := fmt.Sprintf("blk.%d.", i)
	required := map[*boundMetalTensor]string{
		&l.attnSinks:   "attn_sinks.weight",
		&l.attnKV:      "attn_kv.weight",
		&l.attnKVNorm:  "attn_kv_a_norm.weight",
		&l.attnNorm:    "attn_norm.weight",
		&l.attnOutA:    "attn_output_a.weight",
		&l.attnOutB:    "attn_output_b.weight",
		&l.attnQA:      "attn_q_a.weight",
		&l.attnQANorm:  "attn_q_a_norm.weight",
		&l.attnQB:      "attn_q_b.weight",
		&l.expertGate:  "ffn_gate_exps.weight",
		&l.expertUp:    "ffn_up_exps.weight",
		&l.expertDown:  "ffn_down_exps.weight",
		&l.router:      "ffn_gate_inp.weight",
		&l.sharedGate:  "ffn_gate_shexp.weight",
		&l.sharedUp:    "ffn_up_shexp.weight",
		&l.sharedDown:  "ffn_down_shexp.weight",
		&l.ffnNorm:     "ffn_norm.weight",
		&l.hcAttnBase:  "hc_attn_base.weight",
		&l.hcAttnFn:    "hc_attn_fn.weight",
		&l.hcAttnScale: "hc_attn_scale.weight",
		&l.hcFFNBase:   "hc_ffn_base.weight",
		&l.hcFFNFn:     "hc_ffn_fn.weight",
		&l.hcFFNScale:  "hc_ffn_scale.weight",
	}
	if i < 3 {
		required[&l.expertMap] = "ffn_gate_tid2eid.weight"
	} else {
		required[&l.expertBias] = "exp_probs_b.bias"
	}
	if i >= 2 {
		required[&l.attnCompressorAPE] = "attn_compressor_ape.weight"
		required[&l.attnCompressorNorm] = "attn_compressor_norm.weight"
		required[&l.attnCompressorGate] = "attn_compressor_gate.weight"
		required[&l.attnCompressorKV] = "attn_compressor_kv.weight"
		if i%2 == 0 {
			required[&l.indexerCompressorAPE] = "indexer_compressor_ape.weight"
			required[&l.indexerCompressorNorm] = "indexer_compressor_norm.weight"
			required[&l.indexerCompressorGate] = "indexer_compressor_gate.weight"
			required[&l.indexerCompressorKV] = "indexer_compressor_kv.weight"
			required[&l.indexerProj] = "indexer.proj.weight"
			required[&l.indexerQB] = "indexer.attn_q_b.weight"
		}
	}
	for dst, suffix := range required {
		t, err := bind(prefix + suffix)
		if err != nil {
			return err
		}
		*dst = t
	}
	return nil
}

func (r *metalRuntime) allocState() {
	a := r.arena
	s := &r.state
	rawCap := min(r.ctx, ds4RawCap)
	s.hc[0] = metalmem.AllocIn[float32](a, "hc-a", ds4HCWidth)
	s.hc[1] = metalmem.AllocIn[float32](a, "hc-b", ds4HCWidth)
	s.attnPre = metalmem.AllocIn[float32](a, "attention-pre", 2*ds4Width)
	s.attnCur, _ = s.attnPre.Slice(0, ds4Width)
	s.attnNorm, _ = s.attnPre.Slice(ds4Width, 2*ds4Width)
	s.qLora = metalmem.AllocIn[float32](a, "query-lora", 1024)
	s.qHeads = metalmem.AllocIn[float32](a, "query-heads", 64*ds4KVWidth)
	s.kvRaw = metalmem.AllocIn[float32](a, "kv-raw", ds4KVWidth)
	s.kvCur = metalmem.AllocIn[float32](a, "kv-current", ds4KVWidth)
	s.heads = metalmem.AllocIn[float32](a, "attention-heads", 64*ds4KVWidth)
	s.rawScores = metalmem.AllocIn[float32](a, "raw-attention-scores", 64*rawCap)
	s.rawOutput = metalmem.AllocIn[float32](a, "raw-attention-output", 32768+512+32768)
	s.attnLow = metalmem.AllocIn[float32](a, "attention-low", 8192)
	s.attnOut = metalmem.AllocIn[float32](a, "attention-output", ds4Width)
	s.ffnPre = metalmem.AllocIn[float32](a, "ffn-pre", 2*ds4Width)
	s.ffnCur, _ = s.ffnPre.Slice(0, ds4Width)
	s.ffnNorm, _ = s.ffnPre.Slice(ds4Width, 2*ds4Width)
	s.router = metalmem.AllocIn[float32](a, "router", 256)
	s.probs = metalmem.AllocIn[float32](a, "router-probabilities", 256)
	s.expertIDs = metalmem.AllocIn[int32](a, "expert-ids", 6)
	s.expertW = metalmem.AllocIn[float32](a, "expert-weights", 6)
	s.sharedGate = metalmem.AllocIn[float32](a, "shared-gate", 2048)
	s.sharedUp = metalmem.AllocIn[float32](a, "shared-up", 2048)
	s.sharedMid = metalmem.AllocIn[float32](a, "shared-middle", 2048)
	s.ffnBlocks = metalmem.AllocIn[float32](a, "ffn-blocks", 2*ds4Width)
	s.routedOut, _ = s.ffnBlocks.Slice(0, ds4Width)
	s.sharedOut, _ = s.ffnBlocks.Slice(ds4Width, 2*ds4Width)
	s.expertGateRaw = metalmem.AllocIn[float32](a, "expert-gate-raw", 6*2048)
	s.expertUpRaw = metalmem.AllocIn[float32](a, "expert-up-raw", 6*2048)
	s.expertGate = metalmem.AllocIn[float32](a, "expert-gate", 6*2048)
	s.expertUp = metalmem.AllocIn[float32](a, "expert-up", 6*2048)
	s.expertMid = metalmem.AllocIn[float32](a, "expert-middle", 6*2048)
	s.expertDown = metalmem.AllocIn[float32](a, "expert-down", 6*ds4Width)
	s.ffnOut = metalmem.AllocIn[float32](a, "ffn-output", ds4Width)
	s.q8 = metalmem.AllocIn[byte](a, "q8-scratch", ds4Width/256*292)
	s.outputNorm = metalmem.AllocIn[float32](a, "output-norm", ds4Width)
	s.logits = metalmem.AllocIn[float32](a, "logits", 129280)
	for i := range s.layers {
		s.layers[i].rawKV = metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-raw-kv", i), rawCap*ds4KVWidth)
		if i < 2 {
			continue
		}
		ratio, width := 4, 1024
		if i%2 != 0 {
			ratio, width = 128, 512
		}
		cap := r.ctx/ratio + 2
		rows := ratio
		if ratio == 4 {
			rows = 2 * ratio
		}
		s.layers[i].attn = metalCompressedState{
			cache:      metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-attn-cache", i), cap*ds4KVWidth),
			stateKV:    metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-attn-state-kv", i), rows*width),
			stateScore: metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-attn-state-score", i), rows*width),
			pooled:     metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-attn-pooled", i), ds4KVWidth),
			headDim:    ds4KVWidth,
			width:      width,
			ratio:      ratio,
			cap:        cap,
		}
		if ratio == 4 {
			s.layers[i].indexer = metalCompressedState{
				cache:      metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-indexer-cache", i), cap*128),
				stateKV:    metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-indexer-state-kv", i), 2*ratio*256),
				stateScore: metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-indexer-state-score", i), 2*ratio*256),
				pooled:     metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-indexer-pooled", i), 128),
				headDim:    128,
				width:      256,
				ratio:      ratio,
				cap:        cap,
			}
		}
	}
}

func (r *metalRuntime) Reset() error {
	if r == nil || r.arena == nil {
		return errors.New("Metal runtime: closed")
	}
	r.pos = 0
	clear(r.state.hc[0].HostSlice())
	clear(r.state.hc[1].HostSlice())
	clear(r.state.logits.HostSlice())
	for i := range r.state.layers {
		clear(r.state.layers[i].rawKV.HostSlice())
		resetCompressedState(&r.state.layers[i].attn)
		resetCompressedState(&r.state.layers[i].indexer)
	}
	return nil
}

func resetCompressedState(s *metalCompressedState) {
	if s.cap == 0 {
		return
	}
	s.n = 0
	clear(s.cache.HostSlice())
	clear(s.stateKV.HostSlice())
	clear(s.pooled.HostSlice())
	scores := s.stateScore.HostSlice()
	for i := range scores {
		scores[i] = float32(math.Inf(-1))
	}
}

func (r *metalRuntime) evalToken(token int32, position int) ([]float32, error) {
	if position != r.pos || position < 0 || position >= r.ctx {
		return nil, fmt.Errorf("Metal runtime: position %d does not follow %d", position, r.pos)
	}
	if token < 0 || uint64(token) >= r.output.embedding.dimensions[1] {
		return nil, fmt.Errorf("Metal runtime: token %d out of range", token)
	}
	embedding, err := r.ops.GetRowF16(r.output.embedding, uint32(token))
	if err != nil {
		return nil, err
	}
	hc := make([]float32, ds4HCWidth)
	for stream := range 4 {
		copy(hc[stream*ds4Width:], embedding)
	}
	for i := range r.layers {
		hc, err = r.evalLayer(hc, i, token, position)
		if err != nil {
			return nil, fmt.Errorf("Metal runtime: layer %d: %w", i, err)
		}
	}
	logits, err := r.outputLogits(hc)
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: output: %w", err)
	}
	r.pos++
	copy(r.state.logits.HostSlice(), logits)
	return logits, nil
}

func (r *metalRuntime) Eval(token, position int) error {
	_, err := r.evalToken(int32(token), position)
	return err
}

func (r *metalRuntime) Logits() []float32 {
	if r == nil || r.arena == nil || r.pos == 0 {
		return nil
	}
	return r.state.logits.HostSlice()
}

func (r *metalRuntime) evalLayer(hc []float32, i int, token int32, position int) ([]float32, error) {
	layer := &r.layers[i]
	stateIn := r.state.hc[i%len(r.state.hc)]
	if err := stateIn.CopyFromHost(hc); err != nil {
		return nil, fmt.Errorf("attention opening: copy state: %w", err)
	}
	batch, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	if err := batch.EncodeHCPre(stateIn, layer.hcAttnFn, layer.hcAttnScale, layer.hcAttnBase, layer.attnNorm, r.state.attnPre); err != nil {
		return nil, err
	}
	if err := batch.EncodeMatvecQ80(layer.attnQA, r.state.attnNorm, r.state.qLora); err != nil {
		return nil, err
	}
	if err := batch.EncodeRMS(r.state.qLora, layer.attnQANorm, r.state.qLora, 1, true); err != nil {
		return nil, err
	}
	if err := batch.EncodeMatvecQ80(layer.attnQB, r.state.qLora, r.state.qHeads); err != nil {
		return nil, err
	}
	if err := batch.EncodeRMS(r.state.qHeads, layer.attnQANorm, r.state.qHeads, 64, false); err != nil {
		return nil, err
	}
	if err := batch.EncodeMatvecQ80(layer.attnKV, r.state.attnNorm, r.state.kvRaw); err != nil {
		return nil, err
	}
	if err := batch.EncodeRMS(r.state.kvRaw, layer.attnKVNorm, r.state.kvCur, 1, true); err != nil {
		return nil, err
	}
	state := &r.state.layers[i]
	headsGPU := r.state.heads
	if i < 2 {
		if err := batch.EncodeRopeAttentionRaw(r.state.qHeads, r.state.kvCur, layer.attnSinks, state.rawKV, r.state.rawScores, r.state.rawOutput, position); err != nil {
			return nil, err
		}
		headsGPU, err = r.state.rawOutput.Slice(32768+512, 32768+512+32768)
		if err != nil {
			return nil, err
		}
	}
	if err := batch.Wait(context.Background()); err != nil {
		return nil, err
	}
	r.lastOpeningBatch = batch.Stats()
	if i < 2 {
		r.lastRawOpeningBatch = batch.Stats()
	}
	var heads []float32
	if i >= 2 {
		attnNorm := r.state.attnNorm.HostSlice()
		qNorm := r.state.qLora.HostSlice()
		q := r.state.qHeads.HostSlice()
		kv := r.state.kvCur.HostSlice()
		compKV, e := r.ops.MatvecF16(layer.attnCompressorKV, attnNorm)
		if e != nil {
			return nil, e
		}
		compScore, e := r.ops.MatvecF16(layer.attnCompressorGate, attnNorm)
		if e != nil {
			return nil, e
		}
		if _, _, e = r.ops.CompressorAttention(compKV, compScore, layer.attnCompressorAPE, layer.attnCompressorNorm, &state.attn, position); e != nil {
			return nil, e
		}
		var allowed []uint32
		if i%2 == 0 {
			indexKV, e := r.ops.MatvecF16(layer.indexerCompressorKV, attnNorm)
			if e != nil {
				return nil, e
			}
			indexScore, e := r.ops.MatvecF16(layer.indexerCompressorGate, attnNorm)
			if e != nil {
				return nil, e
			}
			if _, _, e = r.ops.CompressorIndexer(indexKV, indexScore, layer.indexerCompressorAPE, layer.indexerCompressorNorm, &state.indexer, position); e != nil {
				return nil, e
			}
			allowed, e = r.ops.IndexerAllowed(qNorm, attnNorm, layer.indexerQB, layer.indexerProj, state.indexer.cache, state.indexer.n, position, i)
			if e != nil {
				return nil, e
			}
		}
		_, _, heads, err = r.ops.RopeAttentionMixed(q, kv, layer.attnSinks, state.rawKV, state.attn.cache, state.attn.n, allowed, position, i)
	}
	if err != nil {
		return nil, err
	}
	if i >= 2 {
		if err := headsGPU.CopyFromHost(heads); err != nil {
			return nil, fmt.Errorf("attention closing: copy heads: %w", err)
		}
	}
	closing, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	for group := range 8 {
		rows, err := layer.attnOutA.Rows(group*1024, 1024)
		if err != nil {
			return nil, err
		}
		head, err := headsGPU.Slice(group*4096, (group+1)*4096)
		if err != nil {
			return nil, err
		}
		low, err := r.state.attnLow.Slice(group*1024, (group+1)*1024)
		if err != nil {
			return nil, err
		}
		if err := closing.EncodeMatvecQ80(rows, head, low); err != nil {
			return nil, err
		}
	}
	if err := closing.EncodeMatvecQ80(layer.attnOutB, r.state.attnLow, r.state.attnOut); err != nil {
		return nil, err
	}
	attentionState := r.state.hc[(i+1)%len(r.state.hc)]
	if err := closing.EncodeHCPost(stateIn, r.state.attnOut, layer.hcAttnFn, layer.hcAttnScale, layer.hcAttnBase, attentionState); err != nil {
		return nil, err
	}
	if err := closing.Wait(context.Background()); err != nil {
		return nil, err
	}
	r.lastAttentionBatch = closing.Stats()
	stateIn = attentionState
	r.lastFFNBatches = metalBatchStats{}
	ffnOpening, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	if err := ffnOpening.EncodeHCPre(stateIn, layer.hcFFNFn, layer.hcFFNScale, layer.hcFFNBase, layer.ffnNorm, r.state.ffnPre); err != nil {
		return nil, err
	}
	if err := ffnOpening.EncodeMatvecF16(layer.router, r.state.ffnNorm, r.state.router); err != nil {
		return nil, err
	}
	if err := ffnOpening.EncodeMatvecQ80(layer.sharedGate, r.state.ffnNorm, r.state.sharedGate); err != nil {
		return nil, err
	}
	if err := ffnOpening.EncodeMatvecQ80(layer.sharedUp, r.state.ffnNorm, r.state.sharedUp); err != nil {
		return nil, err
	}
	if err := ffnOpening.Wait(context.Background()); err != nil {
		return nil, err
	}
	r.addFFNBatch(ffnOpening)
	router := r.state.router.HostSlice()
	var selected []int32
	var weights []float32
	if i < 3 {
		_, selected, weights, err = r.ops.RouterHash(router, layer.expertMap, uint32(token))
	} else {
		_, selected, weights, err = r.ops.RouterBiased(router, layer.expertBias)
	}
	if err != nil {
		return nil, err
	}
	if len(selected) != r.state.expertIDs.Len() || len(weights) != r.state.expertW.Len() {
		return nil, fmt.Errorf("routed experts: got %d IDs and %d weights, want %d", len(selected), len(weights), r.state.expertIDs.Len())
	}
	if err := r.state.expertIDs.CopyFromHost(selected); err != nil {
		return nil, fmt.Errorf("routed experts: copy IDs: %w", err)
	}
	if err := r.state.expertW.CopyFromHost(weights); err != nil {
		return nil, fmt.Errorf("routed experts: copy weights: %w", err)
	}
	shared, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	if err := shared.EncodeSwiGLU(r.state.sharedGate, r.state.sharedUp, r.state.sharedMid); err != nil {
		return nil, err
	}
	if err := shared.EncodeMatvecQ80(layer.sharedDown, r.state.sharedMid, r.state.sharedOut); err != nil {
		return nil, err
	}
	if err := shared.Wait(context.Background()); err != nil {
		return nil, err
	}
	r.addFFNBatch(shared)
	routed, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	if err := routed.EncodeMatvecIQ2Selected(layer.expertGate, layer.expertUp, r.state.expertIDs, r.state.ffnNorm, r.state.expertGateRaw, r.state.expertUpRaw); err != nil {
		return nil, err
	}
	if err := routed.EncodeClampGate10(r.state.expertGateRaw, r.state.expertGate); err != nil {
		return nil, err
	}
	if err := routed.EncodeClamp10(r.state.expertUpRaw, r.state.expertUp); err != nil {
		return nil, err
	}
	for slot, weight := range weights {
		start := slot * 2048
		end := start + 2048
		gate, err := r.state.expertGate.Slice(start, end)
		if err != nil {
			return nil, err
		}
		up, err := r.state.expertUp.Slice(start, end)
		if err != nil {
			return nil, err
		}
		mid, err := r.state.expertMid.Slice(start, end)
		if err != nil {
			return nil, err
		}
		if err := routed.EncodeSwiGLUWeighted(gate, up, mid, weight); err != nil {
			return nil, err
		}
	}
	if err := routed.EncodeMatvecQ2KSelected(layer.expertDown, r.state.expertIDs, r.state.expertMid, r.state.expertDown); err != nil {
		return nil, err
	}
	if err := routed.EncodeSumVectors(r.state.expertDown, r.state.routedOut, 6); err != nil {
		return nil, err
	}
	if err := routed.Wait(context.Background()); err != nil {
		return nil, err
	}
	r.lastRoutedBatch = routed.Stats()
	r.addFFNBatch(routed)
	ffnClosing, err := newMetalBatch(r.ops)
	if err != nil {
		return nil, err
	}
	if err := ffnClosing.EncodeSumVectors(r.state.ffnBlocks, r.state.ffnOut, 2); err != nil {
		return nil, err
	}
	stateOut := r.state.hc[i%len(r.state.hc)]
	if err := ffnClosing.EncodeHCPost(stateIn, r.state.ffnOut, layer.hcFFNFn, layer.hcFFNScale, layer.hcFFNBase, stateOut); err != nil {
		return nil, err
	}
	if err := ffnClosing.Wait(context.Background()); err != nil {
		return nil, err
	}
	r.addFFNBatch(ffnClosing)
	return stateOut.HostSlice(), nil
}

func (r *metalRuntime) addFFNBatch(batch *metalBatch) {
	stats := batch.Stats()
	r.lastFFNBatches.Encoded += stats.Encoded
	r.lastFFNBatches.Commits += stats.Commits
	r.lastFFNBatches.Waits += stats.Waits
}

func (r *metalRuntime) outputLogits(hc []float32) ([]float32, error) {
	flat, err := r.ops.RMS(hc, r.output.norm, 1, false)
	if err != nil {
		return nil, err
	}
	pre, err := r.ops.MatvecF16(r.output.hcFn, flat)
	if err != nil {
		return nil, err
	}
	base := boundF32(r.output.hcBase)
	scale := boundF32(r.output.hcScale)
	if len(base) != 4 || len(scale) != 1 {
		return nil, errors.New("invalid output HC weights")
	}
	out := make([]float32, ds4Width)
	for stream := range 4 {
		weight := float32(1/(1+math.Exp(-float64(pre[stream]*scale[0]+base[stream]))) + 1e-6)
		for i, value := range hc[stream*ds4Width : (stream+1)*ds4Width] {
			out[i] += value * weight
		}
	}
	out, err = r.ops.RMS(out, r.output.norm, 1, true)
	if err != nil {
		return nil, err
	}
	return r.ops.MatvecQ80(r.output.weight, out)
}

func boundF32(t boundMetalTensor) []float32 {
	b := t.data.HostSlice()
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func (r *metalRuntime) Close() error {
	if r == nil {
		return nil
	}
	if r.arena != nil {
		r.arena.Free()
		r.arena = nil
	}
	if r.ops != nil {
		r.ops.Close()
		r.ops = nil
	}
	if r.views != nil {
		r.views.Close()
		r.views = nil
	}
	if r.file != nil {
		_ = r.file.Close()
		r.file = nil
	}
	if r.dev != nil {
		return r.dev.Close()
	}
	return nil
}
