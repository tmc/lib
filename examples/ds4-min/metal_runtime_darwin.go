//go:build darwin && arm64

package main

import (
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
// DS4 session. It deliberately does not implement sessionBackend yet: doing so
// before multi-token attention and cache updates exist would make token two
// silently incorrect.
type metalRuntime struct {
	file   *os.File
	dev    *libmetal.Device
	views  *metalModelViews
	ops    *metalPinnedOps
	arena  *metalmem.Arena
	model  *ggufFile
	layers [ds4Layers]metalLayer
	output metalOutput
	state  metalSessionState
	ctx    int
	pos    int
}

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
	kv    metalmem.Slice[float32]
	score metalmem.Slice[float32]
	width int
	cap   int
}

type metalLayerState struct {
	rawKV   metalmem.Slice[float32]
	attn    metalCompressedState
	indexer metalCompressedState
}

type metalSessionState struct {
	hc         [2]metalmem.Slice[float32]
	attnCur    metalmem.Slice[float32]
	attnNorm   metalmem.Slice[float32]
	qLora      metalmem.Slice[float32]
	qHeads     metalmem.Slice[float32]
	kvRaw      metalmem.Slice[float32]
	kvCur      metalmem.Slice[float32]
	heads      metalmem.Slice[float32]
	attnLow    metalmem.Slice[float32]
	attnOut    metalmem.Slice[float32]
	ffnNorm    metalmem.Slice[float32]
	router     metalmem.Slice[float32]
	probs      metalmem.Slice[float32]
	expertIDs  metalmem.Slice[int32]
	expertW    metalmem.Slice[float32]
	sharedGate metalmem.Slice[float32]
	sharedUp   metalmem.Slice[float32]
	sharedMid  metalmem.Slice[float32]
	sharedOut  metalmem.Slice[float32]
	expertGate metalmem.Slice[float32]
	expertUp   metalmem.Slice[float32]
	expertMid  metalmem.Slice[float32]
	expertDown metalmem.Slice[float32]
	q8         metalmem.Slice[byte]
	outputNorm metalmem.Slice[float32]
	logits     metalmem.Slice[float32]
	layers     [ds4Layers]metalLayerState
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
	s.hc[0] = metalmem.AllocIn[float32](a, "hc-a", ds4HCWidth)
	s.hc[1] = metalmem.AllocIn[float32](a, "hc-b", ds4HCWidth)
	s.attnCur = metalmem.AllocIn[float32](a, "attention-current", ds4Width)
	s.attnNorm = metalmem.AllocIn[float32](a, "attention-norm", ds4Width)
	s.qLora = metalmem.AllocIn[float32](a, "query-lora", 1024)
	s.qHeads = metalmem.AllocIn[float32](a, "query-heads", 64*ds4KVWidth)
	s.kvRaw = metalmem.AllocIn[float32](a, "kv-raw", ds4KVWidth)
	s.kvCur = metalmem.AllocIn[float32](a, "kv-current", ds4KVWidth)
	s.heads = metalmem.AllocIn[float32](a, "attention-heads", 64*ds4KVWidth)
	s.attnLow = metalmem.AllocIn[float32](a, "attention-low", 8192)
	s.attnOut = metalmem.AllocIn[float32](a, "attention-output", ds4Width)
	s.ffnNorm = metalmem.AllocIn[float32](a, "ffn-norm", ds4Width)
	s.router = metalmem.AllocIn[float32](a, "router", 256)
	s.probs = metalmem.AllocIn[float32](a, "router-probabilities", 256)
	s.expertIDs = metalmem.AllocIn[int32](a, "expert-ids", 6)
	s.expertW = metalmem.AllocIn[float32](a, "expert-weights", 6)
	s.sharedGate = metalmem.AllocIn[float32](a, "shared-gate", 2048)
	s.sharedUp = metalmem.AllocIn[float32](a, "shared-up", 2048)
	s.sharedMid = metalmem.AllocIn[float32](a, "shared-middle", 2048)
	s.sharedOut = metalmem.AllocIn[float32](a, "shared-output", ds4Width)
	s.expertGate = metalmem.AllocIn[float32](a, "expert-gate", 6*2048)
	s.expertUp = metalmem.AllocIn[float32](a, "expert-up", 6*2048)
	s.expertMid = metalmem.AllocIn[float32](a, "expert-middle", 6*2048)
	s.expertDown = metalmem.AllocIn[float32](a, "expert-down", 6*ds4Width)
	s.q8 = metalmem.AllocIn[byte](a, "q8-scratch", ds4Width/256*292)
	s.outputNorm = metalmem.AllocIn[float32](a, "output-norm", ds4Width)
	s.logits = metalmem.AllocIn[float32](a, "logits", 129280)
	rawCap := min(r.ctx, ds4RawCap)
	for i := range s.layers {
		s.layers[i].rawKV = metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-raw-kv", i), rawCap*ds4KVWidth)
		if i < 2 {
			continue
		}
		ratio, width := 4, 1024
		if i%2 != 0 {
			ratio, width = 128, 512
		}
		cap := (r.ctx + ratio - 1) / ratio
		s.layers[i].attn = metalCompressedState{
			kv:    metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-attn-kv", i), cap*width),
			score: metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-attn-score", i), cap*width),
			width: width,
			cap:   cap,
		}
		if ratio == 4 {
			s.layers[i].indexer = metalCompressedState{
				kv:    metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-indexer-kv", i), cap*256),
				score: metalmem.AllocIn[float32](a, fmt.Sprintf("layer-%d-indexer-score", i), cap*256),
				width: 256,
				cap:   cap,
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
	clear(s.kv.HostSlice())
	for i := range s.score.HostSlice() {
		s.score.HostSlice()[i] = float32(math.Inf(-1))
	}
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
