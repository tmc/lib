//go:build darwin && arm64

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/gguf"
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
	file      *os.File
	dev       *libmetal.Device
	views     *metalModelViews
	ops       *metalPinnedOps
	arena     *metalmem.Arena
	model     *gguf.File
	layers    [ds4Layers]metalLayer
	output    metalOutput
	state     metalSessionState
	ctx       int
	pos       int
	lastBatch metalBatchStats
}

var (
	_ sessionBackend = (*metalRuntime)(nil)
	_ batchEvaluator = (*metalRuntime)(nil)
)

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
	hc             [2]metalmem.Slice[float32]
	attnPre        metalmem.Slice[float32]
	attnCur        metalmem.Slice[float32]
	attnNorm       metalmem.Slice[float32]
	qLora          metalmem.Slice[float32]
	qHeads         metalmem.Slice[float32]
	kvRaw          metalmem.Slice[float32]
	kvCur          metalmem.Slice[float32]
	compressorKV   metalmem.Slice[float32]
	compressorGate metalmem.Slice[float32]
	indexerKV      metalmem.Slice[float32]
	indexerGate    metalmem.Slice[float32]
	heads          metalmem.Slice[float32]
	rawScores      metalmem.Slice[float32]
	rawOutput      metalmem.Slice[float32]
	mixedScores    metalmem.Slice[float32]
	mixedOutput    metalmem.Slice[float32]
	allowed        metalmem.Slice[uint32]
	attnLow        metalmem.Slice[float32]
	attnOut        metalmem.Slice[float32]
	ffnPre         metalmem.Slice[float32]
	ffnCur         metalmem.Slice[float32]
	ffnNorm        metalmem.Slice[float32]
	router         metalmem.Slice[float32]
	probs          metalmem.Slice[float32]
	expertIDs      metalmem.Slice[int32]
	expertW        metalmem.Slice[float32]
	sharedGate     metalmem.Slice[float32]
	sharedUp       metalmem.Slice[float32]
	sharedMid      metalmem.Slice[float32]
	sharedOut      metalmem.Slice[float32]
	expertGate     metalmem.Slice[float32]
	expertUp       metalmem.Slice[float32]
	expertGateRaw  metalmem.Slice[float32]
	expertUpRaw    metalmem.Slice[float32]
	expertMid      metalmem.Slice[float32]
	expertDown     metalmem.Slice[float32]
	routedOut      metalmem.Slice[float32]
	ffnBlocks      metalmem.Slice[float32]
	ffnOut         metalmem.Slice[float32]
	q8             metalmem.Slice[byte]
	outputNorm     metalmem.Slice[float32]
	logits         metalmem.Slice[float32]
	layers         [ds4Layers]metalLayerState
}

// newMetalRuntime opens the model quietly. Callers that want the ds4-style
// initialization diagnostics use newMetalRuntimeLogged.
func newMetalRuntime(path string, context int) (*metalRuntime, error) {
	return newMetalRuntimeLogged(path, context, nil)
}

// newMetalRuntimeLogged opens the model and, when log is non-nil, reports the
// same initialization lines ds4_cli.c prints: device, view creation, warmup and
// the mapped span. See ds4_metal.m ds4_gpu_warm_model_views for the warmup,
// which exists so the first-dispatch page-table validation of the mmapped span
// is paid at load time instead of inside the prefill timing.
func newMetalRuntimeLogged(path string, context int, log io.Writer) (_ *metalRuntime, err error) {
	logf := func(format string, args ...any) {
		if log != nil {
			fmt.Fprintf(log, "ds4-min: "+format+"\n", args...)
		}
	}
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
	r.model, err = gguf.Read(f, info.Size())
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: read model: %w", err)
	}
	if err := validatePinnedProfile(r.model); err != nil {
		return nil, fmt.Errorf("Metal runtime: model profile: %w", err)
	}
	var maxTensor uint64
	for _, tensor := range r.model.Tensors {
		n, err := tensor.Bytes()
		if err != nil {
			return nil, err
		}
		maxTensor = max(maxTensor, n)
	}
	r.dev, err = libmetal.NewDevice()
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: create device: %w", err)
	}
	logf("Metal device %s, %.2f GiB recommended working set", r.dev.Raw().Name(),
		float64(r.dev.Raw().RecommendedMaxWorkingSetSize())/(1<<30))
	fileBytes := uint64(info.Size())
	if r.model.DataOffset >= fileBytes {
		return nil, fmt.Errorf("Metal runtime: GGUF data offset %d outside %d-byte model", r.model.DataOffset, fileBytes)
	}
	modelBytes := fileBytes - r.model.DataOffset
	viewStart := time.Now()
	r.views, err = mapMetalModelRange(r.dev, f, r.model.DataOffset, modelBytes, maxTensor)
	if err != nil {
		return nil, err
	}
	viewMS := float64(time.Since(viewStart)) / float64(time.Millisecond)
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
	warmStart := time.Now()
	warmed, err := r.warmViews()
	if err != nil {
		return nil, err
	}
	warmMS := float64(time.Since(warmStart)) / float64(time.Millisecond)
	logf("Metal model views created in %.3f ms, warmup %.3f ms (mapped %.2f MiB from offset %.2f MiB)",
		viewMS, warmMS, float64(modelBytes)/(1<<20), float64(r.model.DataOffset)/(1<<20))
	logf("Metal mapped mmaped model as %d overlapping shared buffers, %d warmed", len(r.views.views), warmed)
	return r, nil
}

// warmViews touches one tensor in each mapped view so the GPU page-table
// validation of the whole mmapped span is paid here rather than on the first
// prefill dispatch. It returns the number of views it reached; a view holding
// no F16 tensor is left cold, since GetRowF16 is the only op that can read an
// arbitrary model tensor without a surrounding graph.
func (r *metalRuntime) warmViews() (int, error) {
	warmed := 0
	for _, view := range r.views.views {
		end := view.fileOffset + uint64(view.data.Len())
		for _, tensor := range r.model.Tensors {
			if tensor.TypeID != 1 || len(tensor.Dimensions) != 2 {
				continue
			}
			n, err := tensor.Bytes()
			if err != nil {
				return warmed, err
			}
			off := r.model.DataOffset + tensor.Offset
			if off < view.fileOffset || off+n > end {
				continue
			}
			t, err := r.views.Bind(r.model, tensor.Name)
			if err != nil {
				return warmed, err
			}
			if _, err := r.ops.GetRowF16(t, 0); err != nil {
				return warmed, err
			}
			warmed++
			break
		}
	}
	return warmed, nil
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
	s.compressorKV = metalmem.AllocIn[float32](a, "compressor-kv", 1024)
	s.compressorGate = metalmem.AllocIn[float32](a, "compressor-gate", 1024)
	s.indexerKV = metalmem.AllocIn[float32](a, "indexer-kv", 256)
	s.indexerGate = metalmem.AllocIn[float32](a, "indexer-gate", 256)
	s.heads = metalmem.AllocIn[float32](a, "attention-heads", 64*ds4KVWidth)
	s.rawScores = metalmem.AllocIn[float32](a, "raw-attention-scores", 64*rawCap)
	s.rawOutput = metalmem.AllocIn[float32](a, "raw-attention-output", 32768+512+32768)
	maxComp := r.ctx/4 + 2
	s.mixedScores = metalmem.AllocIn[float32](a, "mixed-attention-scores", 64*(rawCap+maxComp))
	s.mixedOutput = metalmem.AllocIn[float32](a, "mixed-attention-output", 32768+512+32768)
	s.allowed = metalmem.AllocIn[uint32](a, "mixed-attention-allowed", max(1, maxComp))
	for i := range s.allowed.HostSlice() {
		s.allowed.HostSlice()[i] = 1
	}
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
	// routedOut and sharedOut are the two halves of one allocation because the
	// sum at the end of the FFN reduces across them: adjacency is what makes
	// EncodeSumVectors(ffnBlocks, ffnOut, 2) the right call, not a convenience.
	//
	// Widening this to N tokens: the layout must stay [token][routed|shared],
	// so that each token's two rows remain adjacent. EncodeSumVectorsTokens
	// handles the consumer correctly at that layout -- it takes the width from
	// out.Len()/tokens and gives each token its own dispatch over its own two
	// rows. What does not survive is these two stored sub-slices. tokenRow is
	// contiguous (s.Slice(token*width, (token+1)*width)), so a token-major
	// routedOut of N*4096 would place token t at 4096*t when it belongs at
	// 8192*t. The producers, not the consumer, are the blocker: the expert sum
	// writing routedOut and the shared projection writing sharedOut each need a
	// per-token window computed from ffnBlocks directly, and these two fields
	// should go away rather than be widened.
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
	return r.evalPosition(token, position, true)
}

// evalPosition advances the session by one token. When logits is false the
// output projection is skipped: it reads the whole output weight tensor, the
// single largest weight read of a step, and its result is discarded for every
// position of a prefill run but the last. Nothing else differs, so the state
// this leaves behind is bit-identical to the logits path.
func (r *metalRuntime) evalPosition(token int32, position int, logits bool) ([]float32, error) {
	if position != r.pos || position < 0 || position >= r.ctx {
		return nil, fmt.Errorf("Metal runtime: position %d does not follow %d", position, r.pos)
	}
	if token < 0 || uint64(token) >= r.output.embedding.dimensions[1] {
		return nil, fmt.Errorf("Metal runtime: token %d out of range", token)
	}
	defer begin(levelToken, "token", nil).end()
	embedding, err := r.ops.GetRowF16(r.output.embedding, uint32(token))
	if err != nil {
		return nil, err
	}
	dst := r.state.hc[0].HostSlice()
	if len(dst) < ds4HCWidth {
		return nil, errors.New("Metal runtime: invalid residual buffer")
	}
	for stream := range 4 {
		copy(dst[stream*ds4Width:], embedding)
	}
	var batch *metalBatch
	for i := range r.layers {
		if batch == nil {
			batch, err = newMetalBatch(r.ops)
			if err != nil {
				return nil, err
			}
			batch.inRegion("layer", 0, position)
		}
		batch, err = r.evalLayer(batch, i, token, position)
		if err != nil {
			return nil, fmt.Errorf("Metal runtime: layer %d: %w", i, err)
		}
	}
	if batch != nil {
		if err := batch.Commit(); err != nil {
			return nil, err
		}
		if err := batch.Wait(context.Background()); err != nil {
			return nil, err
		}
		r.lastBatch = batch.Stats()
	}
	if !logits {
		r.pos++
		return nil, nil
	}
	out, err := r.outputLogits(r.state.hc[0].HostSlice())
	if err != nil {
		return nil, fmt.Errorf("Metal runtime: output: %w", err)
	}
	r.pos++
	copy(r.state.logits.HostSlice(), out)
	return out, nil
}

func (r *metalRuntime) Eval(token, position int) error {
	_, err := r.evalToken(int32(token), position)
	return err
}

// EvalBatch evaluates a prefill run. Every position runs the same layer
// pipeline in the same order as Eval; only the discarded output projections of
// the run's earlier positions are skipped, so the final logits are identical
// to those of the single-token loop.
func (r *metalRuntime) EvalBatch(tokens []int, position int) error {
	for i, token := range tokens {
		if _, err := r.evalPosition(int32(token), position+i, i == len(tokens)-1); err != nil {
			return err
		}
	}
	return nil
}

func (r *metalRuntime) Logits() []float32 {
	if r == nil || r.arena == nil || r.pos == 0 {
		return nil
	}
	return r.state.logits.HostSlice()
}

func (r *metalRuntime) evalLayer(batch *metalBatch, i int, token int32, position int) (*metalBatch, error) {
	defer beginLayer(i).end()
	layer := &r.layers[i]
	stateIn := r.state.hc[0]
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
	indexerWillEmit := i >= 2 && i%2 == 0 && (position+1)%state.indexer.ratio == 0
	batchedCompressed := i >= 2 && (i%2 != 0 || state.indexer.n < 512 || state.indexer.n == 512 && !indexerWillEmit)
	attnEmit, indexerEmit := false, false
	if i < 2 {
		if err := batch.EncodeRopeAttentionRaw(r.state.qHeads, r.state.kvCur, layer.attnSinks, state.rawKV, r.state.rawScores, r.state.rawOutput, position); err != nil {
			return nil, err
		}
		var err error
		headsGPU, err = r.state.rawOutput.Slice(32768+512, 32768+512+32768)
		if err != nil {
			return nil, err
		}
	} else if batchedCompressed {
		compKV, e := r.state.compressorKV.Slice(0, state.attn.width)
		if e != nil {
			return nil, e
		}
		compGate, e := r.state.compressorGate.Slice(0, state.attn.width)
		if e != nil {
			return nil, e
		}
		if err := batch.EncodeMatvecF16(layer.attnCompressorKV, r.state.attnNorm, compKV); err != nil {
			return nil, err
		}
		if err := batch.EncodeMatvecF16(layer.attnCompressorGate, r.state.attnNorm, compGate); err != nil {
			return nil, err
		}
		var err error
		attnEmit, err = batch.EncodeCompressor(compKV, compGate, layer.attnCompressorAPE, layer.attnCompressorNorm, &state.attn, position, false)
		if err != nil {
			return nil, err
		}
		if i%2 == 0 {
			if err := batch.EncodeMatvecF16(layer.indexerCompressorKV, r.state.attnNorm, r.state.indexerKV); err != nil {
				return nil, err
			}
			if err := batch.EncodeMatvecF16(layer.indexerCompressorGate, r.state.attnNorm, r.state.indexerGate); err != nil {
				return nil, err
			}
			indexerEmit, err = batch.EncodeCompressor(r.state.indexerKV, r.state.indexerGate, layer.indexerCompressorAPE, layer.indexerCompressorNorm, &state.indexer, position, true)
			if err != nil {
				return nil, err
			}
		}
		nComp := state.attn.n
		if attnEmit {
			nComp++
		}
		if err := batch.EncodeRopeAttentionMixed(r.state.qHeads, r.state.kvCur, layer.attnSinks, state.rawKV, state.attn.cache, nComp, r.state.allowed, r.state.mixedScores, r.state.mixedOutput, position, i); err != nil {
			return nil, err
		}
		headsGPU, err = r.state.mixedOutput.Slice(32768+512, 32768+512+32768)
		if err != nil {
			return nil, err
		}
	}
	if attnEmit {
		state.attn.n++
	}
	if indexerEmit {
		state.indexer.n++
	}
	closing := batch
	if i >= 2 && !batchedCompressed {
		if err := batch.Commit(); err != nil {
			return nil, err
		}
		if err := batch.Wait(context.Background()); err != nil {
			return nil, err
		}
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
		var heads []float32
		var err error
		_, _, heads, err = r.ops.RopeAttentionMixed(q, kv, layer.attnSinks, state.rawKV, state.attn.cache, state.attn.n, allowed, position, i)
		if err != nil {
			return nil, err
		}
		if err := headsGPU.CopyFromHost(heads); err != nil {
			return nil, fmt.Errorf("attention closing: copy heads: %w", err)
		}
		closing, err = newMetalBatch(r.ops)
		if err != nil {
			return nil, err
		}
		closing.inRegion("closing-ffn", i, position)
	}
	// attn_output_a is block diagonal: group g reads heads[g*4096:(g+1)*4096]
	// and writes attnLow[g*1024:(g+1)*1024], touching only its own 1024 rows.
	// One dispatch over the whole tensor produces the same floats as the eight
	// it replaces, bit for bit, because each output row still sums the same
	// terms in the same order -- see TestMetalMatvecQ80BlockDiagMatchesLoop.
	if err := closing.EncodeMatvecQ80BlockDiag(layer.attnOutA, headsGPU, r.state.attnLow, 8); err != nil {
		return nil, err
	}
	if err := closing.EncodeMatvecQ80(layer.attnOutB, r.state.attnLow, r.state.attnOut); err != nil {
		return nil, err
	}
	attentionState := r.state.hc[1]
	// The old command boundary made this post recompute its split rather than
	// reuse HCPre's. Keep that arithmetic when both share a buffer.
	if closing == batch {
		r.ops.hc.pending = nil
	}
	if err := closing.EncodeHCPost(stateIn, r.state.attnOut, layer.hcAttnFn, layer.hcAttnScale, layer.hcAttnBase, attentionState); err != nil {
		return nil, err
	}
	// The FFN opening reads only GPU state the attention closing writes,
	// so both ride one command buffer with GPU routing.
	stateIn = attentionState
	if err := closing.EncodeHCPre(stateIn, layer.hcFFNFn, layer.hcFFNScale, layer.hcFFNBase, layer.ffnNorm, r.state.ffnPre); err != nil {
		return nil, err
	}
	if err := closing.EncodeMatvecF16(layer.router, r.state.ffnNorm, r.state.router); err != nil {
		return nil, err
	}
	if err := closing.EncodeMatvecQ80(layer.sharedGate, r.state.ffnNorm, r.state.sharedGate); err != nil {
		return nil, err
	}
	if err := closing.EncodeMatvecQ80(layer.sharedUp, r.state.ffnNorm, r.state.sharedUp); err != nil {
		return nil, err
	}
	var err error
	if i < 3 {
		err = closing.EncodeRouterHash(r.state.router, layer.expertMap, uint32(token), r.state.probs, r.state.expertIDs, r.state.expertW)
	} else {
		err = closing.EncodeRouterBiased(r.state.router, layer.expertBias, r.state.probs, r.state.expertIDs, r.state.expertW)
	}
	if err != nil {
		return nil, err
	}
	// The shared expert, the routed experts and the FFN closing depend only on
	// GPU state the attention closing already writes, and the routed weights
	// now stay on the device, so one command buffer carries the whole layer.
	ffn := closing
	if err := ffn.EncodeSwiGLU(r.state.sharedGate, r.state.sharedUp, r.state.sharedMid); err != nil {
		return nil, err
	}
	if err := ffn.EncodeMatvecQ80(layer.sharedDown, r.state.sharedMid, r.state.sharedOut); err != nil {
		return nil, err
	}
	if err := ffn.EncodeMatvecIQ2Selected(layer.expertGate, layer.expertUp, r.state.expertIDs, r.state.ffnNorm, r.state.expertGateRaw, r.state.expertUpRaw); err != nil {
		return nil, err
	}
	if err := ffn.EncodeClampGate10(r.state.expertGateRaw, r.state.expertGate); err != nil {
		return nil, err
	}
	if err := ffn.EncodeClamp10(r.state.expertUpRaw, r.state.expertUp); err != nil {
		return nil, err
	}
	// The routed weights stay on the device: binding one from the router's
	// output buffer gives the kernel the same uniform an encode-time scalar
	// does, so no host readback splits the layer. Reading them back is what
	// used to force a command boundary here.
	for slot := range 6 {
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
		weight, err := r.state.expertW.Slice(slot, slot+1)
		if err != nil {
			return nil, err
		}
		if err := ffn.EncodeSwiGLUWeightedGPU(gate, up, mid, weight); err != nil {
			return nil, err
		}
	}
	if err := ffn.EncodeMatvecQ2KSelected(layer.expertDown, r.state.expertIDs, r.state.expertMid, r.state.expertDown); err != nil {
		return nil, err
	}
	if err := ffn.EncodeSumVectors(r.state.expertDown, r.state.routedOut, 6); err != nil {
		return nil, err
	}
	if err := ffn.EncodeSumVectors(r.state.ffnBlocks, r.state.ffnOut, 2); err != nil {
		return nil, err
	}
	stateOut := r.state.hc[0]
	// The old command boundary made this post recompute its split rather than
	// reuse HCPre's. Clearing the cache owner keeps that arithmetic exactly,
	// now that the two share a buffer.
	r.ops.hc.pending = nil
	if err := ffn.EncodeHCPost(stateIn, r.state.ffnOut, layer.hcFFNFn, layer.hcFFNScale, layer.hcFFNBase, stateOut); err != nil {
		return nil, err
	}
	if closing != batch {
		if err := ffn.Commit(); err != nil {
			return nil, err
		}
		if err := ffn.Wait(context.Background()); err != nil {
			return nil, err
		}
		r.lastBatch = ffn.Stats()
		return nil, nil
	}
	return batch, nil
}

func (r *metalRuntime) outputLogits(hc []float32) ([]float32, error) {
	defer beginOp("output.logits").end()
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
	var err error
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
		err = errors.Join(err, r.file.Close())
		r.file = nil
	}
	if r.dev != nil {
		err = errors.Join(err, r.dev.Close())
		r.dev = nil
	}
	return err
}
