package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// The tiny stack is the multi-layer companion to the single-layer tiny model in
// cpu_model.go. It keeps the same operators and the same float32 reference
// arithmetic, but it stacks layers so the layer-type alternation of the
// accepted architecture is exercised: layers 0 and 1 are raw sliding-window
// attention, even layers from 2 run the dual-lane fine compressor with an
// indexer and a capped selection, and odd layers from 3 run the single-lane
// coarse compressor with no indexer.
//
// Two constants are deliberately scaled down. The accepted architecture uses a
// 128-token raw window and a ratio-128 coarse compressor; a sixteen-token tiny
// sequence can never cross either. tinyStackCoarseRatio stands in for 128 and
// tinyStackRawWindow for 128 so that both compressors emit rows, and rows fall
// out of the raw window and become attendable, inside the fixture sequence. The
// fine ratio (4) and the layer-type rule are the real ones. tinyStackRowCap
// stands in for the 512 selected-compressed-row cap on the indexed path; at
// tiny scale one row is the cap, and it binds because the fixture produces four
// fine rows.
const (
	tinyStackLayers      = 6
	tinyStackRawWindow   = 2
	tinyStackFineRatio   = 4 // matches the architecture
	tinyStackCoarseRatio = 8 // stands in for 128
	tinyStackRowCap      = 1 // stands in for 512
)

type tinyLayerKind int

const (
	tinyRawLayer tinyLayerKind = iota
	tinyFineLayer
	tinyCoarseLayer
)

func (k tinyLayerKind) String() string {
	switch k {
	case tinyRawLayer:
		return "raw"
	case tinyFineLayer:
		return "fine"
	default:
		return "coarse"
	}
}

// tinyLayerShape reports the attention shape of layer i under the rule stated in
// DESIGN.md and implemented by metalRuntime.bindLayer and evalLayer: layers 0
// and 1 are raw, even layers from 2 are fine and indexed, odd layers from 3 are
// coarse.
func tinyLayerShape(i int) (tinyLayerKind, int) {
	switch {
	case i < 2:
		return tinyRawLayer, 0
	case i%2 == 0:
		return tinyFineLayer, tinyStackFineRatio
	default:
		return tinyCoarseLayer, tinyStackCoarseRatio
	}
}

type tinyStackLayer struct {
	kind                              tinyLayerKind
	ratio                             int
	norm, q, kv, o, router            []float32
	routerBias                        []float32
	experts                           []tinyExpert
	hcAttnFn, hcAttnScale, hcAttnBase []float32
	hcFFNFn, hcFFNScale, hcFFNBase    []float32
	attention, indexer                compressorWeights
	indexQuery, indexWeight           []float32
	sink                              float32
}

type tinyStackWeights struct {
	layers                 []tinyStackLayer
	embedding              []float32
	outputHead, outputNorm []float32
}

type tinyLayerTrace struct {
	Layer      int       `json:"layer"`
	Kind       string    `json:"kind"`
	Norm       []float32 `json:"norm"`
	HCAttnMix  []float32 `json:"hc_attn_mix"`
	HCFFNMix   []float32 `json:"hc_ffn_mix"`
	Query      []float32 `json:"query"`
	Key        []float32 `json:"key"`
	Attention  []float32 `json:"attention"`
	Compressed []int     `json:"compressed"`
	Experts    []int     `json:"experts"`
	Weights    []float32 `json:"weights"`
	MoE        []float32 `json:"moe"`
	HC         []float32 `json:"hc"`
}

type tinyStackTrace struct {
	Token    int              `json:"token"`
	Position int              `json:"position"`
	Layers   []tinyLayerTrace `json:"layers"`
	Logits   []float32        `json:"logits"`
}

type tinyStack struct {
	weights tinyStackWeights
	caches  []*compressedAttention
	pos     int
}

func newTinyStack(w tinyStackWeights) (*tinyStack, error) {
	if len(w.layers) == 0 || len(w.embedding) != tinyVocab*tinyDim ||
		len(w.outputHead) != tinyVocab*tinyDim || len(w.outputNorm) != tinyDim {
		return nil, errShape
	}
	s := &tinyStack{weights: w}
	for i, layer := range w.layers {
		var (
			cache *compressedAttention
			err   error
		)
		switch layer.kind {
		case tinyRawLayer:
			cache, err = newRawAttention(tinyDim, tinyStackRawWindow, layer.sink)
		case tinyFineLayer:
			cache, err = newCompressedAttention(tinyDim, tinyDim, tinyStackRawWindow, layer.ratio,
				layer.attention, layer.indexer, layer.indexQuery, layer.indexWeight, layer.sink)
		default:
			cache, err = newCoarseAttention(tinyDim, tinyDim, tinyStackRawWindow, layer.ratio, layer.attention, layer.sink)
		}
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		s.caches = append(s.caches, cache)
	}
	return s, nil
}

func newTinyStackFromSeed(seed uint64) (*tinyStack, error) {
	return newTinyStack(makeTinyStackWeights(seed))
}

// Step evaluates one token through every layer and returns the per-layer trace
// and the output logits.
func (s *tinyStack) Step(token int) (tinyStackTrace, error) {
	if token < 0 || token >= tinyVocab {
		return tinyStackTrace{}, fmt.Errorf("token %d: %w", token, errShape)
	}
	ones := make([]float32, tinyDim*tinyStreams)
	for i := range ones {
		ones[i] = 1
	}
	residual := make([]float32, tinyDim*tinyStreams)
	for stream := range tinyStreams {
		copy(residual[stream*tinyDim:], s.weights.embedding[token*tinyDim:(token+1)*tinyDim])
	}
	trace := tinyStackTrace{Token: token, Position: s.pos}
	for i := range s.weights.layers {
		layer := &s.weights.layers[i]
		lt := tinyLayerTrace{Layer: i, Kind: layer.kind.String()}
		flat, err := rmsNorm(residual, ones, 1e-5)
		if err != nil {
			return tinyStackTrace{}, err
		}
		lt.HCAttnMix, err = matvec(layer.hcAttnFn, flat, tinyStreams*(tinyStreams+2), tinyDim*tinyStreams)
		if err != nil {
			return tinyStackTrace{}, err
		}
		pre, post, combine, err := hcSplit(lt.HCAttnMix, layer.hcAttnScale, layer.hcAttnBase, tinyStreams, tinySinkhornIterations, 1e-6)
		if err != nil {
			return tinyStackTrace{}, err
		}
		plain, err := hcWeightedSum(residual, pre, tinyDim)
		if err != nil {
			return tinyStackTrace{}, err
		}
		lt.Norm, err = rmsNorm(plain, layer.norm, 1e-5)
		if err != nil {
			return tinyStackTrace{}, err
		}
		q, err := matvec(layer.q, lt.Norm, tinyDim, tinyDim)
		if err != nil {
			return tinyStackTrace{}, err
		}
		k, err := matvec(layer.kv, lt.Norm, tinyDim, tinyDim)
		if err != nil {
			return tinyStackTrace{}, err
		}
		lt.Query, _ = rotary(q, s.pos, 10000)
		lt.Key, _ = rotary(k, s.pos, 10000)
		if err := s.caches[i].Add(lt.Norm, lt.Key, s.pos); err != nil {
			return tinyStackTrace{}, fmt.Errorf("layer %d: %w", i, err)
		}
		attn, selected, err := s.caches[i].Attend(lt.Query, lt.Norm, lt.Norm, s.pos, tinyStackRowCap)
		if err != nil {
			return tinyStackTrace{}, fmt.Errorf("layer %d: %w", i, err)
		}
		lt.Compressed = selected
		attn, _ = rotary(attn, -s.pos, 10000)
		lt.Attention, err = matvec(layer.o, attn, tinyDim, tinyDim)
		if err != nil {
			return tinyStackTrace{}, err
		}
		hc, err := hcPost(lt.Attention, residual, post, combine)
		if err != nil {
			return tinyStackTrace{}, err
		}
		flat, err = rmsNorm(hc, ones, 1e-5)
		if err != nil {
			return tinyStackTrace{}, err
		}
		lt.HCFFNMix, err = matvec(layer.hcFFNFn, flat, tinyStreams*(tinyStreams+2), tinyDim*tinyStreams)
		if err != nil {
			return tinyStackTrace{}, err
		}
		pre, post, combine, err = hcSplit(lt.HCFFNMix, layer.hcFFNScale, layer.hcFFNBase, tinyStreams, tinySinkhornIterations, 1e-6)
		if err != nil {
			return tinyStackTrace{}, err
		}
		ffnInput, err := hcWeightedSum(hc, pre, tinyDim)
		if err != nil {
			return tinyStackTrace{}, err
		}
		ffnInput, err = rmsNorm(ffnInput, layer.norm, 1e-5)
		if err != nil {
			return tinyStackTrace{}, err
		}
		gate, err := matvec(layer.router, ffnInput, tinyExperts, tinyDim)
		if err != nil {
			return tinyStackTrace{}, err
		}
		decision, err := routeExperts(gate, layer.routerBias, tinyTopK)
		if err != nil {
			return tinyStackTrace{}, err
		}
		lt.MoE, err = expertSum(ffnInput, layer.experts, decision)
		if err != nil {
			return tinyStackTrace{}, err
		}
		lt.Experts = append([]int(nil), decision.SelectedExperts...)
		lt.Weights = append([]float32(nil), decision.Weights...)
		residual, err = hcPost(lt.MoE, hc, post, combine)
		if err != nil {
			return tinyStackTrace{}, err
		}
		lt.HC = residual
		if err := requireFinite(fmt.Sprintf("tiny stack layer %d", i), lt.Norm, lt.Query, lt.Key, lt.Attention, lt.MoE, lt.HC); err != nil {
			return tinyStackTrace{}, err
		}
		trace.Layers = append(trace.Layers, lt)
	}
	head, err := hcWeightedSum(residual, []float32{0.55, 0.45}, tinyDim)
	if err != nil {
		return tinyStackTrace{}, err
	}
	head, err = rmsNorm(head, s.weights.outputNorm, 1e-5)
	if err != nil {
		return tinyStackTrace{}, err
	}
	trace.Logits, err = matvec(s.weights.outputHead, head, tinyVocab, tinyDim)
	if err != nil {
		return tinyStackTrace{}, err
	}
	if err := requireFinite("tiny stack logits", trace.Logits); err != nil {
		return tinyStackTrace{}, err
	}
	s.pos++
	return trace, nil
}

// makeTinyStackWeights derives every stack weight from seed. Each layer draws
// from its own stream so a layer's weights do not depend on the layer count.
func makeTinyStackWeights(seed uint64) tinyStackWeights {
	model := tinyRand{state: seed}
	draw := func(r *tinyRand, count int) []float32 {
		out := make([]float32, count)
		for i := range out {
			out[i] = r.next()
		}
		return out
	}
	w := tinyStackWeights{
		embedding:  draw(&model, tinyVocab*tinyDim),
		outputNorm: draw(&model, tinyDim),
		outputHead: draw(&model, tinyVocab*tinyDim),
	}
	for i := range w.outputNorm {
		w.outputNorm[i] += 1.25
	}
	for i := range tinyStackLayers {
		w.layers = append(w.layers, makeTinyStackLayer(seed, i))
	}
	return w
}

func makeTinyStackLayer(seed uint64, i int) tinyStackLayer {
	r := tinyRand{state: seed ^ (uint64(i+1) * 0x9e3779b97f4a7c15)}
	n := func(count int) []float32 {
		out := make([]float32, count)
		for i := range out {
			out[i] = r.next()
		}
		return out
	}
	kind, ratio := tinyLayerShape(i)
	l := tinyStackLayer{
		kind:        kind,
		ratio:       ratio,
		norm:        n(tinyDim),
		q:           n(tinyDim * tinyDim),
		kv:          n(tinyDim * tinyDim),
		o:           n(tinyDim * tinyDim),
		router:      n(tinyExperts * tinyDim),
		routerBias:  n(tinyExperts),
		hcAttnFn:    n(tinyStreams * (tinyStreams + 2) * tinyDim * tinyStreams),
		hcAttnScale: n(3),
		hcAttnBase:  n(tinyStreams * (tinyStreams + 2)),
		hcFFNFn:     n(tinyStreams * (tinyStreams + 2) * tinyDim * tinyStreams),
		hcFFNScale:  n(3),
		hcFFNBase:   n(tinyStreams * (tinyStreams + 2)),
	}
	for j := range l.norm {
		l.norm[j] += 1.25
	}
	l.sink = r.next()
	for range tinyExperts {
		l.experts = append(l.experts, tinyExpert{tinyInter, n(tinyInter * tinyDim), n(tinyInter * tinyDim), n(tinyDim * tinyInter)})
	}
	switch kind {
	case tinyFineLayer:
		width := 2 * tinyDim
		l.attention = compressorWeights{n(width * tinyDim), n(width * tinyDim), n(ratio * width), n(tinyDim)}
		l.indexer = compressorWeights{n(width * tinyDim), n(width * tinyDim), n(ratio * width), n(tinyDim)}
		l.indexQuery = n(tinyDim * tinyDim)
		l.indexWeight = n(tinyDim)
	case tinyCoarseLayer:
		l.attention = compressorWeights{n(tinyDim * tinyDim), n(tinyDim * tinyDim), n(ratio * tinyDim), n(tinyDim)}
	}
	return l
}

// GGUF loading.
//
// tinyStackFromGGUF rebuilds the stack from a parsed GGUF file so the tiny
// model exercises the real loader rather than bypassing it. It reads the layer
// count and shape from metadata and every weight from an F32 tensor, using the
// tensor names of the accepted architecture.
func tinyStackFromGGUF(g *ggufFile, r io.ReaderAt) (*tinyStack, error) {
	if v, ok := g.metadata["general.architecture"]; !ok || v.value != "deepseek4" {
		return nil, fmt.Errorf("%w: architecture is not deepseek4", errInvalidGGUF)
	}
	meta := func(key string) (int, error) {
		v, ok := g.metadata[key]
		if !ok {
			return 0, fmt.Errorf("%w: missing metadata %q", errInvalidGGUF, key)
		}
		n, ok := valueUint(v)
		if !ok || n > math.MaxInt32 {
			return 0, fmt.Errorf("%w: metadata %q is not an unsigned integer", errInvalidGGUF, key)
		}
		return int(n), nil
	}
	var (
		w     tinyStackWeights
		first error
	)
	get := func(key string) int {
		n, err := meta(key)
		if err != nil && first == nil {
			first = err
		}
		return n
	}
	layers := get("deepseek4.block_count")
	dim := get("deepseek4.embedding_length")
	vocab := get("deepseek4.vocab_size")
	experts := get("deepseek4.expert_count")
	used := get("deepseek4.expert_used_count")
	inter := get("deepseek4.feed_forward_length")
	streams := get("deepseek4.hyper_connection.stream_count")
	sinkhorn := get("deepseek4.hyper_connection.sinkhorn_iterations")
	window := get("deepseek4.attention.raw_window")
	fine := get("deepseek4.attention.compression_ratio")
	coarse := get("deepseek4.attention.coarse_compression_ratio")
	rowCap := get("deepseek4.attention.selected_row_cap")
	if first != nil {
		return nil, first
	}
	// The tiny reference implementation is compiled for one shape. Refuse a file
	// that claims another rather than silently reinterpreting its tensors.
	for _, c := range []struct {
		name      string
		got, want int
	}{
		{"block_count", layers, tinyStackLayers},
		{"embedding_length", dim, tinyDim},
		{"vocab_size", vocab, tinyVocab},
		{"expert_count", experts, tinyExperts},
		{"expert_used_count", used, tinyTopK},
		{"feed_forward_length", inter, tinyInter},
		{"stream_count", streams, tinyStreams},
		{"sinkhorn_iterations", sinkhorn, tinySinkhornIterations},
		{"raw_window", window, tinyStackRawWindow},
		{"compression_ratio", fine, tinyStackFineRatio},
		{"coarse_compression_ratio", coarse, tinyStackCoarseRatio},
		{"selected_row_cap", rowCap, tinyStackRowCap},
	} {
		if c.got != c.want {
			return nil, fmt.Errorf("%w: %s %d, want %d", errUnsupported, c.name, c.got, c.want)
		}
	}
	index := make(map[string]ggufTensor, len(g.tensors))
	for _, t := range g.tensors {
		index[t.name] = t
	}
	read := func(name string, count int) []float32 {
		out, err := readGGUFFloats(g, r, index, name, count)
		if err != nil && first == nil {
			first = err
		}
		return out
	}
	w.embedding = read("token_embd.weight", tinyVocab*tinyDim)
	w.outputNorm = read("output_norm.weight", tinyDim)
	w.outputHead = read("output.weight", tinyVocab*tinyDim)
	for i := range layers {
		kind, ratio := tinyLayerShape(i)
		p := fmt.Sprintf("blk.%d.", i)
		l := tinyStackLayer{
			kind:        kind,
			ratio:       ratio,
			norm:        read(p+"attn_norm.weight", tinyDim),
			q:           read(p+"attn_q_b.weight", tinyDim*tinyDim),
			kv:          read(p+"attn_kv.weight", tinyDim*tinyDim),
			o:           read(p+"attn_output_b.weight", tinyDim*tinyDim),
			router:      read(p+"ffn_gate_inp.weight", tinyExperts*tinyDim),
			routerBias:  read(p+"exp_probs_b.bias", tinyExperts),
			hcAttnFn:    read(p+"hc_attn_fn.weight", tinyStreams*(tinyStreams+2)*tinyDim*tinyStreams),
			hcAttnScale: read(p+"hc_attn_scale.weight", 3),
			hcAttnBase:  read(p+"hc_attn_base.weight", tinyStreams*(tinyStreams+2)),
			hcFFNFn:     read(p+"hc_ffn_fn.weight", tinyStreams*(tinyStreams+2)*tinyDim*tinyStreams),
			hcFFNScale:  read(p+"hc_ffn_scale.weight", 3),
			hcFFNBase:   read(p+"hc_ffn_base.weight", tinyStreams*(tinyStreams+2)),
		}
		if sink := read(p+"attn_sinks.weight", 1); len(sink) == 1 {
			l.sink = sink[0]
		}
		gate := read(p+"ffn_gate_exps.weight", tinyExperts*tinyInter*tinyDim)
		up := read(p+"ffn_up_exps.weight", tinyExperts*tinyInter*tinyDim)
		down := read(p+"ffn_down_exps.weight", tinyExperts*tinyDim*tinyInter)
		if first != nil {
			return nil, first
		}
		for e := range tinyExperts {
			l.experts = append(l.experts, tinyExpert{tinyInter,
				gate[e*tinyInter*tinyDim : (e+1)*tinyInter*tinyDim],
				up[e*tinyInter*tinyDim : (e+1)*tinyInter*tinyDim],
				down[e*tinyDim*tinyInter : (e+1)*tinyDim*tinyInter]})
		}
		switch kind {
		case tinyFineLayer:
			width := 2 * tinyDim
			l.attention = compressorWeights{
				read(p+"attn_compressor_kv.weight", width*tinyDim),
				read(p+"attn_compressor_gate.weight", width*tinyDim),
				read(p+"attn_compressor_ape.weight", ratio*width),
				read(p+"attn_compressor_norm.weight", tinyDim),
			}
			l.indexer = compressorWeights{
				read(p+"indexer_compressor_kv.weight", width*tinyDim),
				read(p+"indexer_compressor_gate.weight", width*tinyDim),
				read(p+"indexer_compressor_ape.weight", ratio*width),
				read(p+"indexer_compressor_norm.weight", tinyDim),
			}
			l.indexQuery = read(p+"indexer.attn_q_b.weight", tinyDim*tinyDim)
			l.indexWeight = read(p+"indexer.proj.weight", tinyDim)
		case tinyCoarseLayer:
			l.attention = compressorWeights{
				read(p+"attn_compressor_kv.weight", tinyDim*tinyDim),
				read(p+"attn_compressor_gate.weight", tinyDim*tinyDim),
				read(p+"attn_compressor_ape.weight", ratio*tinyDim),
				read(p+"attn_compressor_norm.weight", tinyDim),
			}
		}
		w.layers = append(w.layers, l)
	}
	if first != nil {
		return nil, first
	}
	return newTinyStack(w)
}

func readGGUFFloats(g *ggufFile, r io.ReaderAt, index map[string]ggufTensor, name string, count int) ([]float32, error) {
	t, ok := index[name]
	if !ok {
		return nil, fmt.Errorf("%w: missing tensor %q", errInvalidGGUF, name)
	}
	if t.typeID != 0 {
		return nil, fmt.Errorf("%w: tensor %q type %d, want float32", errUnsupported, name, t.typeID)
	}
	elements := uint64(1)
	for _, d := range t.dimensions {
		var ok bool
		if elements, ok = multiply(elements, d); !ok {
			return nil, fmt.Errorf("%w: tensor %q element count overflow", errInvalidGGUF, name)
		}
	}
	if elements != uint64(count) {
		return nil, fmt.Errorf("%w: tensor %q has %d elements, want %d", errInvalidGGUF, name, elements, count)
	}
	start, ok := add(g.dataOffset, t.offset)
	if !ok {
		return nil, fmt.Errorf("%w: tensor %q offset overflow", errInvalidGGUF, name)
	}
	buf := make([]byte, count*4)
	if _, err := r.ReadAt(buf, int64(start)); err != nil {
		return nil, fmt.Errorf("read tensor %q: %w", name, err)
	}
	out := make([]float32, count)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return out, nil
}
