package main

import "fmt"

const (
	tinyDim                = 4
	tinyVocab              = 6
	tinyExperts            = 4
	tinyTopK               = 2
	tinyInter              = 3
	tinyStreams            = 2
	tinySinkhornIterations = 20
)

type tinyWeights struct {
	embedding                         []float32
	norm, q, kv, o, router            []float32
	routerBias                        []float32
	experts                           []tinyExpert
	hcAttnFn, hcAttnScale, hcAttnBase []float32
	hcFFNFn, hcFFNScale, hcFFNBase    []float32
	attention, indexer                compressorWeights
	indexQuery, indexWeight           []float32
	attentionSink                     float32
	outputHead, outputNorm            []float32
}

type tinyTrace struct {
	Token      int       `json:"token"`
	Position   int       `json:"position"`
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
	Logits     []float32 `json:"logits"`
}

type tinyCPU struct {
	weights tinyWeights
	cache   *compressedAttention
	pos     int
}

func newTinyCPU(seed uint64) *tinyCPU {
	w := makeTinyWeights(seed)
	cache, _ := newCompressedAttention(tinyDim, tinyDim, 2, 4, w.attention, w.indexer, w.indexQuery, w.indexWeight, w.attentionSink)
	return &tinyCPU{weights: w, cache: cache}
}

func (m *tinyCPU) Step(token int) (tinyTrace, error) {
	if token < 0 || token >= tinyVocab {
		return tinyTrace{}, fmt.Errorf("token %d: %w", token, errShape)
	}
	w := &m.weights
	embedding := append([]float32(nil), w.embedding[token*tinyDim:(token+1)*tinyDim]...)
	residual := make([]float32, tinyDim*tinyStreams)
	for stream := range tinyStreams {
		copy(residual[stream*tinyDim:], embedding)
	}
	flat, err := rmsNorm(residual, []float32{1, 1, 1, 1, 1, 1, 1, 1}, 1e-5)
	if err != nil {
		return tinyTrace{}, err
	}
	attnMix, err := matvec(w.hcAttnFn, flat, tinyStreams*(tinyStreams+2), tinyDim*tinyStreams)
	if err != nil {
		return tinyTrace{}, err
	}
	pre, post, combine, err := hcSplit(attnMix, w.hcAttnScale, w.hcAttnBase, tinyStreams, tinySinkhornIterations, 1e-6)
	if err != nil {
		return tinyTrace{}, err
	}
	plain, _ := hcWeightedSum(residual, pre, tinyDim)
	norm, err := rmsNorm(plain, w.norm, 1e-5)
	if err != nil {
		return tinyTrace{}, err
	}
	q, _ := matvec(w.q, norm, tinyDim, tinyDim)
	k, _ := matvec(w.kv, norm, tinyDim, tinyDim)
	q, _ = rotary(q, m.pos, 10000)
	k, _ = rotary(k, m.pos, 10000)
	// The current KV row is visible to its own causal attention, as in
	// layer_attention_raw_swa_one.
	if err := m.cache.Add(norm, k, m.pos); err != nil {
		return tinyTrace{}, err
	}
	attn, selected, err := m.cache.Attend(q, norm, norm, m.pos, 1)
	if err != nil {
		return tinyTrace{}, err
	}
	attn, _ = rotary(attn, -m.pos, 10000)
	attn, _ = matvec(w.o, attn, tinyDim, tinyDim)
	hc, _ := hcPost(attn, residual, post, combine)
	flat, err = rmsNorm(hc, []float32{1, 1, 1, 1, 1, 1, 1, 1}, 1e-5)
	if err != nil {
		return tinyTrace{}, err
	}
	ffnMix, err := matvec(w.hcFFNFn, flat, tinyStreams*(tinyStreams+2), tinyDim*tinyStreams)
	if err != nil {
		return tinyTrace{}, err
	}
	pre, post, combine, err = hcSplit(ffnMix, w.hcFFNScale, w.hcFFNBase, tinyStreams, tinySinkhornIterations, 1e-6)
	if err != nil {
		return tinyTrace{}, err
	}
	ffnInput, _ := hcWeightedSum(hc, pre, tinyDim)
	ffnInput, _ = rmsNorm(ffnInput, w.norm, 1e-5)
	gate, _ := matvec(w.router, ffnInput, tinyExperts, tinyDim)
	decision, err := routeExperts(gate, w.routerBias, tinyTopK)
	if err != nil {
		return tinyTrace{}, err
	}
	moeOut, err := expertSum(ffnInput, w.experts, decision)
	if err != nil {
		return tinyTrace{}, err
	}
	hc, _ = hcPost(moeOut, hc, post, combine)
	head, _ := hcWeightedSum(hc, []float32{0.55, 0.45}, tinyDim)
	head, _ = rmsNorm(head, w.outputNorm, 1e-5)
	logits, _ := matvec(w.outputHead, head, tinyVocab, tinyDim)
	if err := requireFinite("tiny CPU step", norm, q, k, attn, moeOut, hc, logits); err != nil {
		return tinyTrace{}, err
	}
	trace := tinyTrace{token, m.pos, norm, attnMix, ffnMix, q, k, attn, selected,
		append([]int(nil), decision.SelectedExperts...), append([]float32(nil), decision.Weights...),
		moeOut, hc, logits}
	m.pos++
	return trace, nil
}

func makeTinyWeights(seed uint64) tinyWeights {
	r := tinyRand{state: seed}
	n := func(count int) []float32 {
		out := make([]float32, count)
		for i := range out {
			out[i] = r.next()
		}
		return out
	}
	w := tinyWeights{
		embedding:   n(tinyVocab * tinyDim),
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
		indexQuery:  n(tinyDim * tinyDim),
		indexWeight: n(tinyDim),
		outputHead:  n(tinyVocab * tinyDim),
		outputNorm:  n(tinyDim),
	}
	w.attentionSink = r.next()
	makeCompressor := func(ratio int) compressorWeights {
		width := tinyDim
		if ratio == 4 {
			width *= 2
		}
		return compressorWeights{n(width * tinyDim), n(width * tinyDim), n(ratio * width), n(tinyDim)}
	}
	for i := range w.norm {
		w.norm[i] += 1.25
		w.outputNorm[i] += 1.25
	}
	for range tinyExperts {
		w.experts = append(w.experts, tinyExpert{tinyInter, n(tinyInter * tinyDim), n(tinyInter * tinyDim), n(tinyDim * tinyInter)})
	}
	w.attention = makeCompressor(4)
	w.indexer = makeCompressor(4)
	return w
}

type tinyRand struct{ state uint64 }

func (r *tinyRand) next() float32 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	v := int32(r.state >> 40)
	return float32(v) / float32(1<<23) * 0.25
}

func argmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}
