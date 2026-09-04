// Command generate independently computes the tiny DS4 CPU fixture.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
)

const (
	dim                = 4
	vocab              = 6
	nexpert            = 4
	topk               = 2
	inter              = 3
	streams            = 2
	sinkhornIterations = 20
)

type expert struct{ gate, up, down []float32 }
type compressor struct{ kv, gate, ape, norm []float32 }
type weights struct {
	embed, norm, q, kv, o, router, bias []float32
	experts                             []expert
	attnfn, attnscale, attnbase         []float32
	ffnfn, ffnscale, ffnbase            []float32
	indexq, indexw, head, outnorm       []float32
	sink                                float32
	attention, indexer                  compressor
}
type row struct {
	key, value, index []float32
	end               int
}
type compstate struct{ kv, score []float32 }
type cache struct {
	raw, compressed    []row
	attention, indexer compstate
}
type trace struct {
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
type payload struct {
	Schema string  `json:"schema"`
	Seed   uint64  `json:"seed"`
	Prompt []int   `json:"prompt"`
	Decode []int   `json:"decode"`
	Traces []trace `json:"traces"`
}
type manifest struct {
	Payload     payload `json:"payload"`
	SHA256      string  `json:"payload_sha256"`
	Stack       stack   `json:"stack"`
	StackSHA256 string  `json:"stack_sha256"`
}

func main() {
	out := flag.String("out", "", "output path")
	flag.Parse()
	if *out == "" {
		fmt.Fprintln(os.Stderr, "missing -out")
		os.Exit(2)
	}
	const seed = 0x445334
	w := makeWeights(seed)
	c := new(cache)
	p := payload{Schema: "ds4-min-cpu-fixture-v1", Seed: seed, Prompt: []int{0, 2, 4}}
	tokens := append([]int(nil), p.Prompt...)
	for step := 0; step < len(p.Prompt)+4; step++ {
		t := run(w, c, tokens[step], step)
		p.Traces = append(p.Traces, t)
		if step+1 == len(tokens) && len(p.Decode) < 4 {
			next := argmax(t.Logits)
			p.Decode = append(p.Decode, next)
			tokens = append(tokens, next)
		}
	}
	st := runStack(seed)
	b, _ := json.Marshal(p)
	sum := sha256.Sum256(b)
	sb, _ := json.Marshal(st)
	stackSum := sha256.Sum256(sb)
	result, _ := json.MarshalIndent(manifest{p, hex.EncodeToString(sum[:]), st, hex.EncodeToString(stackSum[:])}, "", "  ")
	result = append(result, '\n')
	if err := os.WriteFile(*out, result, 0o644); err != nil {
		panic(err)
	}
}

func run(w weights, c *cache, token, pos int) trace {
	emb := clone(w.embed[token*dim : (token+1)*dim])
	residual := append(clone(emb), emb...)
	flat := rms(residual, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	attnMix := mv(w.attnfn, flat, streams*(streams+2))
	pre, post, comb := split(attnMix, w.attnscale, w.attnbase)
	plain := weighted(residual, pre)
	norm := rms(plain, w.norm)
	q := rotate(mv(w.q, norm, dim), pos)
	k := rotate(mv(w.kv, norm, dim), pos)
	add(c, w, norm, k, pos)
	attn, selected := attend(c, w, q, norm)
	attn = mv(w.o, rotate(attn, -pos), dim)
	hc := postHC(attn, residual, post, comb)
	ffnMix := mv(w.ffnfn, rms(hc, []float32{1, 1, 1, 1, 1, 1, 1, 1}), streams*(streams+2))
	pre, post, comb = split(ffnMix, w.ffnscale, w.ffnbase)
	ffn := rms(weighted(hc, pre), w.norm)
	ids, rw := route(mv(w.router, ffn, nexpert), w.bias)
	mix := make([]float32, dim)
	for slot, id := range ids {
		e := w.experts[id]
		hidden := swiglu(mv(e.gate, ffn, inter), mv(e.up, ffn, inter))
		value := mv(e.down, hidden, dim)
		for i := range mix {
			mix[i] += rw[slot] * value[i]
		}
	}
	hc = postHC(mix, hc, post, comb)
	out := rms(weighted(hc, []float32{.55, .45}), w.outnorm)
	logits := mv(w.head, out, vocab)
	return trace{token, pos, norm, attnMix, ffnMix, q, k, attn, selected, ids, rw, mix, hc, logits}
}

func makeWeights(seed uint64) weights {
	r := seed
	next := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			r = r*6364136223846793005 + 1442695040888963407
			x[i] = float32(int32(r>>40)) / float32(1<<23) * .25
		}
		return x
	}
	w := weights{embed: next(vocab * dim), norm: next(dim), q: next(dim * dim), kv: next(dim * dim), o: next(dim * dim), router: next(nexpert * dim), bias: next(nexpert), attnfn: next(streams * (streams + 2) * dim * streams), attnscale: next(3), attnbase: next(streams * (streams + 2)), ffnfn: next(streams * (streams + 2) * dim * streams), ffnscale: next(3), ffnbase: next(streams * (streams + 2)), indexq: next(dim * dim), indexw: next(dim), head: next(vocab * dim), outnorm: next(dim)}
	w.sink = next(1)[0]
	for i := range w.norm {
		w.norm[i] += 1.25
		w.outnorm[i] += 1.25
	}
	for range nexpert {
		w.experts = append(w.experts, expert{next(inter * dim), next(inter * dim), next(dim * inter)})
	}
	makeComp := func() compressor {
		return compressor{next(2 * dim * dim), next(2 * dim * dim), next(4 * 2 * dim), next(dim)}
	}
	w.attention = makeComp()
	w.indexer = makeComp()
	return w
}
func clone(x []float32) []float32 { return append([]float32(nil), x...) }
func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
func mv(m, x []float32, rows int) []float32 {
	cols := len(x)
	o := make([]float32, rows)
	for r := range rows {
		for c := range cols {
			o[r] += m[r*cols+c] * x[c]
		}
	}
	return o
}
func rms(x, w []float32) []float32 {
	var s float64
	for _, v := range x {
		s += float64(v * v)
	}
	scale := float32(1 / math.Sqrt(s/float64(len(x))+1e-5))
	o := make([]float32, len(x))
	for i := range x {
		o[i] = x[i] * scale * w[i]
	}
	return o
}
func rotate(x []float32, p int) []float32 {
	o := clone(x)
	for i := 0; i < len(x); i += 2 {
		a := float64(p) / math.Pow(10000, float64(i)/float64(len(x)))
		c, s := float32(math.Cos(a)), float32(math.Sin(a))
		o[i] = x[i]*c - x[i+1]*s
		o[i+1] = x[i]*s + x[i+1]*c
	}
	return o
}
func sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
func split(mix, scale, base []float32) ([]float32, []float32, []float32) {
	pre := make([]float32, streams)
	post := make([]float32, streams)
	c := make([]float32, streams*streams)
	for i := range streams {
		pre[i] = sigmoid(mix[i]*scale[0]+base[i]) + 1e-6
		post[i] = 2 * sigmoid(mix[streams+i]*scale[1]+base[streams+i])
	}
	for d := range streams {
		mx := float32(-math.MaxFloat32)
		for s := range streams {
			i := s + d*streams
			c[i] = mix[2*streams+i]*scale[2] + base[2*streams+i]
			mx = max(mx, c[i])
		}
		var sum float32
		for s := range streams {
			i := s + d*streams
			c[i] = float32(math.Exp(float64(c[i] - mx)))
			sum += c[i]
		}
		for s := range streams {
			i := s + d*streams
			c[i] = c[i]/sum + 1e-6
		}
	}
	for it := 0; it < sinkhornIterations; it++ {
		for s := range streams {
			var sum float32
			for d := range streams {
				sum += c[s+d*streams]
			}
			for d := range streams {
				c[s+d*streams] /= sum + 1e-6
			}
		}
		if it+1 < sinkhornIterations {
			for d := range streams {
				var sum float32
				for s := range streams {
					sum += c[s+d*streams]
				}
				for s := range streams {
					c[s+d*streams] /= sum + 1e-6
				}
			}
		}
	}
	return pre, post, c
}
func weighted(x, w []float32) []float32 {
	o := make([]float32, dim)
	for h, v := range w {
		for d := range dim {
			o[d] += x[h*dim+d] * v
		}
	}
	return o
}
func postHC(block, res, post, comb []float32) []float32 {
	o := make([]float32, len(res))
	for dst := range streams {
		for d := range dim {
			o[dst*dim+d] = block[d] * post[dst]
			for src := range streams {
				o[dst*dim+d] += comb[dst+src*streams] * res[src*dim+d]
			}
		}
	}
	return o
}
func compUpdate(x []float32, pos int, w compressor, s *compstate) []float32 {
	const ratio = 4
	width := 2 * dim
	if s.kv == nil {
		s.kv = make([]float32, 2*ratio*width)
		s.score = make([]float32, 2*ratio*width)
		for i := range s.score {
			s.score[i] = -math.MaxFloat32
		}
	}
	kv, sc := mv(w.kv, x, width), mv(w.gate, x, width)
	mod := pos % ratio
	for j := range width {
		sc[j] += w.ape[mod*width+j]
	}
	row := ratio + mod
	copy(s.kv[row*width:], kv)
	copy(s.score[row*width:], sc)
	if (pos+1)%ratio != 0 {
		return nil
	}
	pool := make([]float32, dim)
	for j := range dim {
		mx := float32(-math.MaxFloat32)
		for r := range ratio {
			mx = max(mx, s.score[r*width+j])
			mx = max(mx, s.score[(ratio+r)*width+dim+j])
		}
		var den, sum float32
		for r := range ratio {
			a := float32(math.Exp(float64(s.score[r*width+j] - mx)))
			b := float32(math.Exp(float64(s.score[(ratio+r)*width+dim+j] - mx)))
			den += a + b
			sum += a*s.kv[r*width+j] + b*s.kv[(ratio+r)*width+dim+j]
		}
		pool[j] = sum / den
	}
	pool = rotate(rms(pool, w.norm), pos+1-ratio)
	copy(s.kv[:ratio*width], s.kv[ratio*width:])
	copy(s.score[:ratio*width], s.score[ratio*width:])
	copy(s.kv[ratio*width:], s.kv[:ratio*width])
	copy(s.score[ratio*width:], s.score[:ratio*width])
	return pool
}
func add(c *cache, w weights, x, kv []float32, pos int) {
	r := row{key: clone(kv), value: clone(kv), end: pos}
	c.raw = append(c.raw, r)
	if len(c.raw) > 2 {
		c.raw = c.raw[len(c.raw)-2:]
	}
	a := compUpdate(x, pos, w.attention, &c.attention)
	i := compUpdate(x, pos, w.indexer, &c.indexer)
	if a != nil {
		c.compressed = append(c.compressed, row{a, clone(a), i, pos})
	}
}
func attend(c *cache, w weights, q, x []float32) ([]float32, []int) {
	iq := rotate(mv(w.indexq, x, dim), c.raw[len(c.raw)-1].end)
	iw := mv(w.indexw, x, 1)[0] / 2
	type scored struct {
		r row
		i int
		s float32
	}
	var cs []scored
	for i, r := range c.compressed {
		if r.end < c.raw[0].end {
			cs = append(cs, scored{r, i, max(float32(0), dot(iq, r.index)) * iw})
		}
	}
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].s != cs[j].s {
			return cs[i].s > cs[j].s
		}
		return cs[i].i < cs[j].i
	})
	if len(cs) > 1 {
		cs = cs[:1]
	}
	rows := make([]row, 0, len(cs)+len(c.raw))
	ids := make([]int, len(cs))
	for i, v := range cs {
		rows = append(rows, v.r)
		ids[i] = v.i
	}
	rows = append(rows, c.raw...)
	scores := make([]float32, len(rows))
	mx := w.sink
	for i, r := range rows {
		scores[i] = dot(q, r.key) / 2
		mx = max(mx, scores[i])
	}
	den := float32(math.Exp(float64(w.sink - mx)))
	for i := range scores {
		scores[i] = float32(math.Exp(float64(scores[i] - mx)))
		den += scores[i]
	}
	o := make([]float32, dim)
	for i, r := range rows {
		for d := range dim {
			o[d] += scores[i] / den * r.value[d]
		}
	}
	return o, ids
}
func route(logits, bias []float32) ([]int, []float32) {
	ids := []int{0, 1, 2, 3}
	sort.Slice(ids, func(i, j int) bool {
		a, b := logits[ids[i]]+bias[ids[i]], logits[ids[j]]+bias[ids[j]]
		if a != b {
			return a > b
		}
		return ids[i] < ids[j]
	})
	ids = ids[:topk]
	mx := max(logits[ids[0]], logits[ids[1]])
	w := []float32{float32(math.Exp(float64(logits[ids[0]] - mx))), float32(math.Exp(float64(logits[ids[1]] - mx)))}
	sum := w[0] + w[1]
	w[0] /= sum
	w[1] /= sum
	return ids, w
}
func swiglu(g, u []float32) []float32 {
	o := make([]float32, len(g))
	for i := range g {
		o[i] = g[i] / (1 + float32(math.Exp(float64(-g[i])))) * u[i]
	}
	return o
}
func argmax(x []float32) int {
	b := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[b] {
			b = i
		}
	}
	return b
}

// The multi-layer stack fixture.
//
// The accepted architecture alternates layer types: layers 0 and 1 are raw
// sliding-window attention, even layers from 2 use the dual-lane ratio-4
// compressor plus an indexer with a capped row selection, and odd layers from 3
// use the single-lane ratio-128 compressor with no indexer. A tiny sequence can
// never cross ratio 128 or a 128-token raw window, so this fixture substitutes
// coarseRatio for 128 and rawWindow for 128. The fine ratio and the alternation
// rule are the real ones.
const (
	stackLayers = 6
	rawWindow   = 2
	fineRatio   = 4
	coarseRatio = 8
	rowCap      = 1
	decodeSteps = 13
)

const (
	rawKind = iota
	fineKind
	coarseKind
)

func layerShape(i int) (int, int) {
	switch {
	case i < 2:
		return rawKind, 0
	case i%2 == 0:
		return fineKind, fineRatio
	default:
		return coarseKind, coarseRatio
	}
}

func kindName(k int) string {
	switch k {
	case rawKind:
		return "raw"
	case fineKind:
		return "fine"
	}
	return "coarse"
}

type layerW struct {
	kind, ratio                  int
	norm, q, kv, o, router, bias []float32
	experts                      []expert
	attnfn, attnscale, attnbase  []float32
	ffnfn, ffnscale, ffnbase     []float32
	attention, indexer           compressor
	indexq, indexw               []float32
	sink                         float32
}

type stackW struct {
	layers               []layerW
	embed, head, outnorm []float32
}

type lcache struct {
	raw, compressed []row
	attn, index     compstate
	ratio           int
	dual, indexed   bool
}

type layerTrace struct {
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

type stackTrace struct {
	Token    int          `json:"token"`
	Position int          `json:"position"`
	Layers   []layerTrace `json:"layers"`
	Logits   []float32    `json:"logits"`
}

type stack struct {
	Schema         string       `json:"schema"`
	Seed           uint64       `json:"seed"`
	Layers         int          `json:"layers"`
	RawWindow      int          `json:"raw_window"`
	FineRatio      int          `json:"fine_ratio"`
	CoarseRatio    int          `json:"coarse_ratio"`
	SelectedRowCap int          `json:"selected_row_cap"`
	Prompt         []int        `json:"prompt"`
	Decode         []int        `json:"decode"`
	Traces         []stackTrace `json:"traces"`
}

func draw(state *uint64, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		*state = *state*6364136223846793005 + 1442695040888963407
		x[i] = float32(int32(*state>>40)) / float32(1<<23) * .25
	}
	return x
}

func makeStack(seed uint64) stackW {
	model := seed
	w := stackW{embed: draw(&model, vocab*dim), outnorm: draw(&model, dim), head: draw(&model, vocab*dim)}
	for i := range w.outnorm {
		w.outnorm[i] += 1.25
	}
	for i := 0; i < stackLayers; i++ {
		w.layers = append(w.layers, makeLayer(seed, i))
	}
	return w
}

func makeLayer(seed uint64, i int) layerW {
	r := seed ^ (uint64(i+1) * 0x9e3779b97f4a7c15)
	n := func(c int) []float32 { return draw(&r, c) }
	kind, ratio := layerShape(i)
	l := layerW{kind: kind, ratio: ratio,
		norm: n(dim), q: n(dim * dim), kv: n(dim * dim), o: n(dim * dim),
		router: n(nexpert * dim), bias: n(nexpert),
		attnfn: n(streams * (streams + 2) * dim * streams), attnscale: n(3), attnbase: n(streams * (streams + 2)),
		ffnfn: n(streams * (streams + 2) * dim * streams), ffnscale: n(3), ffnbase: n(streams * (streams + 2))}
	for j := range l.norm {
		l.norm[j] += 1.25
	}
	l.sink = n(1)[0]
	for range nexpert {
		l.experts = append(l.experts, expert{n(inter * dim), n(inter * dim), n(dim * inter)})
	}
	switch kind {
	case fineKind:
		width := 2 * dim
		l.attention = compressor{n(width * dim), n(width * dim), n(ratio * width), n(dim)}
		l.indexer = compressor{n(width * dim), n(width * dim), n(ratio * width), n(dim)}
		l.indexq = n(dim * dim)
		l.indexw = n(dim)
	case coarseKind:
		l.attention = compressor{n(dim * dim), n(dim * dim), n(ratio * dim), n(dim)}
	}
	return l
}

func newLCache(l layerW) *lcache {
	c := &lcache{ratio: l.ratio, dual: l.kind == fineKind, indexed: l.kind == fineKind}
	if l.kind == rawKind {
		return c
	}
	width, rows := dim, l.ratio
	if c.dual {
		width, rows = 2*dim, 2*l.ratio
	}
	fill := func(s *compstate) {
		s.kv = make([]float32, rows*width)
		s.score = make([]float32, rows*width)
		for i := range s.score {
			s.score[i] = -math.MaxFloat32
		}
	}
	fill(&c.attn)
	if c.indexed {
		fill(&c.index)
	}
	return c
}

func poolUpdate(x []float32, pos, ratio int, dual bool, w compressor, s *compstate) []float32 {
	width := dim
	if dual {
		width *= 2
	}
	kv, sc := mv(w.kv, x, width), mv(w.gate, x, width)
	mod := pos % ratio
	for j := range width {
		sc[j] += w.ape[mod*width+j]
	}
	r := mod
	if dual {
		r += ratio
	}
	copy(s.kv[r*width:], kv)
	copy(s.score[r*width:], sc)
	if (pos+1)%ratio != 0 {
		return nil
	}
	pool := make([]float32, dim)
	for j := range dim {
		mx := float32(-math.MaxFloat32)
		for k := range ratio {
			mx = max(mx, s.score[k*width+j])
			if dual {
				mx = max(mx, s.score[(ratio+k)*width+dim+j])
			}
		}
		var den, sum float32
		for k := range ratio {
			a := float32(math.Exp(float64(s.score[k*width+j] - mx)))
			den += a
			sum += a * s.kv[k*width+j]
			if dual {
				b := float32(math.Exp(float64(s.score[(ratio+k)*width+dim+j] - mx)))
				den += b
				sum += b * s.kv[(ratio+k)*width+dim+j]
			}
		}
		pool[j] = sum / den
	}
	pool = rotate(rms(pool, w.norm), pos+1-ratio)
	if dual {
		copy(s.kv[:ratio*width], s.kv[ratio*width:])
		copy(s.score[:ratio*width], s.score[ratio*width:])
		copy(s.kv[ratio*width:], s.kv[:ratio*width])
		copy(s.score[ratio*width:], s.score[:ratio*width])
	}
	return pool
}

func addStack(c *lcache, l layerW, x, kv []float32, pos int) {
	c.raw = append(c.raw, row{key: clone(kv), value: clone(kv), end: pos})
	if len(c.raw) > rawWindow {
		c.raw = c.raw[len(c.raw)-rawWindow:]
	}
	if c.ratio == 0 {
		return
	}
	a := poolUpdate(x, pos, c.ratio, c.dual, l.attention, &c.attn)
	var idx []float32
	if c.indexed {
		idx = poolUpdate(x, pos, c.ratio, c.dual, l.indexer, &c.index)
	}
	if a != nil {
		c.compressed = append(c.compressed, row{a, clone(a), idx, pos})
	}
}

func attendStack(c *lcache, l layerW, q, x []float32, pos int) ([]float32, []int) {
	type scored struct {
		r row
		i int
		s float32
	}
	var iq []float32
	var iw float32
	if c.indexed {
		iq = rotate(mv(l.indexq, x, dim), pos)
		iw = mv(l.indexw, x, 1)[0] / 2
	}
	oldest := c.raw[0].end
	var cs []scored
	for i, r := range c.compressed {
		if r.end >= oldest {
			continue
		}
		var sc float32
		if c.indexed {
			sc = max(float32(0), dot(iq, r.index)) * iw
		}
		cs = append(cs, scored{r, i, sc})
	}
	if c.indexed {
		sort.Slice(cs, func(i, j int) bool {
			if cs[i].s != cs[j].s {
				return cs[i].s > cs[j].s
			}
			return cs[i].i < cs[j].i
		})
		if len(cs) > rowCap {
			cs = cs[:rowCap]
		}
	}
	rows := make([]row, 0, len(cs)+len(c.raw))
	ids := make([]int, len(cs))
	for i, v := range cs {
		rows = append(rows, v.r)
		ids[i] = v.i
	}
	rows = append(rows, c.raw...)
	scores := make([]float32, len(rows))
	mx := l.sink
	for i, r := range rows {
		scores[i] = dot(q, r.key) / 2
		mx = max(mx, scores[i])
	}
	den := float32(math.Exp(float64(l.sink - mx)))
	for i := range scores {
		scores[i] = float32(math.Exp(float64(scores[i] - mx)))
		den += scores[i]
	}
	o := make([]float32, dim)
	for i, r := range rows {
		for d := range dim {
			o[d] += scores[i] / den * r.value[d]
		}
	}
	return o, ids
}

func stackStep(w stackW, caches []*lcache, token, pos int) stackTrace {
	ones := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	residual := make([]float32, dim*streams)
	for s := range streams {
		copy(residual[s*dim:], w.embed[token*dim:(token+1)*dim])
	}
	t := stackTrace{Token: token, Position: pos}
	for i, l := range w.layers {
		lt := layerTrace{Layer: i, Kind: kindName(l.kind)}
		lt.HCAttnMix = mv(l.attnfn, rms(residual, ones), streams*(streams+2))
		pre, post, comb := split(lt.HCAttnMix, l.attnscale, l.attnbase)
		lt.Norm = rms(weighted(residual, pre), l.norm)
		lt.Query = rotate(mv(l.q, lt.Norm, dim), pos)
		lt.Key = rotate(mv(l.kv, lt.Norm, dim), pos)
		addStack(caches[i], l, lt.Norm, lt.Key, pos)
		attn, ids := attendStack(caches[i], l, lt.Query, lt.Norm, pos)
		lt.Compressed = ids
		lt.Attention = mv(l.o, rotate(attn, -pos), dim)
		hc := postHC(lt.Attention, residual, post, comb)
		lt.HCFFNMix = mv(l.ffnfn, rms(hc, ones), streams*(streams+2))
		pre, post, comb = split(lt.HCFFNMix, l.ffnscale, l.ffnbase)
		ffn := rms(weighted(hc, pre), l.norm)
		ids2, rw := route(mv(l.router, ffn, nexpert), l.bias)
		mix := make([]float32, dim)
		for slot, id := range ids2 {
			e := l.experts[id]
			value := mv(e.down, swiglu(mv(e.gate, ffn, inter), mv(e.up, ffn, inter)), dim)
			for j := range mix {
				mix[j] += rw[slot] * value[j]
			}
		}
		lt.Experts, lt.Weights, lt.MoE = ids2, rw, mix
		residual = postHC(mix, hc, post, comb)
		lt.HC = residual
		t.Layers = append(t.Layers, lt)
	}
	out := rms(weighted(residual, []float32{.55, .45}), w.outnorm)
	t.Logits = mv(w.head, out, vocab)
	return t
}

func runStack(seed uint64) stack {
	w := makeStack(seed)
	caches := make([]*lcache, len(w.layers))
	for i, l := range w.layers {
		caches[i] = newLCache(l)
	}
	st := stack{Schema: "ds4-min-cpu-stack-fixture-v1", Seed: seed, Layers: stackLayers,
		RawWindow: rawWindow, FineRatio: fineRatio, CoarseRatio: coarseRatio,
		SelectedRowCap: rowCap, Prompt: []int{0, 2, 4}}
	tokens := append([]int(nil), st.Prompt...)
	for step := 0; step < len(st.Prompt)+decodeSteps; step++ {
		t := stackStep(w, caches, tokens[step], step)
		st.Traces = append(st.Traces, t)
		if step+1 == len(tokens) && len(st.Decode) < decodeSteps {
			next := argmax(t.Logits)
			st.Decode = append(st.Decode, next)
			tokens = append(tokens, next)
		}
	}
	return st
}
