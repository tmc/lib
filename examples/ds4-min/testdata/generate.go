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
	Payload payload `json:"payload"`
	SHA256  string  `json:"payload_sha256"`
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
	b, _ := json.Marshal(p)
	sum := sha256.Sum256(b)
	result, _ := json.MarshalIndent(manifest{p, hex.EncodeToString(sum[:])}, "", "  ")
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
