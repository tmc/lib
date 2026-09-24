package cudaagreement_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// The decoder-layer fixture is one fp32 Llama-style layer followed by the
// final norm and the LM head, so that a forward ends in logits:
//
//	x0 = embed(ids)
//	x1 = x0 + Wo·attn(rope(Wqkv·rmsnorm(x0)))
//	x2 = x1 + Wd·(silu(Wg·n) * Wu·n), n = rmsnorm(x1)
//	logits = Wlm·rmsnorm(x2)
//
// The GEMMs are cuBLAS fp32 calls. Everything between them is a chain kernel
// from testdata/decoder.cu: embed, three rmsnorms, rope_kv, softmax,
// merge_heads, two residual adds and silu_mul, ten launches in all. The
// attention is causal over L cached positions plus the T new ones.
type decoderShape struct {
	Name             string
	D, H, F, V, T, L int
}

func (s decoderShape) HD() int { return s.D / s.H }
func (s decoderShape) S() int  { return s.L + s.T }

func (s decoderShape) String() string {
	return fmt.Sprintf("%s D=%d H=%d hd=%d F=%d V=%d T=%d L=%d", s.Name, s.D, s.H, s.HD(), s.F, s.V, s.T, s.L)
}

// rows are the token columns the oracle checks: all of them at a small shape,
// the first, middle and last at a measured one.
func (s decoderShape) rows(all bool) []int {
	if all {
		r := make([]int, s.T)
		for i := range r {
			r[i] = i
		}
		return r
	}
	return slices.Compact([]int{0, s.T / 2, s.T - 1})
}

var (
	decoderCorrectness = []decoderShape{
		{Name: "prefill-small", D: 256, H: 4, F: 704, V: 1024, T: 16, L: 0},
		{Name: "decode-small", D: 256, H: 4, F: 704, V: 1024, T: 4, L: 12},
	}
	// The measured shapes are a 1B-class layer (Llama-style widths with
	// hd=128) at a 128-token prefill and a one-token decode behind 511
	// cached positions.
	decoderMeasured = []decoderShape{
		{Name: "prefill", D: 2048, H: 16, F: 5632, V: 32000, T: 128, L: 0},
		{Name: "decode", D: 2048, H: 16, F: 5632, V: 32000, T: 1, L: 511},
	}
)

const decoderEps = float32(1e-5)

// decoderInputs are the values the host uploads.
type decoderInputs struct {
	ids                                        []int32
	emb, wn1, wqkv, wo, wn2, wgu, wd, wnf, wlm []float32
	invFreq                                    []float32
	kPrefix, vPrefix                           []float32 // [H][L][hd], cache positions before L
	scale                                      float32
}

func newDecoderInputs(s decoderShape, seed uint64) *decoderInputs {
	r := rand.New(rand.NewPCG(seed, 36))
	uniform := func(n int, center, width float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(center + (2*r.Float64()-1)*width)
		}
		return v
	}
	hd := s.HD()
	in := &decoderInputs{
		emb:     uniform(s.V*s.D, 0, 1),
		wn1:     uniform(s.D, 1, 0.1),
		wqkv:    uniform(3*s.D*s.D, 0, 1/math.Sqrt(float64(s.D))),
		wo:      uniform(s.D*s.D, 0, 1/math.Sqrt(float64(s.D))),
		wn2:     uniform(s.D, 1, 0.1),
		wgu:     uniform(2*s.F*s.D, 0, 1/math.Sqrt(float64(s.D))),
		wd:      uniform(s.D*s.F, 0, 1/math.Sqrt(float64(s.F))),
		wnf:     uniform(s.D, 1, 0.1),
		wlm:     uniform(s.V*s.D, 0, 1/math.Sqrt(float64(s.D))),
		kPrefix: uniform(s.H*s.L*hd, 0, 1),
		vPrefix: uniform(s.H*s.L*hd, 0, 1),
		scale:   float32(1 / math.Sqrt(float64(hd))),
	}
	in.ids = make([]int32, s.T)
	for i := range in.ids {
		in.ids[i] = int32(r.IntN(s.V))
	}
	in.invFreq = make([]float32, hd/2)
	for i := range in.invFreq {
		in.invFreq[i] = float32(math.Pow(10000, -2*float64(i)/float64(hd)))
	}
	return in
}

// decoderBuffers are the activations of one forward, by name and length.
func decoderBuffers(s decoderShape) map[string]int {
	hd := s.HD()
	return map[string]int{
		"x0": s.T * s.D, "xn1": s.T * s.D, "qkv": s.T * 3 * s.D,
		"q": s.H * s.T * hd, "kc": s.H * s.S() * hd, "vc": s.H * s.S() * hd,
		"scores": s.H * s.T * s.S(), "probs": s.H * s.T * s.S(), "o": s.H * s.T * hd,
		"attn": s.T * s.D, "ao": s.T * s.D, "x1": s.T * s.D, "xn2": s.T * s.D,
		"gu": s.T * 2 * s.F, "h": s.T * s.F, "dn": s.T * s.D, "x2": s.T * s.D,
		"xnf": s.T * s.D, "logits": s.T * s.V,
	}
}

// Error bounds. Each stage is evaluated in float64 from the fp32 values the
// device actually fed it, so a bound covers one stage's rounding and never
// accumulates across stages. u is the fp32 unit roundoff. Every bound also
// carries the float64 reference's own error: γ(n, u64) for a sum of n terms
// and 16·u64·|ref| for an elementwise stage.
const (
	u   = 0x1p-24
	u64 = 0x1p-53
	// tiny absorbs results that underflow fp32's normal range, where a
	// relative bound means nothing.
	tiny = 0x1p-126
)

func gamma(n int, unit float64) float64 { return float64(n) * unit / (1 - float64(n)*unit) }

// stageCheck accumulates one stage's comparisons.
type stageCheck struct {
	n, bad int
	first  string
	worst  float64 // largest |got-ref|/bound seen
}

// see compares one element. where and args name it, and are formatted only
// for the first failure.
func (c *stageCheck) see(got float32, ref, bound float64, where string, args ...any) {
	c.n++
	diff := math.Abs(float64(got) - ref)
	if bound > 0 {
		c.worst = max(c.worst, diff/bound)
	}
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || diff > bound {
		c.bad++
		if c.first == "" {
			c.first = fmt.Sprintf(where+": got %.9g, want %.9g within %.3g", append(args, got, ref, bound)...)
		}
	}
}

func (c *stageCheck) err() error {
	switch {
	case c.n == 0:
		return fmt.Errorf("no elements checked")
	case c.bad > 0:
		return fmt.Errorf("%d of %d elements outside the bound; first %s", c.bad, c.n, c.first)
	}
	return nil
}

// decoderReport is the oracle's verdict per stage, in forward order.
type decoderReport struct {
	stages []string
	checks map[string]*stageCheck
}

func (r *decoderReport) stage(name string) *stageCheck {
	if r.checks == nil {
		r.checks = make(map[string]*stageCheck)
	}
	c := &stageCheck{}
	r.stages = append(r.stages, name)
	r.checks[name] = c
	return c
}

// failed returns the stages that failed.
func (r *decoderReport) failed() []string {
	var out []string
	for _, s := range r.stages {
		if r.checks[s].err() != nil {
			out = append(out, s)
		}
	}
	return out
}

func (r *decoderReport) String() string {
	var b strings.Builder
	for _, s := range r.stages {
		c := r.checks[s]
		verdict := "ok"
		if err := c.err(); err != nil {
			verdict = err.Error()
		}
		fmt.Fprintf(&b, "%-10s n=%-8d worst=%.3f %s\n", s, c.n, c.worst, verdict)
	}
	return b.String()
}

// checkDecoder checks every stage of one forward over the token columns in
// rows. st holds the device's activations, read back after the forward.
func checkDecoder(s decoderShape, in *decoderInputs, st map[string][]float32, rows []int) *decoderReport {
	r := &decoderReport{}
	D, H, F, hd, S, T, L := s.D, s.H, s.F, s.HD(), s.S(), s.T, s.L
	half := hd / 2
	elem := func(ref float64) float64 { return 16 * u64 * math.Abs(ref) }

	c := r.stage("embed")
	for _, t := range rows {
		for d := range D {
			c.see(st["x0"][t*D+d], float64(in.emb[int(in.ids[t])*D+d]), 0, "x0[%d,%d]", t, d)
		}
	}
	checkRMSNorm(r.stage("rmsnorm1"), D, st["x0"], in.wn1, st["xn1"], rows)
	checkLinear(r.stage("qkv"), in.wqkv, 3*D, D, st["xn1"], st["qkv"], rows)

	c = r.stage("rope_kv")
	for _, t := range rows {
		pos := L + t
		row := st["qkv"][t*3*D:]
		for h := range H {
			for i := range half {
				th := float64(float32(pos) * in.invFreq[i])
				cs, sn := math.Cos(th), math.Sin(th)
				rot := func(name string, a, b float64, out []float32) {
					r1, r2 := a*cs-b*sn, b*cs+a*sn
					b1 := 4*u*(math.Abs(a)+math.Abs(b)) + 3*u*(math.Abs(a*cs)+math.Abs(b*sn)) + elem(r1)
					b2 := 4*u*(math.Abs(a)+math.Abs(b)) + 3*u*(math.Abs(b*cs)+math.Abs(a*sn)) + elem(r2)
					c.see(out[i], r1, b1, "%s[h=%d,t=%d,%d]", name, h, t, i)
					c.see(out[i+half], r2, b2, "%s[h=%d,t=%d,%d]", name, h, t, i+half)
				}
				rot("q", float64(row[h*hd+i]), float64(row[h*hd+i+half]), st["q"][(h*T+t)*hd:])
				rot("k", float64(row[D+h*hd+i]), float64(row[D+h*hd+i+half]), st["kc"][(h*S+pos)*hd:])
				for _, j := range []int{i, i + half} {
					c.see(st["vc"][(h*S+pos)*hd+j], float64(row[2*D+h*hd+j]), 0, "v[h=%d,pos=%d,%d]", h, pos, j)
				}
			}
		}
	}
	// The positions before L are the cache the host uploaded; the forward
	// must not have touched them.
	for h := range H {
		for p := range L {
			for d := range hd {
				c.see(st["kc"][(h*S+p)*hd+d], float64(in.kPrefix[(h*L+p)*hd+d]), 0, "kc[h=%d,pos=%d,%d]", h, p, d)
				c.see(st["vc"][(h*S+p)*hd+d], float64(in.vPrefix[(h*L+p)*hd+d]), 0, "vc[h=%d,pos=%d,%d]", h, p, d)
			}
		}
	}

	// scores[h][t][s] = Σ_d kc[h][s][d]·q[h][t][d].
	c = r.stage("scores")
	g := gamma(hd+1, u) + gamma(hd+1, u64)
	for h := range H {
		for _, t := range rows {
			q := st["q"][(h*T+t)*hd:][:hd]
			for p := range S {
				k := st["kc"][(h*S+p)*hd:][:hd]
				var ref, abs float64
				for d := range hd {
					x := float64(k[d]) * float64(q[d])
					ref += x
					abs += math.Abs(x)
				}
				c.see(st["scores"][(h*T+t)*S+p], ref, g*abs, "scores[h=%d,t=%d,%d]", h, t, p)
			}
		}
	}

	c = r.stage("softmax")
	scale := float64(in.scale)
	for h := range H {
		for _, t := range rows {
			row := st["scores"][(h*T+t)*S:][:S]
			out := st["probs"][(h*T+t)*S:][:S]
			allowed := L + t + 1
			m := math.Inf(-1)
			for p := range allowed {
				m = max(m, scale*float64(row[p]))
			}
			var sum, emax float64
			e := make([]float64, allowed)
			for p := range allowed {
				z := scale * float64(row[p])
				sum += math.Exp(z - m)
				e[p] = 2 * u * (math.Abs(z) + math.Abs(m))
				emax = max(emax, e[p])
			}
			for p := range S {
				if p >= allowed {
					c.see(out[p], 0, 0, "masked probs[h=%d,t=%d,%d]", h, t, p)
					continue
				}
				ref := math.Exp(scale*float64(row[p])-m) / sum
				rel := e[p] + emax + gamma(S, u) + 10*u
				c.see(out[p], ref, rel*ref+tiny+elem(ref), "probs[h=%d,t=%d,%d]", h, t, p)
			}
		}
	}

	// o[h][t][d] = Σ_s vc[h][s][d]·probs[h][t][s].
	c = r.stage("pv")
	g = gamma(S+1, u) + gamma(S+1, u64)
	for h := range H {
		for _, t := range rows {
			p := st["probs"][(h*T+t)*S:][:S]
			for d := range hd {
				var ref, abs float64
				for j := range S {
					x := float64(st["vc"][(h*S+j)*hd+d]) * float64(p[j])
					ref += x
					abs += math.Abs(x)
				}
				c.see(st["o"][(h*T+t)*hd+d], ref, g*abs, "o[h=%d,t=%d,%d]", h, t, d)
			}
		}
	}

	c = r.stage("merge")
	for _, t := range rows {
		for h := range H {
			for d := range hd {
				c.see(st["attn"][t*D+h*hd+d], float64(st["o"][(h*T+t)*hd+d]), 0, "attn[%d,%d]", t, h*hd+d)
			}
		}
	}
	checkLinear(r.stage("oproj"), in.wo, D, D, st["attn"], st["ao"], rows)
	checkAdd(r.stage("residual1"), D, st["x0"], st["ao"], st["x1"], rows)
	checkRMSNorm(r.stage("rmsnorm2"), D, st["x1"], in.wn2, st["xn2"], rows)
	checkLinear(r.stage("gate_up"), in.wgu, 2*F, D, st["xn2"], st["gu"], rows)

	c = r.stage("silu_mul")
	for _, t := range rows {
		for j := range F {
			gate, up := float64(st["gu"][t*2*F+j]), float64(st["gu"][t*2*F+F+j])
			if math.Abs(gate) >= 80 {
				// Past this the fp32 exp(-g) overflows and the bound below
				// no longer holds; the fixture must not get here.
				c.see(float32(math.NaN()), 0, 0, "h[%d,%d] (gate outside the bound's domain)", t, j)
				continue
			}
			ref := gate / (1 + math.Exp(-gate)) * up
			c.see(st["h"][t*F+j], ref, 10*u*math.Abs(ref)+tiny+elem(ref), "h[%d,%d]", t, j)
		}
	}
	checkLinear(r.stage("down"), in.wd, D, F, st["h"], st["dn"], rows)
	checkAdd(r.stage("residual2"), D, st["x1"], st["dn"], st["x2"], rows)
	checkRMSNorm(r.stage("rmsnormf"), D, st["x2"], in.wnf, st["xnf"], rows)
	checkLinear(r.stage("lm_head"), in.wlm, s.V, D, st["xnf"], st["logits"], rows)
	return r
}

// checkLinear checks y = W·x for the columns in rows, where W is m×k
// column-major (W[i*m+j] is row j, column i) and x and y hold one column per
// token. The bound is the standard one for a k-term fp32 dot product in any
// summation order, plus one rounding for a product that is not fused.
func checkLinear(c *stageCheck, w []float32, m, k int, x, y []float32, rows []int) {
	g := gamma(k+1, u) + gamma(k+1, u64)
	ref := make([]float64, m)
	abs := make([]float64, m)
	for _, t := range rows {
		clear(ref)
		clear(abs)
		for i, xi := range x[t*k : (t+1)*k] {
			xv := float64(xi)
			for j, wv := range w[i*m : (i+1)*m] {
				p := float64(wv) * xv
				ref[j] += p
				abs[j] += math.Abs(p)
			}
		}
		for j := range m {
			c.see(y[t*m+j], ref[j], g*abs[j], "[%d,%d]", t, j)
		}
	}
}

// checkRMSNorm checks y = x·w/sqrt(mean(x²)+eps) row by row. The sum of
// squares has relative error γ(D); the divide, the add of eps, the rsqrtf (2
// ulp) and the two multiplies contribute at most 8u more.
func checkRMSNorm(c *stageCheck, D int, x, w, y []float32, rows []int) {
	for _, t := range rows {
		row := x[t*D : (t+1)*D]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1 / math.Sqrt(ss/float64(D)+float64(decoderEps))
		for d, v := range row {
			ref := float64(v) * inv * float64(w[d])
			c.see(y[t*D+d], ref, (gamma(D, u)+8*u)*math.Abs(ref)+16*u64*math.Abs(ref), "[%d,%d]", t, d)
		}
	}
}

// checkAdd checks out = a + b, which IEEE addition rounds correctly.
func checkAdd(c *stageCheck, D int, a, b, out []float32, rows []int) {
	for _, t := range rows {
		for d := range D {
			i := t*D + d
			ref := float64(a[i]) + float64(b[i])
			c.see(out[i], ref, u*math.Abs(ref)+16*u64*math.Abs(ref), "[%d,%d]", t, d)
		}
	}
}

// hostForward evaluates the forward in float32 on the host, in one
// summation order the device need not share. It is the oracle's own control:
// a correct fp32 evaluation must pass every bound, and a planted defect must
// fail the stage it is planted in.
func hostForward(s decoderShape, in *decoderInputs, causal bool) map[string][]float32 {
	D, H, F, hd, S, T, L := s.D, s.H, s.F, s.HD(), s.S(), s.T, s.L
	half := hd / 2
	st := make(map[string][]float32)
	for name, n := range decoderBuffers(s) {
		st[name] = make([]float32, n)
	}
	for h := range H {
		for p := range L {
			copy(st["kc"][(h*S+p)*hd:][:hd], in.kPrefix[(h*L+p)*hd:][:hd])
			copy(st["vc"][(h*S+p)*hd:][:hd], in.vPrefix[(h*L+p)*hd:][:hd])
		}
	}
	linear := func(w []float32, m, k int, x, y []float32) {
		for t := range T {
			for j := range m {
				var acc float32
				for i := range k {
					acc = float32(acc + float32(w[i*m+j]*x[t*k+i]))
				}
				y[t*m+j] = acc
			}
		}
	}
	rmsnorm := func(x, w, y []float32) {
		for t := range T {
			var ss float32
			for _, v := range x[t*D : (t+1)*D] {
				ss = float32(ss + float32(v*v))
			}
			r := float32(1 / math.Sqrt(float64(float32(ss/float32(D))+decoderEps)))
			for d := range D {
				y[t*D+d] = float32(float32(x[t*D+d]*r) * w[d])
			}
		}
	}
	exp32 := func(x float32) float32 { return float32(math.Exp(float64(x))) }

	for t := range T {
		copy(st["x0"][t*D:(t+1)*D], in.emb[int(in.ids[t])*D:])
	}
	rmsnorm(st["x0"], in.wn1, st["xn1"])
	linear(in.wqkv, 3*D, D, st["xn1"], st["qkv"])
	for t := range T {
		pos := L + t
		row := st["qkv"][t*3*D:]
		for h := range H {
			for i := range half {
				th := float32(pos) * in.invFreq[i]
				cs, sn := float32(math.Cos(float64(th))), float32(math.Sin(float64(th)))
				rot := func(a, b float32, out []float32) {
					out[i] = float32(float32(a*cs) - float32(b*sn))
					out[i+half] = float32(float32(b*cs) + float32(a*sn))
				}
				rot(row[h*hd+i], row[h*hd+i+half], st["q"][(h*T+t)*hd:])
				rot(row[D+h*hd+i], row[D+h*hd+i+half], st["kc"][(h*S+pos)*hd:])
				st["vc"][(h*S+pos)*hd+i] = row[2*D+h*hd+i]
				st["vc"][(h*S+pos)*hd+i+half] = row[2*D+h*hd+i+half]
			}
		}
	}
	for h := range H {
		for t := range T {
			for p := range S {
				var acc float32
				for d := range hd {
					acc = float32(acc + float32(st["kc"][(h*S+p)*hd+d]*st["q"][(h*T+t)*hd+d]))
				}
				st["scores"][(h*T+t)*S+p] = acc
			}
			row := st["scores"][(h*T+t)*S:][:S]
			out := st["probs"][(h*T+t)*S:][:S]
			allowed := S
			if causal {
				allowed = L + t + 1
			}
			m := float32(math.Inf(-1))
			for p := range allowed {
				m = max(m, float32(in.scale*row[p]))
			}
			var sum float32
			for p := range allowed {
				sum = float32(sum + exp32(float32(float32(in.scale*row[p])-m)))
			}
			for p := range S {
				if p < allowed {
					out[p] = float32(exp32(float32(float32(in.scale*row[p])-m)) / sum)
				}
			}
			for d := range hd {
				var acc float32
				for p := range S {
					acc = float32(acc + float32(st["vc"][(h*S+p)*hd+d]*out[p]))
				}
				st["o"][(h*T+t)*hd+d] = acc
			}
		}
	}
	for t := range T {
		for h := range H {
			copy(st["attn"][t*D+h*hd:][:hd], st["o"][(h*T+t)*hd:][:hd])
		}
	}
	linear(in.wo, D, D, st["attn"], st["ao"])
	for i := range st["x1"] {
		st["x1"][i] = float32(st["x0"][i] + st["ao"][i])
	}
	rmsnorm(st["x1"], in.wn2, st["xn2"])
	linear(in.wgu, 2*F, D, st["xn2"], st["gu"])
	for t := range T {
		for j := range F {
			g, up := st["gu"][t*2*F+j], st["gu"][t*2*F+F+j]
			st["h"][t*F+j] = float32(float32(g/float32(1+exp32(-g))) * up)
		}
	}
	linear(in.wd, D, F, st["h"], st["dn"])
	for i := range st["x2"] {
		st["x2"][i] = float32(st["x1"][i] + st["dn"][i])
	}
	rmsnorm(st["x2"], in.wnf, st["xnf"])
	linear(in.wlm, s.V, D, st["xnf"], st["logits"])
	return st
}

// TestDecoderOracle is the oracle's host control. A float32 evaluation passes
// every stage at both correctness shapes; each planted defect fails exactly
// the stage it is planted in, because every stage is checked from its own
// inputs.
func TestDecoderOracle(t *testing.T) {
	for _, s := range decoderCorrectness {
		t.Run(s.Name, func(t *testing.T) {
			in := newDecoderInputs(s, 1)
			st := hostForward(s, in, true)
			rep := checkDecoder(s, in, st, s.rows(true))
			if f := rep.failed(); len(f) != 0 {
				t.Fatalf("float32 host forward fails %v:\n%s", f, rep)
			}
			t.Logf("float32 host forward:\n%s", rep)

			mutants := []struct {
				name  string
				plant func(map[string][]float32)
				want  []string
			}{
				{"causal mask removed", nil, []string{"softmax"}},
				{"logit moved by 1e-3", func(st map[string][]float32) { st["logits"][7] += 1e-3 }, []string{"lm_head"}},
				// x0 is also read by rmsnorm1 and residual1, which check it
				// as their input and fail with it.
				{"wrong embedding row", func(st map[string][]float32) { st["x0"][3]++ }, []string{"embed", "rmsnorm1", "residual1"}},
				// A moved key fails the stage that wrote it and the stage
				// that read it: rope_kv (a new position, or the uploaded
				// prefix when L > 0) and scores.
				{"moved key", func(st map[string][]float32) { st["kc"][1]++ }, []string{"rope_kv", "scores"}},
			}
			for _, m := range mutants {
				var mst map[string][]float32
				if m.plant == nil {
					mst = hostForward(s, in, false)
				} else {
					mst = make(map[string][]float32)
					for k, v := range st {
						mst[k] = slices.Clone(v)
					}
					m.plant(mst)
				}
				got := checkDecoder(s, in, mst, s.rows(true)).failed()
				if !slices.Equal(got, m.want) {
					t.Errorf("%s: failed stages %v, want %v", m.name, got, m.want)
				}
			}
		})
	}
}
