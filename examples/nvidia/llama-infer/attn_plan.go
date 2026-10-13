package main

import (
	"errors"
	"fmt"
	"math"
)

// attnPlan describes the attention half of one dense, single-token float32
// decoder layer on the host: RMSNorm, Q/K/V projections with optional bias,
// split-half RoPE, KV-cache append, grouped-query causal attention, output
// projection and residual addition. It does not allocate device memory or
// admit a kernel launch. Callers construct a plan with newAttnPlan.
type attnPlan struct {
	hidden, qHeads, kvHeads, headDim, maxSeq int
	epsilon, theta                           float32
	qDim, kvDim, scratchWords                int
}

// attnBuffers contains read-only inputs. Matrices are output-major: wq is
// qDim×hidden, wk and wv are kvDim×hidden, wo is hidden×qDim. Biases are
// either nil or exactly one value per projection output.
type attnBuffers struct {
	x, norm, wq, wk, wv, wo []float32
	qBias, kBias, vBias     []float32
}

// attnCache is caller-owned key/value storage of maxSeq rows of kvDim values,
// row-major by position. The reference writes only row pos of k and v; all
// other rows are read-only. k and v must not overlap each other or any input;
// the reference does not check this.
type attnCache struct {
	k, v []float32
}

// newAttnPlan checks all element and byte counts before any allocation.
// Scratch holds separate normalized, q, k, v, score, attention and projection
// vectors (2*hidden+2*qDim+2*kvDim+maxSeq float32 words). The output and the
// cache are not included in scratchBytes. Negative budgets are invalid.
func newAttnPlan(hidden, qHeads, kvHeads, headDim, maxSeq int, epsilon, theta float32, scratchBytes int) (attnPlan, error) {
	if hidden <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || maxSeq <= 0 {
		return attnPlan{}, errors.New("attention dimensions must be positive")
	}
	if qHeads%kvHeads != 0 {
		return attnPlan{}, fmt.Errorf("attention query heads %d not a multiple of key/value heads %d", qHeads, kvHeads)
	}
	if headDim%2 != 0 {
		return attnPlan{}, fmt.Errorf("attention head dimension %d must be even for rotary embedding", headDim)
	}
	if epsilon <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return attnPlan{}, errors.New("attention epsilon must be finite and positive")
	}
	if theta <= 0 || math.IsNaN(float64(theta)) || math.IsInf(float64(theta), 0) {
		return attnPlan{}, errors.New("attention rope theta must be finite and positive")
	}
	maxWords := int(^uint(0)>>1) / 4
	if qHeads > maxWords/headDim {
		return attnPlan{}, errors.New("attention query size exceeds int range")
	}
	qDim, kvDim := qHeads*headDim, kvHeads*headDim
	if hidden > maxWords/qDim || maxSeq > maxWords/kvDim {
		return attnPlan{}, errors.New("attention matrix or cache byte size exceeds int range")
	}
	// qDim >= kvDim, so 2*hidden+4*qDim+maxSeq bounds the scratch words.
	if hidden > maxWords/8 || qDim > maxWords/8 || maxSeq > maxWords/4 {
		return attnPlan{}, errors.New("attention scratch byte size exceeds int range")
	}
	words := 2*hidden + 2*qDim + 2*kvDim + maxSeq
	if scratchBytes < words*4 {
		return attnPlan{}, fmt.Errorf("attention scratch needs %d bytes, have %d", words*4, scratchBytes)
	}
	return attnPlan{hidden: hidden, qHeads: qHeads, kvHeads: kvHeads, headDim: headDim, maxSeq: maxSeq,
		epsilon: epsilon, theta: theta, qDim: qDim, kvDim: kvDim, scratchWords: words}, nil
}

// stages returns a detached dependency description. Each stage waits for all
// listed predecessors. This is a logical host plan, not a GPU synchronization
// contract.
func (p attnPlan) stages() []attnStage {
	return []attnStage{
		{name: "normalize"},
		{name: "q", after: []string{"normalize"}},
		{name: "k", after: []string{"normalize"}},
		{name: "v", after: []string{"normalize"}},
		{name: "rope", after: []string{"q", "k"}},
		{name: "append", after: []string{"rope", "v"}},
		{name: "attend", after: []string{"append"}},
		{name: "project", after: []string{"attend"}},
		{name: "residual", after: []string{"project"}},
	}
}

type attnStage struct {
	name  string
	after []string
}

func (p attnPlan) validate(b attnBuffers, c attnCache, pos int) error {
	// Reconstruct to refuse zero or internally inconsistent plans before slicing.
	checked, err := newAttnPlan(p.hidden, p.qHeads, p.kvHeads, p.headDim, p.maxSeq, p.epsilon, p.theta, int(^uint(0)>>1))
	if err != nil {
		return err
	}
	if checked != p {
		return errors.New("attention plan storage counts are inconsistent")
	}
	if pos < 0 || pos >= p.maxSeq {
		return fmt.Errorf("attention position %d outside [0, %d)", pos, p.maxSeq)
	}
	for _, v := range []struct {
		name      string
		got, want int
	}{
		{"input", len(b.x), p.hidden}, {"norm", len(b.norm), p.hidden},
		{"wq", len(b.wq), p.qDim * p.hidden}, {"wk", len(b.wk), p.kvDim * p.hidden},
		{"wv", len(b.wv), p.kvDim * p.hidden}, {"wo", len(b.wo), p.hidden * p.qDim},
		{"key cache", len(c.k), p.maxSeq * p.kvDim}, {"value cache", len(c.v), p.maxSeq * p.kvDim},
	} {
		if v.got != v.want {
			return fmt.Errorf("attention %s length is %d, want %d", v.name, v.got, v.want)
		}
	}
	for _, v := range []struct {
		name string
		bias []float32
		want int
	}{
		{"q bias", b.qBias, p.qDim}, {"k bias", b.kBias, p.kvDim}, {"v bias", b.vBias, p.kvDim},
	} {
		if v.bias != nil && len(v.bias) != v.want {
			return fmt.Errorf("attention %s length is %d, want 0 or %d", v.name, len(v.bias), v.want)
		}
	}
	return nil
}

// reference writes the rotated key and the value for pos into row pos of c,
// attends over rows 0 through pos, and returns a new residual output. Inputs
// and all other cache rows are unchanged. Intermediate regions are private to
// this call. This host reference makes no numerical or performance claim
// about a GPU.
func (p attnPlan) reference(b attnBuffers, c attnCache, pos int) ([]float32, error) {
	if err := p.validate(b, c, pos); err != nil {
		return nil, err
	}
	scratch := make([]float32, p.scratchWords)
	off := 0
	take := func(n int) []float32 {
		s := scratch[off : off+n]
		off += n
		return s
	}
	normalized, q, k, v := take(p.hidden), take(p.qDim), take(p.kvDim), take(p.kvDim)
	scores, attn, proj := take(p.maxSeq)[:pos+1], take(p.qDim), take(p.hidden)

	var sum float32
	for _, x := range b.x {
		sum += x * x
	}
	scale := float32(1 / math.Sqrt(float64(sum/float32(p.hidden)+p.epsilon)))
	for i := range normalized {
		normalized[i] = b.x[i] * scale * b.norm[i]
	}
	project := func(dst, w, bias, in []float32) {
		for o := range dst {
			var acc float32
			for i, x := range in {
				acc += x * w[o*len(in)+i]
			}
			if bias != nil {
				acc += bias[o]
			}
			dst[o] = acc
		}
	}
	project(q, b.wq, b.qBias, normalized)
	project(k, b.wk, b.kBias, normalized)
	project(v, b.wv, b.vBias, normalized)

	half := p.headDim / 2
	rotate := func(vec []float32) {
		for h := 0; h < len(vec)/p.headDim; h++ {
			head := vec[h*p.headDim : (h+1)*p.headDim]
			for i := 0; i < half; i++ {
				angle := float64(pos) * math.Pow(float64(p.theta), -float64(2*i)/float64(p.headDim))
				cos, sin := float32(math.Cos(angle)), float32(math.Sin(angle))
				u, w := head[i], head[i+half]
				head[i], head[i+half] = u*cos-w*sin, u*sin+w*cos
			}
		}
	}
	rotate(q)
	rotate(k)
	copy(c.k[pos*p.kvDim:(pos+1)*p.kvDim], k)
	copy(c.v[pos*p.kvDim:(pos+1)*p.kvDim], v)

	group := p.qHeads / p.kvHeads
	inv := float32(1 / math.Sqrt(float64(p.headDim)))
	for h := 0; h < p.qHeads; h++ {
		kvh := h / group
		qh := q[h*p.headDim : (h+1)*p.headDim]
		peak := float32(math.Inf(-1))
		for t := range scores {
			kt := c.k[t*p.kvDim+kvh*p.headDim : t*p.kvDim+(kvh+1)*p.headDim]
			var dot float32
			for d, x := range qh {
				dot += x * kt[d]
			}
			scores[t] = dot * inv
			peak = float32(math.Max(float64(peak), float64(scores[t])))
		}
		var total float32
		for t, s := range scores {
			scores[t] = float32(math.Exp(float64(s - peak)))
			total += scores[t]
		}
		out := attn[h*p.headDim : (h+1)*p.headDim]
		for t, s := range scores {
			vt := c.v[t*p.kvDim+kvh*p.headDim : t*p.kvDim+(kvh+1)*p.headDim]
			for d := range out {
				out[d] += s * vt[d]
			}
		}
		for d := range out {
			out[d] /= total
		}
	}
	project(proj, b.wo, nil, attn)
	out := make([]float32, p.hidden)
	for i := range out {
		out[i] = b.x[i] + proj[i]
	}
	return out, nil
}
