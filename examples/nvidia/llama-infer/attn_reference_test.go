package main

import (
	"math"
	"reflect"
	"testing"
)

// attnDefect names a known way to get the attention half wrong. The oracle
// computes each defect so the test can show the comparison detects it.
type attnDefect int

const (
	attnCorrect  attnDefect = iota
	attnNoRsqrt             // normalization skips rsqrt(mean(x²)+eps)
	attnVTerm               // forward.cu FusedRMSNormQKV value term
	attnRoPEPrev            // rotary angle uses pos-1
	attnHeadSwap            // query head h reads kv head h%kvHeads
	attnRowNext             // key/value written to row pos+1
)

var attnDefects = []struct {
	defect attnDefect
	name   string
}{
	{attnNoRsqrt, "missing rsqrt"}, {attnVTerm, "forward.cu value term"},
	{attnRoPEPrev, "rope at pos-1"}, {attnHeadSwap, "swapped kv head map"},
	{attnRowNext, "cache row pos+1"},
}

// attnOracle composes the existing CPU primitives (rmsNorm, matmulVec,
// applyRoPE) and a float64 softmax over a model.go KV cache, independently of
// attnPlan.reference. It returns the residual output and the key and value
// rows written for pos.
func attnOracle(p attnPlan, b attnBuffers, kv *KVCacheState, pos int, defect attnDefect) (out, k, v []float32) {
	normalized := rmsNorm(b.x, b.norm, p.epsilon)
	if defect == attnNoRsqrt {
		for i := range normalized {
			normalized[i] = b.x[i] * b.norm[i]
		}
	}
	q := matmulVec(normalized, b.wq, p.hidden, p.qDim)
	k = matmulVec(normalized, b.wk, p.hidden, p.kvDim)
	v = matmulVec(normalized, b.wv, p.hidden, p.kvDim)
	if defect == attnVTerm {
		// forward.cu:309: sumV += n.x*wv.x + n.y*wv.y + n.z*wk.w*wv.w,
		// over complete float4 groups only.
		for o := range v {
			var sum float32
			for i := 0; i+3 < p.hidden; i += 4 {
				wk, wv := b.wk[o*p.hidden:], b.wv[o*p.hidden:]
				sum += normalized[i]*wv[i] + normalized[i+1]*wv[i+1] + normalized[i+2]*wk[i+3]*wv[i+3]
			}
			v[o] = sum
		}
	}
	for _, a := range []struct{ dst, bias []float32 }{{q, b.qBias}, {k, b.kBias}, {v, b.vBias}} {
		for i := range a.bias {
			a.dst[i] += a.bias[i]
		}
	}
	ropePos := pos
	if defect == attnRoPEPrev {
		ropePos = pos - 1
	}
	applyRoPE(q, ropePos, p.qHeads, p.headDim, p.theta)
	applyRoPE(k, ropePos, p.kvHeads, p.headDim, p.theta)
	row := pos
	if defect == attnRowNext {
		row = pos + 1
	}
	if row < p.maxSeq {
		copy(kv.K[0][row], k)
		copy(kv.V[0][row], v)
	}

	attn := make([]float32, p.qDim)
	for h := 0; h < p.qHeads; h++ {
		kvh := h / (p.qHeads / p.kvHeads)
		if defect == attnHeadSwap {
			kvh = h % p.kvHeads
		}
		weights := make([]float64, pos+1)
		peak := math.Inf(-1)
		for t := range weights {
			var dot float64
			for d := 0; d < p.headDim; d++ {
				dot += float64(q[h*p.headDim+d]) * float64(kv.K[0][t][kvh*p.headDim+d])
			}
			weights[t] = dot / math.Sqrt(float64(p.headDim))
			peak = math.Max(peak, weights[t])
		}
		var total float64
		for t := range weights {
			weights[t] = math.Exp(weights[t] - peak)
			total += weights[t]
		}
		for d := 0; d < p.headDim; d++ {
			var acc float64
			for t, w := range weights {
				acc += w * float64(kv.V[0][t][kvh*p.headDim+d])
			}
			attn[h*p.headDim+d] = float32(acc / total)
		}
	}
	out = matmulVec(attn, b.wo, p.qDim, p.hidden)
	for i := range out {
		out[i] += b.x[i]
	}
	return out, k, v
}

// attnTolerance bounds the float32 reference against the oracle, whose
// attention accumulates in float64.
func attnTolerance(want float32) float64 { return 1e-5 * (1 + math.Abs(float64(want))) }

// sameBits reports whether a and b hold identical float32 bit patterns, so
// NaN sentinels compare equal to themselves.
func sameBits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// attnMismatch reports the first index where got differs from want beyond
// attnTolerance, or -1.
func attnMismatch(got, want []float32) int {
	for i := range want {
		if !(math.Abs(float64(got[i]-want[i])) <= attnTolerance(want[i])) {
			return i
		}
	}
	return -1
}

func TestAttnReferenceOracle(t *testing.T) {
	const steps = 6
	sentinel := math.Float32frombits(0x7fc0beef)
	for _, tt := range []struct {
		name                             string
		hidden, qHeads, kvHeads, headDim int
		bias                             bool
		gain                             float32 // scales wq and wk to sharpen softmax
	}{
		{"mha", 8, 2, 2, 4, false, 1},
		{"gqa2 bias", 8, 4, 2, 2, true, 1},
		{"gqa4 wide query", 12, 4, 1, 4, false, 1},
		{"odd hidden", 7, 2, 1, 6, true, 1},
		{"small mha", 5, 2, 2, 2, false, 1},
		{"larger gqa", 64, 8, 2, 8, true, 1},
		{"sharp gqa", 64, 8, 2, 8, true, 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h, maxSeq := tt.hidden, steps+1
			p, err := newAttnPlan(h, tt.qHeads, tt.kvHeads, tt.headDim, maxSeq, 1e-5, 1e4, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			values := func(size, salt int) []float32 {
				v := make([]float32, size)
				for i := range v {
					v[i] = float32((i*7+salt)%23-11) / 31
				}
				return v
			}
			b := attnBuffers{x: values(h, 1), norm: values(h, 3), wq: values(p.qDim*h, 5),
				wk: values(p.kvDim*h, 9), wv: values(p.kvDim*h, 13), wo: values(h*p.qDim, 17)}
			for _, w := range [][]float32{b.wq, b.wk} {
				for i := range w {
					w[i] *= tt.gain
				}
			}
			if tt.bias {
				b.qBias, b.kBias, b.vBias = values(p.qDim, 19), values(p.kvDim, 2), values(p.kvDim, 4)
			}
			clone := func(s []float32) []float32 {
				if s == nil {
					return nil
				}
				return append([]float32(nil), s...)
			}
			before := attnBuffers{x: clone(b.x), norm: clone(b.norm), wq: clone(b.wq), wk: clone(b.wk),
				wv: clone(b.wv), wo: clone(b.wo), qBias: clone(b.qBias), kBias: clone(b.kBias), vBias: clone(b.vBias)}

			c := attnCache{k: make([]float32, maxSeq*p.kvDim), v: make([]float32, maxSeq*p.kvDim)}
			for i := range c.k {
				c.k[i], c.v[i] = sentinel, sentinel
			}
			oracle := NewKVCacheState(1, maxSeq, p.kvDim)
			detected := make(map[attnDefect]int)
			defective := make(map[attnDefect]*KVCacheState)
			for _, d := range attnDefects {
				defective[d.defect] = NewKVCacheState(1, maxSeq, p.kvDim)
			}
			for pos := 0; pos < steps; pos++ {
				prevK, prevV := clone(c.k), clone(c.v)
				// Vary the token so later rows differ from earlier ones.
				b.x[pos%h] += 0.25
				before.x = clone(b.x)

				got, err := p.reference(b, c, pos)
				if err != nil {
					t.Fatal(err)
				}
				want, wantK, wantV := attnOracle(p, b, oracle, pos, attnCorrect)
				if i := attnMismatch(got, want); i >= 0 {
					t.Fatalf("pos %d output %d: got %g want %g", pos, i, got[i], want[i])
				}
				row := func(s []float32, r int) []float32 { return s[r*p.kvDim : (r+1)*p.kvDim] }
				if i := attnMismatch(row(c.k, pos), wantK); i >= 0 {
					t.Fatalf("pos %d key %d: got %g want %g", pos, i, row(c.k, pos)[i], wantK[i])
				}
				if i := attnMismatch(row(c.v, pos), wantV); i >= 0 {
					t.Fatalf("pos %d value %d: got %g want %g", pos, i, row(c.v, pos)[i], wantV[i])
				}
				// Only row pos changes; later rows keep the sentinel bits.
				for r := 0; r < maxSeq; r++ {
					if r == pos {
						continue
					}
					if !sameBits(row(c.k, r), row(prevK, r)) || !sameBits(row(c.v, r), row(prevV, r)) {
						t.Fatalf("pos %d changed cache row %d", pos, r)
					}
				}
				if !reflect.DeepEqual(b, before) {
					t.Fatalf("pos %d reference mutated input", pos)
				}

				for _, d := range attnDefects {
					if d.defect == attnHeadSwap && (tt.kvHeads == 1 || tt.kvHeads == tt.qHeads) {
						continue // h/group == h%kvHeads for every head
					}
					bad, badK, badV := attnOracle(p, b, defective[d.defect], pos, d.defect)
					if attnMismatch(got, bad) < 0 && attnMismatch(row(c.k, pos), badK) < 0 && attnMismatch(row(c.v, pos), badV) < 0 {
						t.Errorf("pos %d: %s not detected", pos, d.name)
						continue
					}
					detected[d.defect]++
				}
			}

			for _, d := range attnDefects {
				skippable := d.defect == attnHeadSwap && (tt.kvHeads == 1 || tt.kvHeads == tt.qHeads)
				if !skippable && detected[d.defect] != steps {
					t.Errorf("%s detected in %d of %d steps", d.name, detected[d.defect], steps)
				}
			}

			// Two outputs are separately owned and independent of the cache.
			first, err := p.reference(b, c, steps-1)
			if err != nil {
				t.Fatal(err)
			}
			second, err := p.reference(b, c, steps-1)
			if err != nil {
				t.Fatal(err)
			}
			firstSnap, secondSnap, cacheSnap := clone(first), clone(second), append(clone(c.k), c.v...)
			first[0] += 99
			if !reflect.DeepEqual(second, secondSnap) || !sameBits(append(clone(c.k), c.v...), cacheSnap) {
				t.Fatal("first output aliases second output or cache")
			}
			second[0] -= 77
			if first[0] != firstSnap[0]+99 || !sameBits(append(clone(c.k), c.v...), cacheSnap) {
				t.Fatal("second output aliases first output or cache")
			}
		})
	}
}
