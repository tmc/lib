package main

// Option scorer and action head. Both run on the host: together they are
// about ten MFLOP per question, spread across the host's cores.

import (
	"math"
	"runtime"
	"sync"
)

// score turns the gathered output rows for one question into per-option logits
// and the action probability. rows holds the K option-marker states followed by
// the CLS state, each of width d.
func (m *model) score(rows []float32, k int) (logits []float32, actProb float32) {
	c, w := m.cfg, m.w
	d := c.D

	mn := make([]float32, k*d)
	layerNorm(mn, rows[:k*d], w.ScLNW, w.ScLNB, k, d, headEps)
	m1 := make([]float32, k*d)
	linear(m1, mn, w.ScW1, w.ScB1, k, d, d)

	logits = make([]float32, k)
	for i := range k {
		row := m1[i*d : (i+1)*d]
		s := w.ScB2[0]
		for j, v := range row {
			s += gelu(v) * w.ScW2[j]
		}
		logits[i] = s
	}

	// Action head: the CLS state plus four summary features of the raw
	// distribution (top probability, its margin, normalized entropy, K/255).
	p := softmax64(logits)
	var t1, t2, ent float64
	for _, v := range p {
		switch {
		case v > t1:
			t1, t2 = v, t1
		case v > t2:
			t2 = v
		}
		ent -= v * math.Log(max(v, 1e-9))
	}
	ent /= math.Log(float64(max(k, 2)))

	in := make([]float32, d+4)
	copy(in, rows[k*d:(k+1)*d])
	in[d], in[d+1] = float32(t1), float32(t1-t2)
	in[d+2], in[d+3] = float32(ent), float32(k)/255.0

	ah := make([]float32, c.ActHidden)
	linear(ah, in, w.ActW1, w.ActB1, 1, d+4, c.ActHidden)
	for i, v := range ah {
		ah[i] = gelu(v)
	}
	al := make([]float32, c.NumAct)
	linear(al, ah, w.ActW2, w.ActB2, 1, c.ActHidden, c.NumAct)

	amx := al[0]
	for _, v := range al[1:] {
		amx = max(amx, v)
	}
	var asum float32
	for _, v := range al {
		asum += float32(math.Exp(float64(v - amx)))
	}
	return logits, float32(math.Exp(float64(al[0]-amx))) / asum
}

// softmax64 is the raw (uncalibrated) distribution over logits, in float64.
func softmax64(logits []float32) []float64 {
	mx := float64(logits[0])
	for _, v := range logits[1:] {
		mx = max(mx, float64(v))
	}
	p, sum := make([]float64, len(logits)), 0.0
	for i, v := range logits {
		p[i] = math.Exp(float64(v) - mx)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

// layerNorm normalizes each of rows rows of width d, accumulating the mean and
// variance in float64 the way laya.c does.
func layerNorm(out, x, w, b []float32, rows, d int, eps float32) {
	for t := range rows {
		xr := x[t*d : (t+1)*d]
		var mu float64
		for _, v := range xr {
			mu += float64(v)
		}
		mu /= float64(d)
		var vr float64
		for _, v := range xr {
			c := float64(v) - mu
			vr += c * c
		}
		inv := float32(1.0 / math.Sqrt(vr/float64(d)+float64(eps)))
		mean := float32(mu)
		o := out[t*d : (t+1)*d]
		for i, v := range xr {
			o[i] = (v - mean) * inv * w[i]
			if b != nil {
				o[i] += b[i]
			}
		}
	}
}

// linear computes C[M,N] = A[M,K] * W[N,K]^T + bias[N], with W in the [out, in]
// layout safetensors stores.
//
// Each output is one dot product summed in a fixed order, the order laya.c
// uses, so splitting the outputs across goroutines changes when they are
// computed and nothing else. The scorer's d by d product is the only one
// large enough to be worth it: at d=1024 and twenty option rows it is
// about 10 ms on one core.
func linear(c, a, w, bias []float32, M, K, N int) {
	dot := func(lo, hi int) {
		for i := range M {
			row := a[i*K : (i+1)*K]
			for j := lo; j < hi; j++ {
				wr := w[j*K : (j+1)*K]
				var s float32
				for u, v := range row {
					s += v * wr[u]
				}
				if bias != nil {
					s += bias[j]
				}
				c[i*N+j] = s
			}
		}
	}
	n := min(runtime.GOMAXPROCS(0), M*N*K>>18, N)
	if n <= 1 {
		dot(0, N)
		return
	}
	var wg sync.WaitGroup
	for p := range n {
		wg.Go(func() { dot(p*N/n, (p+1)*N/n) })
	}
	wg.Wait()
}

func gelu(x float32) float32 {
	return 0.5 * x * (1.0 + float32(math.Erf(float64(x)*0.70710678118654752)))
}
