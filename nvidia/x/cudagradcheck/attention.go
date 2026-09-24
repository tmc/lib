package cudagradcheck

import "math"

// A Mask reports whether query i may attend to key j.
type Mask func(i, j int) bool

// Full lets every query attend to every key.
func Full(i, j int) bool { return true }

// Causal lets query i attend to keys 0 through i.
func Causal(i, j int) bool { return j <= i }

// SlidingWindow lets query i attend to keys within w positions on either
// side, as local attention layers such as ModernBERT's do.
func SlidingWindow(w int) Mask {
	return func(i, j int) bool { return i-w <= j && j <= i+w }
}

// Segments packs sequences of the given lengths end to end, as ragged
// batches are packed: a query attends only to keys of its own sequence, and
// m applies to positions counted from the start of that sequence. Positions
// past the sum of the lengths attend to nothing, so their rows are fully
// masked.
func Segments(lengths []int, m Mask) Mask {
	var starts []int
	start := 0
	for _, n := range lengths {
		starts = append(starts, start)
		start += n
	}
	seg := func(i int) int {
		for s := len(starts) - 1; s >= 0; s-- {
			if i >= starts[s] {
				if i < starts[s]+lengths[s] {
					return s
				}
				return -1
			}
		}
		return -1
	}
	return func(i, j int) bool {
		s := seg(i)
		return s >= 0 && s == seg(j) && m(i-starts[s], j-starts[s])
	}
}

// Attention computes single-head attention softmax(scale*q·kᵀ)·v, with q of
// tq rows and k and v of tk rows, all of d columns, and keys hidden by m
// excluded. It also returns the log-sum-exp of each row's scaled scores. A
// row m hides entirely has output zero and log-sum-exp -Inf.
func Attention(q, k, v []float64, tq, tk, d int, scale float64, m Mask) (out, lse []float64) {
	out = make([]float64, tq*d)
	lse = make([]float64, tq)
	for i := range tq {
		p, l := probs(q, k, i, tk, d, scale, m)
		lse[i] = l
		for j, pj := range p {
			for c := range d {
				out[i*d+c] += pj * v[j*d+c]
			}
		}
	}
	return out, lse
}

// probs returns the attention probabilities of query i and its log-sum-exp.
// A fully masked row has all probabilities zero.
func probs(q, k []float64, i, tk, d int, scale float64, m Mask) ([]float64, float64) {
	s := make([]float64, tk)
	mx := math.Inf(-1)
	for j := range tk {
		if !m(i, j) {
			s[j] = math.Inf(-1)
			continue
		}
		var dot float64
		for c := range d {
			dot += q[i*d+c] * k[j*d+c]
		}
		s[j] = scale * dot
		mx = max(mx, s[j])
	}
	if math.IsInf(mx, -1) {
		return make([]float64, tk), mx
	}
	var sum float64
	for j := range s {
		s[j] = math.Exp(s[j] - mx) // exp(-Inf) is 0 for masked keys
		sum += s[j]
	}
	for j := range s {
		s[j] /= sum
	}
	return s, mx + math.Log(sum)
}

// AttentionBackward returns the gradients of [Attention] with respect to q,
// k and v given the upstream gradient dout. It recomputes the forward.
func AttentionBackward(q, k, v, dout []float64, tq, tk, d int, scale float64, m Mask) (dq, dk, dv []float64) {
	dq = make([]float64, tq*d)
	dk = make([]float64, tk*d)
	dv = make([]float64, tk*d)
	if dropped == "attention.mask" {
		m = Full
	}
	for i := range tq {
		p, _ := probs(q, k, i, tk, d, scale, m)
		// delta = rowsum(dout*out) = Σ_j p_j (dout·v_j).
		dp := make([]float64, tk)
		var delta float64
		for j, pj := range p {
			for c := range d {
				dp[j] += dout[i*d+c] * v[j*d+c]
				dv[j*d+c] += pj * dout[i*d+c]
			}
			delta += pj * dp[j]
		}
		delta *= keep("attention.delta")
		kscale := scale
		if dropped == "attention.dk-scale" {
			kscale = 1
		}
		for j, pj := range p {
			ds := pj * (dp[j] - delta)
			for c := range d {
				dq[i*d+c] += scale * ds * k[j*d+c]
				dk[j*d+c] += kscale * ds * q[i*d+c]
			}
		}
	}
	return dq, dk, dv
}
