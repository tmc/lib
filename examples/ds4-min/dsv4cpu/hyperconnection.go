package dsv4cpu

import (
	"errors"
	"math"
)

// Split computes the pre/post weights and Sinkhorn combine matrix from an HC
// control projection. out has 2*streams+streams*streams elements.
func Split(out, mix, scale, base []float32, streams, iterations int, eps float32) error {
	n := streams
	if n < 1 || n > 16 || iterations < 1 || !(eps > 0) || math.IsInf(float64(eps), 0) || len(scale) != 3 || len(out) != n*(n+2) || len(mix) != len(out) || len(base) != len(out) {
		return errors.New("invalid hyperconnection split shape")
	}
	expf := func(x float32) float32 { return float32(math.Exp(float64(x))) }
	for i := 0; i < n; i++ {
		z := float32(mix[i]*scale[0]) + base[i]
		out[i] = 1/(1+expf(-z)) + eps
		z = float32(mix[n+i]*scale[1]) + base[n+i]
		out[n+i] = 2 / (1 + expf(-z))
	}
	c := make([]float32, n*n)
	for dst := 0; dst < n; dst++ {
		max := float32(math.Inf(-1))
		for src := 0; src < n; src++ {
			i := src + dst*n
			c[i] = float32(mix[2*n+i]*scale[2]) + base[2*n+i]
			if c[i] > max {
				max = c[i]
			}
		}
		var sum float32
		for src := 0; src < n; src++ {
			i := src + dst*n
			c[i] = expf(c[i] - max)
			sum += c[i]
		}
		inv := float32(1) / sum
		for src := 0; src < n; src++ {
			i := src + dst*n
			c[i] = float32(c[i]*inv) + eps
		}
	}
	columns := func() {
		for src := 0; src < n; src++ {
			var sum float32
			for dst := 0; dst < n; dst++ {
				sum += c[src+dst*n]
			}
			inv := float32(1) / (sum + eps)
			for dst := 0; dst < n; dst++ {
				c[src+dst*n] *= inv
			}
		}
	}
	columns()
	for iter := 1; iter < iterations; iter++ {
		for dst := 0; dst < n; dst++ {
			var sum float32
			for src := 0; src < n; src++ {
				sum += c[src+dst*n]
			}
			inv := float32(1) / (sum + eps)
			for src := 0; src < n; src++ {
				c[src+dst*n] *= inv
			}
		}
		columns()
	}
	copy(out[2*n:], c)
	return nil
}

// Collapse reduces stream-major residual streams into a plain embedding.
// dst must not overlap state or weights.
func Collapse(dst, state, weights []float32) error {
	if len(dst) == 0 || len(weights) == 0 || len(weights) > 16 || len(state)/len(weights) != len(dst) || len(state)%len(weights) != 0 {
		return errors.New("invalid hyperconnection collapse shape")
	}
	for d := range dst {
		var acc float32
		for h, w := range weights {
			acc += float32(state[h*len(dst)+d] * w)
		}
		dst[d] = acc
	}
	return nil
}

// Combine forms the next stream-major residual from the sublayer output.
// comb uses upstream's destination-plus-source-times-streams indexing.
// dst must not overlap any input.
func Combine(dst, block, residual, post, comb []float32) error {
	n := len(post)
	if len(block) == 0 || n == 0 || n > 16 || len(comb) != n*n || len(residual)/n != len(block) || len(residual)%n != 0 || len(dst) != len(residual) {
		return errors.New("invalid hyperconnection combine shape")
	}
	for h, p := range post {
		for d, v := range block {
			acc := float32(v * p)
			for src := range post {
				acc += float32(comb[h+src*n] * residual[src*len(block)+d])
			}
			dst[h*len(block)+d] = acc
		}
	}
	return nil
}
