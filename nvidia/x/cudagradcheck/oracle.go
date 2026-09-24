package cudagradcheck

import "math"

// dropped names one backward term that keep zeroes. Only the mutation tests
// set it, to show that the tolerance catches a missing term.
var dropped string

// keep returns 0 if term is dropped and 1 otherwise.
func keep(term string) float64 {
	if term == dropped {
		return 0
	}
	return 1
}

// LayerNorm normalizes each row of n elements of x over the row and returns
// y = xhat*w + b, where xhat = (x-mean)*rstd and rstd = 1/sqrt(var+eps) with
// the biased variance. b may be nil for a LayerNorm without bias. mean and
// rstd hold one value per row, for [LayerNormBackward].
func LayerNorm(x, w, b []float64, n int, eps float64) (y, mean, rstd []float64) {
	rows := len(x) / n
	y = make([]float64, len(x))
	mean = make([]float64, rows)
	rstd = make([]float64, rows)
	for r := range rows {
		row := x[r*n : (r+1)*n]
		var mu float64
		for _, v := range row {
			mu += v
		}
		mu /= float64(n)
		var sq float64
		for _, v := range row {
			sq += (v - mu) * (v - mu)
		}
		rs := 1 / math.Sqrt(sq/float64(n)+eps)
		mean[r], rstd[r] = mu, rs
		for i, v := range row {
			o := (v - mu) * rs * w[i]
			if b != nil {
				o += b[i]
			}
			y[r*n+i] = o
		}
	}
	return y, mean, rstd
}

// LayerNormBackward returns the gradients of [LayerNorm] with respect to x,
// w and b given the upstream gradient dy and the forward's mean and rstd.
// db is the column sum of dy; a LayerNorm without bias ignores it.
func LayerNormBackward(dy, x, w, mean, rstd []float64, n int) (dx, dw, db []float64) {
	rows := len(x) / n
	dx = make([]float64, len(x))
	dw = make([]float64, n)
	db = make([]float64, n)
	xhat := make([]float64, n)
	for r := range rows {
		var sumG, sumGX float64
		for i := range n {
			xhat[i] = (x[r*n+i] - mean[r]) * rstd[r]
			g := dy[r*n+i] * w[i]
			sumG += g
			sumGX += g * xhat[i]
			dw[i] += dy[r*n+i] * xhat[i]
			db[i] += dy[r*n+i]
		}
		mg := sumG / float64(n) * keep("layernorm.mean")
		mgx := sumGX / float64(n) * keep("layernorm.var")
		for i := range n {
			g := dy[r*n+i] * w[i]
			dx[r*n+i] = rstd[r] * (g - mg - xhat[i]*mgx)
		}
	}
	return dx, dw, db
}

// Softmax returns the softmax of each row of n elements of x.
func Softmax(x []float64, n int) []float64 {
	y := make([]float64, len(x))
	for r := range len(x) / n {
		row := x[r*n : (r+1)*n]
		m := math.Inf(-1)
		for _, v := range row {
			m = max(m, v)
		}
		var s float64
		for i, v := range row {
			y[r*n+i] = math.Exp(v - m)
			s += y[r*n+i]
		}
		for i := range row {
			y[r*n+i] /= s
		}
	}
	return y
}

// SoftmaxBackward returns the gradient of [Softmax] with respect to its input
// given its output y and the upstream gradient dy: y*(dy - sum(dy*y)) per row.
func SoftmaxBackward(y, dy []float64, n int) []float64 {
	dx := make([]float64, len(y))
	for r := range len(y) / n {
		var s float64
		for i := range n {
			s += dy[r*n+i] * y[r*n+i]
		}
		s *= keep("softmax.sum")
		for i := range n {
			dx[r*n+i] = y[r*n+i] * (dy[r*n+i] - s)
		}
	}
	return dx
}

func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
}

func geluGrad(x float64) float64 {
	pdf := math.Exp(-x*x/2) / math.Sqrt(2*math.Pi)
	return 0.5*(1+math.Erf(x/math.Sqrt2)) + x*pdf*keep("gelu.pdf")
}

// GELU returns x*Φ(x) elementwise, the exact erf form rather than the tanh
// approximation.
func GELU(x []float64) []float64 {
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = gelu(v)
	}
	return y
}

// GELUBackward returns the gradient of [GELU] given its input x and the
// upstream gradient dy.
func GELUBackward(x, dy []float64) []float64 {
	dx := make([]float64, len(x))
	for i, v := range x {
		dx[i] = dy[i] * geluGrad(v)
	}
	return dx
}

// GeGLU splits each row of 2n elements of x into an input half a (the first
// n) and a gate half g (the second n), as torch's chunk(2, dim=-1) does, and
// returns the rows GELU(a)*g of n elements.
func GeGLU(x []float64, n int) []float64 {
	rows := len(x) / (2 * n)
	y := make([]float64, rows*n)
	for r := range rows {
		a, g := x[2*r*n:2*r*n+n], x[2*r*n+n:2*(r+1)*n]
		for i := range n {
			y[r*n+i] = gelu(a[i]) * g[i]
		}
	}
	return y
}

// GeGLUBackward returns the gradient of [GeGLU] with respect to x, in the
// same layout as x, given the upstream gradient dy of rows of n elements.
func GeGLUBackward(x, dy []float64, n int) []float64 {
	rows := len(x) / (2 * n)
	dx := make([]float64, len(x))
	for r := range rows {
		for i := range n {
			a, g := x[2*r*n+i], x[2*r*n+n+i]
			d := dy[r*n+i]
			gate := g*keep("geglu.gate") + (1 - keep("geglu.gate"))
			dx[2*r*n+i] = d * gate * geluGrad(a)
			dx[2*r*n+n+i] = d * gelu(a)
		}
	}
	return dx
}

// RoPE applies rotary position embeddings in the rotate-half layout used by
// GPT-NeoX and Hugging Face models: x holds len(pos) tokens of heads heads of
// dim elements, and elements i and i+dim/2 of each head rotate together by
// pos*base^(-2i/dim). dim must be even.
func RoPE(x []float64, pos []int, heads, dim int, base float64) []float64 {
	return rope(x, pos, heads, dim, base, 1)
}

// RoPEBackward returns the gradient of [RoPE] given the upstream gradient dy:
// the rotation by the opposite angle.
func RoPEBackward(dy []float64, pos []int, heads, dim int, base float64) []float64 {
	sign := -1.0
	if dropped == "rope.sign" {
		sign = 1
	}
	return rope(dy, pos, heads, dim, base, sign)
}

func rope(x []float64, pos []int, heads, dim int, base, sign float64) []float64 {
	y := make([]float64, len(x))
	half := dim / 2
	for t, p := range pos {
		for i := range half {
			theta := float64(p) * math.Pow(base, -2*float64(i)/float64(dim))
			s, c := math.Sincos(theta)
			s *= sign
			for h := range heads {
				o := (t*heads + h) * dim
				a, b := x[o+i], x[o+i+half]
				y[o+i] = a*c - b*s
				y[o+i+half] = b*c + a*s
			}
		}
	}
	return y
}
