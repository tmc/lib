package cudaref

import (
	"fmt"
	"math"
)

// WarpReduceSum computes the collective sum across a 32-thread warp.
func WarpReduceSum(vals [32]float32) float32 {
	var sum float32
	for _, v := range vals {
		sum += v
	}
	return sum
}

// BlockReduceSum computes the collective sum across an entire thread block.
func BlockReduceSum(vals []float32) float32 {
	var sum float32
	for _, v := range vals {
		sum += v
	}
	return sum
}

// MatMul computes standard row-major matrix multiplication C = A x B on CPU.
// A has shape (M x K), B has shape (K x N), returning C of shape (M x N).
func MatMul(a, b []float32, M, N, K int) ([]float32, error) {
	if M < 0 || N < 0 || K < 0 {
		return nil, fmt.Errorf("cudaref: negative MatMul dimension (%d x %d, %d x %d)", M, K, K, N)
	}

	maxInt := int(^uint(0) >> 1)
	product := func(name string, x, y int) (int, error) {
		if x != 0 && y > maxInt/x {
			return 0, fmt.Errorf("cudaref: %s element count overflows int", name)
		}
		return x * y, nil
	}
	inputElements, err := product("MatMul A", M, K)
	if err != nil {
		return nil, err
	}
	rightElements, err := product("MatMul B", K, N)
	if err != nil {
		return nil, err
	}
	outputElements, err := product("MatMul output", M, N)
	if err != nil {
		return nil, err
	}
	for _, buffer := range []struct {
		name     string
		elements int
	}{
		{"MatMul A", inputElements},
		{"MatMul B", rightElements},
		{"MatMul output", outputElements},
	} {
		if buffer.elements > maxInt/4 {
			return nil, fmt.Errorf("cudaref: %s float32 byte size overflows int", buffer.name)
		}
	}

	if len(a) != inputElements || len(b) != rightElements {
		return nil, fmt.Errorf("cudaref: dimension mismatch in MatMul (%dx%d, %dx%d)", M, K, K, N)
	}

	c := make([]float32, outputElements)
	if outputElements == 0 {
		return c, nil
	}
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var acc float64
			for k := 0; k < K; k++ {
				acc += float64(a[i*K+k]) * float64(b[k*N+j])
			}
			c[i*N+j] = float32(acc)
		}
	}
	return c, nil
}

// Softmax computes the numerically stable softmax over a 1D vector on CPU.
func Softmax(src []float32) []float32 {
	if len(src) == 0 {
		return nil
	}

	maxVal := src[0]
	for _, v := range src {
		if math.IsNaN(float64(v)) {
			dst := make([]float32, len(src))
			for i := range dst {
				dst[i] = float32(math.NaN())
			}
			return dst
		}
		if v > maxVal {
			maxVal = v
		}
	}

	dst := make([]float32, len(src))
	var sumExp float64
	for i, v := range src {
		e := math.Exp(float64(v - maxVal))
		dst[i] = float32(e)
		sumExp += e
	}

	for i := range dst {
		dst[i] = float32(float64(dst[i]) / sumExp)
	}
	return dst
}

// ElementwiseUnary evaluates unary operations on CPU with explicit IEEE-754 adherence.
// relu_grad chooses the zero subgradient at x == 0; all unary operations,
// including relu_grad and gelu_grad, propagate NaN.
func ElementwiseUnary(src []float32, op string) ([]float32, error) {
	switch op {
	case "relu", "relu_grad", "gelu", "gelu_grad", "silu", "exp", "log", "sqrt", "neg":
	default:
		return nil, fmt.Errorf("cudaref: unknown unary op %s", op)
	}

	dst := make([]float32, len(src))
	for i, v := range src {
		if math.IsNaN(float64(v)) {
			dst[i] = float32(math.NaN())
			continue
		}
		switch op {
		case "relu":
			if v > 0 {
				dst[i] = v
			} else {
				dst[i] = 0.0
			}
		case "relu_grad":
			if v > 0 {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		case "gelu":
			// Exact erf formulation: 0.5 * x * (1 + erf(x / sqrt(2)))
			x := float64(v)
			dst[i] = float32(0.5 * x * (1.0 + math.Erf(x/math.Sqrt2)))
		case "gelu_grad":
			// Exact derivative: Phi(x) + x*phi(x).
			x := float64(v)
			cdf := 0.5 * (1.0 + math.Erf(x/math.Sqrt2))
			pdf := math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
			dst[i] = float32(cdf + x*pdf)
		case "silu":
			x := float64(v)
			dst[i] = float32(x / (1.0 + math.Exp(-x)))
		case "exp":
			dst[i] = float32(math.Exp(float64(v)))
		case "log":
			dst[i] = float32(math.Log(float64(v)))
		case "sqrt":
			dst[i] = float32(math.Sqrt(float64(v)))
		case "neg":
			dst[i] = -v
		}
	}
	return dst, nil
}

// ElementwiseBinary evaluates binary operations on CPU with explicit IEEE-754 adherence.
func ElementwiseBinary(a, b []float32, op string) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("cudaref: length mismatch (%d vs %d)", len(a), len(b))
	}
	switch op {
	case "add", "sub", "mul", "multiply", "div":
	default:
		return nil, fmt.Errorf("cudaref: unknown binary op %s", op)
	}

	dst := make([]float32, len(a))
	for i := range a {
		switch op {
		case "add":
			dst[i] = a[i] + b[i]
		case "sub":
			dst[i] = a[i] - b[i]
		case "mul", "multiply":
			dst[i] = a[i] * b[i]
		case "div":
			dst[i] = a[i] / b[i]
		}
	}
	return dst, nil
}
