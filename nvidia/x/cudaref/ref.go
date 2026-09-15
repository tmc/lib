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
	if len(a) != M*K || len(b) != K*N {
		return nil, fmt.Errorf("cudaref: dimension mismatch in MatMul (%dx%d, %dx%d)", M, K, K, N)
	}

	c := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var acc float32
			for k := 0; k < K; k++ {
				acc += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = acc
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
		if v > maxVal {
			maxVal = v
		}
	}

	dst := make([]float32, len(src))
	var sumExp float32
	for i, v := range src {
		e := float32(math.Exp(float64(v - maxVal)))
		dst[i] = e
		sumExp += e
	}

	for i := range dst {
		dst[i] /= sumExp
	}
	return dst
}

// ElementwiseUnary evaluates unary operations on CPU.
func ElementwiseUnary(src []float32, op string) ([]float32, error) {
	dst := make([]float32, len(src))
	for i, v := range src {
		switch op {
		case "relu":
			if v > 0 {
				dst[i] = v
			} else {
				dst[i] = 0
			}
		case "exp":
			dst[i] = float32(math.Exp(float64(v)))
		case "sqrt":
			dst[i] = float32(math.Sqrt(float64(v)))
		case "neg":
			dst[i] = -v
		default:
			return nil, fmt.Errorf("cudaref: unknown unary op %s", op)
		}
	}
	return dst, nil
}

// ElementwiseBinary evaluates binary operations on CPU.
func ElementwiseBinary(a, b []float32, op string) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("cudaref: length mismatch (%d vs %d)", len(a), len(b))
	}
	dst := make([]float32, len(a))
	for i := range a {
		switch op {
		case "add":
			dst[i] = a[i] + b[i]
		case "sub":
			dst[i] = a[i] - b[i]
		case "mul":
			dst[i] = a[i] * b[i]
		case "div":
			dst[i] = a[i] / b[i]
		default:
			return nil, fmt.Errorf("cudaref: unknown binary op %s", op)
		}
	}
	return dst, nil
}
