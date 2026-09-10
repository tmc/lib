package gomlx

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

// FusedGelu implements compute.FusedGelu by composing StableHLO operations.
//
// It is named for the compute contract. This implementation does not claim a
// native fused kernel.
func (f *function) FusedGelu(v compute.Value, exact bool) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(v)
	if err != nil {
		return nil, fmt.Errorf("fused gelu input: %w", err)
	}
	if err := fusedUnaryShape(s); err != nil {
		return nil, err
	}
	w, dtype, err := f.fusedUnaryCompute(v, s.DType)
	if err != nil {
		return nil, err
	}
	input := w
	half, err := f.denseScalar(dtype, 0.5)
	if err != nil {
		return nil, err
	}
	one, err := f.denseScalar(dtype, 1)
	if err != nil {
		return nil, err
	}
	if exact {
		sqrt2, e := f.denseScalar(dtype, math.Sqrt2)
		if e != nil {
			return nil, e
		}
		w, e = f.Div(w, sqrt2)
		if e != nil {
			return nil, e
		}
		w, e = f.Erf(w)
		if e != nil {
			return nil, e
		}
	} else {
		c, e := f.denseScalar(dtype, math.Sqrt(2/math.Pi))
		if e != nil {
			return nil, e
		}
		k, e := f.denseScalar(dtype, 0.044715)
		if e != nil {
			return nil, e
		}
		cube, e := f.Mul(w, w)
		if e != nil {
			return nil, e
		}
		cube, e = f.Mul(cube, w)
		if e != nil {
			return nil, e
		}
		cube, e = f.Mul(cube, k)
		if e != nil {
			return nil, e
		}
		w, e = f.Add(w, cube)
		if e != nil {
			return nil, e
		}
		w, e = f.Mul(w, c)
		if e != nil {
			return nil, e
		}
		w, e = f.Tanh(w)
		if e != nil {
			return nil, e
		}
	}
	w, err = f.Add(w, one)
	if err != nil {
		return nil, err
	}
	w, err = f.Mul(w, half)
	if err != nil {
		return nil, err
	}
	w, err = f.Mul(w, input)
	if err != nil {
		return nil, err
	}
	return f.fusedUnaryResult(w, s.DType)
}

// FusedSoftmax implements compute.FusedSoftmax by composing a stable softmax.
// The axis must be non-negative and name one input dimension.
func (f *function) FusedSoftmax(v compute.Value, axis int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(v)
	if err != nil {
		return nil, fmt.Errorf("fused softmax input: %w", err)
	}
	if err := fusedUnaryShape(s); err != nil {
		return nil, err
	}
	if axis < 0 || axis >= len(s.Dimensions) {
		return nil, fmt.Errorf("fused softmax: axis %d out of range for rank %d", axis, len(s.Dimensions))
	}
	w, dtype, err := f.fusedUnaryCompute(v, s.DType)
	if err != nil {
		return nil, err
	}
	ws := s.Clone()
	ws.DType = dtype
	max, err := f.ReduceMax(w, axis)
	if err != nil {
		return nil, err
	}
	max, err = f.BroadcastInDim(max, ws, fusedUnaryAxes(len(s.Dimensions), axis))
	if err != nil {
		return nil, err
	}
	w, err = f.Sub(w, max)
	if err != nil {
		return nil, err
	}
	w, err = f.Exp(w)
	if err != nil {
		return nil, err
	}
	sum, err := f.ReduceSum(w, axis)
	if err != nil {
		return nil, err
	}
	sum, err = f.BroadcastInDim(sum, ws, fusedUnaryAxes(len(s.Dimensions), axis))
	if err != nil {
		return nil, err
	}
	w, err = f.Div(w, sum)
	if err != nil {
		return nil, err
	}
	return f.fusedUnaryResult(w, s.DType)
}

func fusedUnaryShape(s shapes.Shape) error {
	if !denseDType(s.DType) {
		return unsupported
	}
	for _, d := range s.Dimensions {
		if d < 0 {
			return unsupported
		}
	}
	return nil
}

func fusedUnaryAxes(rank, axis int) []int {
	axes := make([]int, 0, rank-1)
	for i := range rank {
		if i != axis {
			axes = append(axes, i)
		}
	}
	return axes
}

// fusedUnaryCompute promotes reduced-precision values to f32. In particular,
// this keeps softmax's maximum, exponentials, and sum from being rounded at
// every intermediate operation.
func (f *function) fusedUnaryCompute(v compute.Value, dtype dtypes.DType) (compute.Value, dtypes.DType, error) {
	if dtype != dtypes.Float16 && dtype != dtypes.BFloat16 {
		return v, dtype, nil
	}
	w, err := f.ConvertDType(v, dtypes.Float32)
	if err != nil {
		return nil, 0, err
	}
	return w, dtypes.Float32, nil
}

func (f *function) fusedUnaryResult(v compute.Value, dtype dtypes.DType) (compute.Value, error) {
	if dtype != dtypes.Float16 && dtype != dtypes.BFloat16 {
		return v, nil
	}
	return f.ConvertDType(v, dtype)
}
