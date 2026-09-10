package gomlx

import (
	"fmt"
	"math"
	"sort"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

// FusedLayerNorm implements compute.FusedLayerNorm by composing StableHLO
// operations. It does not claim a native fused kernel.
func (f *function) FusedLayerNorm(x compute.Value, axes []int, epsilon float64, gamma, beta compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	xs, err := f.Shape(x)
	if err != nil {
		return nil, fmt.Errorf("fused layer norm input: %w", err)
	}
	if !denseDType(xs.DType) {
		return nil, unsupported
	}
	if math.IsNaN(epsilon) || math.IsInf(epsilon, 0) || epsilon < 0 {
		return nil, fmt.Errorf("fused layer norm: epsilon must be finite and non-negative")
	}
	for _, d := range xs.Dimensions {
		if d < 0 {
			return nil, unsupported
		}
	}
	axes, err = layerNormAxes(len(xs.Dimensions), axes)
	if err != nil {
		return nil, err
	}
	if err := f.layerNormScaleShape("gamma", gamma, xs, axes); err != nil {
		return nil, err
	}
	if err := f.layerNormScaleShape("beta", beta, xs, axes); err != nil {
		return nil, err
	}
	count, err := layerNormCount(xs.Dimensions, axes)
	if err != nil {
		return nil, err
	}
	// StableHLO reduction and broadcast dimensions are ordered. Gamma and beta
	// retain the contract's caller-axis order, so their values are transposed
	// before broadcasting when the caller supplied unsorted axes.
	orderedAxes := append([]int(nil), axes...)
	sort.Ints(orderedAxes)

	w, dtype, err := f.layerNormCompute(x, xs.DType)
	if err != nil {
		return nil, err
	}
	ws := xs.Clone()
	ws.DType = dtype
	countValue, err := f.denseScalar(dtype, float64(count))
	if err != nil {
		return nil, err
	}
	mean, err := f.ReduceSum(w, orderedAxes...)
	if err != nil {
		return nil, err
	}
	mean, err = f.Div(mean, countValue)
	if err != nil {
		return nil, err
	}
	mean, err = f.BroadcastInDim(mean, ws, layerNormOuterAxes(len(xs.Dimensions), orderedAxes))
	if err != nil {
		return nil, err
	}
	centered, err := f.Sub(w, mean)
	if err != nil {
		return nil, err
	}
	variance, err := f.Mul(centered, centered)
	if err != nil {
		return nil, err
	}
	variance, err = f.ReduceSum(variance, orderedAxes...)
	if err != nil {
		return nil, err
	}
	variance, err = f.Div(variance, countValue)
	if err != nil {
		return nil, err
	}
	variance, err = f.BroadcastInDim(variance, ws, layerNormOuterAxes(len(xs.Dimensions), orderedAxes))
	if err != nil {
		return nil, err
	}
	eps, err := f.denseScalar(dtype, epsilon)
	if err != nil {
		return nil, err
	}
	variance, err = f.Add(variance, eps)
	if err != nil {
		return nil, err
	}
	variance, err = f.Rsqrt(variance)
	if err != nil {
		return nil, err
	}
	w, err = f.Mul(centered, variance)
	if err != nil {
		return nil, err
	}
	if gamma != nil {
		w, err = f.layerNormScale(w, gamma, ws, axes, orderedAxes)
		if err != nil {
			return nil, err
		}
	}
	if beta != nil {
		b, e := f.layerNormBroadcast(beta, ws, axes, orderedAxes)
		if e != nil {
			return nil, e
		}
		w, err = f.Add(w, b)
		if err != nil {
			return nil, err
		}
	}
	return f.layerNormResult(w, xs.DType)
}

func layerNormCount(dims, axes []int) (int, error) {
	count := 1
	maxInt := int(^uint(0) >> 1)
	for _, axis := range axes {
		dim := dims[axis]
		if dim == 0 {
			return 0, unsupported
		}
		if count > maxInt/dim {
			return 0, unsupported
		}
		count *= dim
	}
	return count, nil
}

func layerNormAxes(rank int, axes []int) ([]int, error) {
	if len(axes) == 0 {
		return nil, fmt.Errorf("fused layer norm: axes must not be empty")
	}
	out := make([]int, len(axes))
	seen := make([]bool, rank)
	for i, axis := range axes {
		if axis < 0 {
			axis += rank
		}
		if axis < 0 || axis >= rank || seen[axis] {
			return nil, fmt.Errorf("fused layer norm: invalid axis %d for rank %d", axes[i], rank)
		}
		seen[axis] = true
		out[i] = axis
	}
	return out, nil
}

func layerNormOuterAxes(rank int, axes []int) []int {
	outer := make([]int, 0, rank-len(axes))
	for axis := range rank {
		found := false
		for _, normalized := range axes {
			found = found || axis == normalized
		}
		if !found {
			outer = append(outer, axis)
		}
	}
	return outer
}

func (f *function) layerNormScaleShape(name string, v compute.Value, x shapes.Shape, axes []int) error {
	if v == nil {
		return nil
	}
	s, err := f.Shape(v)
	if err != nil {
		return fmt.Errorf("fused layer norm %s: %w", name, err)
	}
	if s.DType != x.DType || len(s.Dimensions) != len(axes) {
		return fmt.Errorf("fused layer norm: %s must have dtype %v and rank %d", name, x.DType, len(axes))
	}
	for i, axis := range axes {
		if s.Dimensions[i] < 0 {
			return unsupported
		}
		if s.Dimensions[i] != x.Dimensions[axis] {
			return fmt.Errorf("fused layer norm: %s shape %v does not match axes %v", name, s.Dimensions, axes)
		}
	}
	return nil
}

func (f *function) layerNormScale(v, scale compute.Value, shape shapes.Shape, axes, orderedAxes []int) (compute.Value, error) {
	scale, err := f.layerNormBroadcast(scale, shape, axes, orderedAxes)
	if err != nil {
		return nil, err
	}
	return f.Mul(v, scale)
}

func (f *function) layerNormBroadcast(v compute.Value, shape shapes.Shape, axes, orderedAxes []int) (compute.Value, error) {
	dtype, err := f.Shape(v)
	if err != nil {
		return nil, err
	}
	w, _, err := f.layerNormCompute(v, dtype.DType)
	if err != nil {
		return nil, err
	}
	permutation := make([]int, len(axes))
	for sortedIndex, axis := range orderedAxes {
		for originalIndex, originalAxis := range axes {
			if axis == originalAxis {
				permutation[sortedIndex] = originalIndex
				break
			}
		}
	}
	for i, p := range permutation {
		if i != p {
			w, err = f.Transpose(w, permutation...)
			if err != nil {
				return nil, err
			}
			break
		}
	}
	return f.BroadcastInDim(w, shape, orderedAxes)
}

func (f *function) layerNormCompute(v compute.Value, dtype dtypes.DType) (compute.Value, dtypes.DType, error) {
	if dtype != dtypes.Float16 && dtype != dtypes.BFloat16 {
		return v, dtype, nil
	}
	w, err := f.ConvertDType(v, dtypes.Float32)
	return w, dtypes.Float32, err
}

func (f *function) layerNormResult(v compute.Value, dtype dtypes.DType) (compute.Value, error) {
	if dtype != dtypes.Float16 && dtype != dtypes.BFloat16 {
		return v, nil
	}
	return f.ConvertDType(v, dtype)
}
