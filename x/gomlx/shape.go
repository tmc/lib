package gomlx

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Transpose(v compute.Value, permutation ...int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if len(permutation) != len(x.Type.Shape) {
		return nil, fmt.Errorf("transpose: permutation rank mismatch")
	}
	dims := make([]int64, len(permutation))
	seen := make([]bool, len(permutation))
	for i, axis := range permutation {
		if axis < 0 || axis >= len(dims) || seen[axis] {
			return nil, fmt.Errorf("transpose: invalid or repeated axis %d", axis)
		}
		seen[axis] = true
		dims[i] = x.Type.Shape[axis]
	}
	z, err := f.raw.Transpose(x, int64s(permutation), hlo.Type{Shape: dims, ElementType: x.Type.ElementType})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}

func (f *function) Iota(shape shapes.Shape, axis int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if err := validate(shape); err != nil {
		return nil, err
	}
	if shape.DType == dtypes.Bool {
		return nil, unsupported
	}
	if axis < 0 || axis >= len(shape.Dimensions) {
		return nil, fmt.Errorf("iota: invalid axis %d", axis)
	}
	info, _ := element(shape.DType)
	z := hlo.Value{ID: fmt.Sprintf("%%iota%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(shape.Dimensions), ElementType: info.hlo}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.iota\"() {iota_dimension = %d : i64} : () -> %s", z.ID, axis, z.Type))
	return &node{f, z}, nil
}

func (f *function) Slice(v compute.Value, starts, limits, strides []int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	rank := len(x.Type.Shape)
	if len(starts) != rank || len(limits) != rank || len(strides) != 0 && len(strides) != rank {
		return nil, fmt.Errorf("slice: bounds and strides must match rank")
	}
	steps := make([]int64, rank)
	ends := int64s(limits)
	dims := make([]int64, rank)
	for i, n := range x.Type.Shape {
		steps[i] = 1
		if len(strides) != 0 {
			steps[i] = int64(strides[i])
		}
		if ends[i] == -1 {
			ends[i] = n
		}
		start := int64(starts[i])
		if start < 0 || ends[i] < start || ends[i] > n || steps[i] <= 0 {
			return nil, fmt.Errorf("slice: invalid bounds or stride at axis %d", i)
		}
		if ends[i] > start {
			dims[i] = 1 + (ends[i]-start-1)/steps[i]
		}
	}
	z, err := f.raw.Slice(x, int64s(starts), ends, steps, hlo.Type{Shape: dims, ElementType: x.Type.ElementType})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}

func (f *function) Concatenate(axis int, operands ...compute.Value) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if len(operands) == 0 {
		return nil, fmt.Errorf("concatenate: no operands")
	}
	values := make([]hlo.Value, len(operands))
	for i, v := range operands {
		x, err := f.value(v)
		if err != nil {
			return nil, err
		}
		values[i] = x
	}
	typ := values[0].Type
	if axis < 0 || axis >= len(typ.Shape) {
		return nil, fmt.Errorf("concatenate: invalid axis %d", axis)
	}
	dims := append([]int64(nil), typ.Shape...)
	dims[axis] = 0
	for _, v := range values {
		if len(v.Type.Shape) != len(dims) {
			return nil, fmt.Errorf("concatenate: rank mismatch")
		}
		n := v.Type.Shape[axis]
		if n > math.MaxInt64-dims[axis] {
			return nil, fmt.Errorf("concatenate: dimension overflow")
		}
		dims[axis] += n
	}
	z, err := f.raw.Concatenate(values, int64(axis), hlo.Type{Shape: dims, ElementType: typ.ElementType})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
