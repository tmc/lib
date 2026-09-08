package gomlx

import (
	"fmt"

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
