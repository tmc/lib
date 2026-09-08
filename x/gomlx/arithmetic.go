package gomlx

import (
	"fmt"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Mul(a, b compute.Value) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(a)
	if err != nil {
		return nil, err
	}
	y, err := f.value(b)
	if err != nil {
		return nil, err
	}
	x, y, err = f.broadcastPair(x, y)
	if err != nil {
		return nil, err
	}
	z, err := f.raw.Multiply(x, y)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) BroadcastInDim(v compute.Value, s shapes.Shape, axes []int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if err := validate(s); err != nil {
		return nil, err
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	z, err := f.raw.BroadcastInDim(x, int64s(axes), hlo.Type{Shape: int64s(s.Dimensions), ElementType: "f32"})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) Reshape(v compute.Value, dims ...int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if err := validate(shapes.Shape{DType: dtypes.Float32, Dimensions: dims}); err != nil {
		return nil, err
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	z, err := f.raw.Reshape(x, hlo.Type{Shape: int64s(dims), ElementType: "f32"})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) ReduceSum(v compute.Value, axes ...int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if len(axes) == 0 {
		axes = make([]int, len(x.Type.Shape))
		for i := range axes {
			axes[i] = i
		}
	}
	seen := make([]bool, len(x.Type.Shape))
	for _, a := range axes {
		if a < 0 || a >= len(seen) || seen[a] {
			return nil, fmt.Errorf("invalid reduction axis %d", a)
		}
		seen[a] = true
	}
	var dims []int64
	for i, d := range x.Type.Shape {
		if !seen[i] {
			dims = append(dims, d)
		}
	}
	zero, err := f.raw.Constant(0, hlo.Type{ElementType: "f32"})
	if err != nil {
		return nil, err
	}
	z, err := f.raw.Reduce(x, zero, int64s(axes), hlo.Type{Shape: dims, ElementType: "f32"}, "add")
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
