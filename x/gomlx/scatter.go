package gomlx

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) ScatterSum(operand, indices, updates compute.Value, vectorAxis int, windowAxes, insertedAxes, mapping []int, sorted, unique bool) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(operand)
	if err != nil {
		return nil, err
	}
	idx, err := f.value(indices)
	if err != nil {
		return nil, err
	}
	u, err := f.value(updates)
	if err != nil {
		return nil, err
	}
	xs, err := f.Shape(operand)
	if err != nil {
		return nil, err
	}
	is, err := f.Shape(indices)
	if err != nil {
		return nil, err
	}
	us, err := f.Shape(updates)
	if err != nil {
		return nil, err
	}
	if x.Type.ElementType == "i1" {
		return nil, fmt.Errorf("scatter sum: numeric operand required")
	}
	if vectorAxis < 0 || vectorAxis > is.Rank() {
		return nil, fmt.Errorf("scatter sum: invalid index vector axis")
	}
	// Normalize the implicit axis for inference, including its error paths.
	if vectorAxis == is.Rank() {
		is.Dimensions = append(slices.Clone(is.Dimensions), 1)
	}
	for i, axes := range [][]int{windowAxes, insertedAxes, mapping} {
		seen := make(map[int]bool)
		for _, axis := range axes {
			if seen[axis] {
				return nil, fmt.Errorf("scatter sum: duplicate dimension")
			}
			seen[axis] = true
		}
		if i < 2 && !slices.IsSorted(axes) {
			return nil, fmt.Errorf("scatter sum: window dimensions must be sorted")
		}
	}
	shape, err := shapeinference.Scatter(xs, is, us, vectorAxis, windowAxes, insertedAxes, mapping)
	if err != nil {
		return nil, fmt.Errorf("scatter sum: %w", err)
	}
	// Inference checks ranks and window sizes but not the batch dimensions.
	batch := 0
	for axis, size := range us.Dimensions {
		if slices.Contains(windowAxes, axis) {
			continue
		}
		if batch == vectorAxis {
			batch++
		}
		if size != is.Dimensions[batch] {
			return nil, fmt.Errorf("scatter sum: batch dimensions do not match")
		}
		batch++
	}
	if err := validate(shape); err != nil {
		return nil, err
	}
	z := hlo.Value{ID: fmt.Sprintf("%%scatter%d", len(f.raw.Ops)), Type: x.Type}
	scalar := hlo.Type{ElementType: x.Type.ElementType}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.scatter"(%s, %s, %s) ({
 ^bb0(%%lhs: %s, %%rhs: %s):
  %%sum = stablehlo.add %%lhs, %%rhs : %s
  stablehlo.return %%sum : %s
 }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = %s, inserted_window_dims = %s, scatter_dims_to_operand_dims = %s, index_vector_dim = %d>, indices_are_sorted = %t, unique_indices = %t} : (%s, %s, %s) -> %s`, z.ID, x.ID, idx.ID, u.ID, scalar, scalar, scalar, scalar, indexArray(windowAxes), indexArray(insertedAxes), indexArray(mapping), vectorAxis, sorted, unique, x.Type, idx.Type, u.Type, z.Type))
	return &node{f, z}, nil
}
