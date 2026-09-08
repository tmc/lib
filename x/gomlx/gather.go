package gomlx

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Gather(operand, indices compute.Value, indexVectorAxis int, offsetAxes, collapsedAxes, startMap, sizes []int, sorted bool) (compute.Value, error) {
	if f.returned {
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
	switch idx.Type.ElementType {
	case "i8", "i32", "i64", "ui8", "ui32", "ui64":
	default:
		return nil, fmt.Errorf("gather: integer indices required")
	}
	xs, err := f.Shape(operand)
	if err != nil {
		return nil, err
	}
	is, err := f.Shape(indices)
	if err != nil {
		return nil, err
	}
	if indexVectorAxis < 0 || indexVectorAxis > len(is.Dimensions) {
		return nil, fmt.Errorf("gather: invalid index vector axis")
	}
	// Shape inference indexes the vector axis even when the final size-one
	// dimension is implicit in the operation.
	if indexVectorAxis == len(is.Dimensions) {
		is.Dimensions = append(slices.Clone(is.Dimensions), 1)
	}
	for _, axes := range [][]int{offsetAxes, collapsedAxes} {
		if !slices.IsSorted(axes) {
			return nil, fmt.Errorf("gather: offset and collapsed axes must be sorted")
		}
	}
	seen := make(map[int]bool)
	for _, axis := range startMap {
		if seen[axis] {
			return nil, fmt.Errorf("gather: duplicate start index axis")
		}
		seen[axis] = true
	}
	for _, n := range sizes {
		if n < 0 {
			return nil, fmt.Errorf("gather: static slice sizes required")
		}
	}
	shape, err := shapeinference.Gather(xs, is, indexVectorAxis, offsetAxes, collapsedAxes, startMap, sizes, sorted)
	if err != nil {
		return nil, fmt.Errorf("gather: %w", err)
	}
	if err := validate(shape); err != nil {
		return nil, err
	}
	z := hlo.Value{ID: fmt.Sprintf("%%gather%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(shape.Dimensions), ElementType: x.Type.ElementType}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.gather\"(%s, %s) {dimension_numbers = #stablehlo.gather<offset_dims = %s, collapsed_slice_dims = %s, start_index_map = %s, index_vector_dim = %d>, indices_are_sorted = %t, slice_sizes = array<i64: %s>} : (%s, %s) -> %s", z.ID, x.ID, idx.ID, indexArray(offsetAxes), indexArray(collapsedAxes), indexArray(startMap), indexVectorAxis, sorted, indexValues(sizes), x.Type, idx.Type, z.Type))
	return &node{f, z}, nil
}
func indexValues(v []int) string {
	s := make([]string, len(v))
	for i, n := range v {
		s[i] = strconv.Itoa(n)
	}
	return strings.Join(s, ", ")
}
func indexArray(v []int) string { return "[" + indexValues(v) + "]" }
