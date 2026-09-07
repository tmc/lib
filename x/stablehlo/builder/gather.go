package builder

import (
	"fmt"
	"slices"
)

// GatherRows selects rows along operand's first axis using i32 or i64 indices.
// indices may have any rank, including zero. The result has shape
// [indices.shape..., operand.shape[1:]...]. Shapes must be static and operand's
// first axis nonempty. StableHLO clamps runtime indices to the available rows;
// callers needing rejection must validate indices before execution.
// An error leaves the builder unchanged.
func (b *FuncBuilder) GatherRows(operand, indices Value) (Value, error) {
	if err := checkType(operand.Type); err != nil {
		return Value{}, fmt.Errorf("gather rows: %w", err)
	}
	if len(operand.Type.Shape) == 0 || operand.Type.Shape[0] <= 0 || (indices.Type.ElementType != "i32" && indices.Type.ElementType != "i64") {
		return Value{}, fmt.Errorf("gather rows: nonempty first axis and integer indices required")
	}
	shape := append(slices.Clone(indices.Type.Shape), operand.Type.Shape[1:]...)
	for _, dims := range [][]int64{operand.Type.Shape, indices.Type.Shape, shape} {
		for _, n := range dims {
			if n < 0 {
				return Value{}, fmt.Errorf("gather rows: static shapes required")
			}
		}
		if _, _, err := elements(dims); err != nil {
			return Value{}, fmt.Errorf("gather rows: %w", err)
		}
	}
	return b.gatherRows(operand, indices, shape), nil
}

func (b *FuncBuilder) gatherRows(operand, indices Value, shape []int64) Value {
	offsets := make([]int64, len(operand.Type.Shape)-1)
	for i := range offsets {
		offsets[i] = int64(len(indices.Type.Shape) + i)
	}
	sizes := slices.Clone(operand.Type.Shape)
	sizes[0] = 1
	out := Value{ID: b.next(), Type: Type{Shape: shape, ElementType: operand.Type.ElementType}}
	b.Ops = append(b.Ops, fmt.Sprintf(`%s = "stablehlo.gather"(%s, %s) {dimension_numbers = #stablehlo.gather<offset_dims = %s, collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = %d>, slice_sizes = %s, indices_are_sorted = false} : (%s, %s) -> %s`, out.ID, operand.ID, indices.ID, fmtArr(offsets), len(indices.Type.Shape), fmtI64Array(sizes), operand.Type, indices.Type, out.Type))
	return out
}
