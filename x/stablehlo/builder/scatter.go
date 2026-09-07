package builder

import (
	"fmt"
	"slices"
)

// ScatterRows replaces rows along operand's first axis. indices is a vector
// of i32 or i64 row indices; updates has shape [len(indices), operand.shape[1:]...]
// and operand's element type. Shapes must be static and operand non-scalar.
// Out-of-range indices are ignored. For repeated in-range indices, the chosen
// update is unspecified. An error leaves the builder unchanged.
func (b *FuncBuilder) ScatterRows(operand, indices, updates Value) (Value, error) {
	if err := sameElements(operand.Type, updates.Type); err != nil {
		return Value{}, fmt.Errorf("scatter rows: %w", err)
	}
	rank := len(operand.Type.Shape)
	if rank == 0 || len(updates.Type.Shape) != rank || len(indices.Type.Shape) != 1 ||
		(indices.Type.ElementType != "i32" && indices.Type.ElementType != "i64") {
		return Value{}, fmt.Errorf("scatter rows: matching non-scalar ranks and integer index vector required")
	}
	for _, typ := range []Type{operand.Type, indices.Type, updates.Type} {
		for _, n := range typ.Shape {
			if n < 0 {
				return Value{}, fmt.Errorf("scatter rows: static shapes required")
			}
		}
		_, known, err := elements(typ.Shape)
		if err != nil || !known {
			return Value{}, fmt.Errorf("scatter rows: static shapes with representable element counts required")
		}
	}
	if updates.Type.Shape[0] != indices.Type.Shape[0] || !slices.Equal(operand.Type.Shape[1:], updates.Type.Shape[1:]) {
		return Value{}, fmt.Errorf("scatter rows: update shape must match indices and operand rows")
	}
	return b.scatterRows(operand, indices, updates), nil
}

func (b *FuncBuilder) scatterRows(operand, indices, updates Value) Value {
	window := make([]int64, len(operand.Type.Shape)-1)
	for i := range window {
		window[i] = int64(i + 1)
	}
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(operand.Type.Shape), ElementType: operand.Type.ElementType}}
	scalar := Type{ElementType: operand.Type.ElementType}
	b.Ops = append(b.Ops, fmt.Sprintf(`%s = "stablehlo.scatter"(%s, %s, %s) ({
      ^bb0(%%old: %s, %%new: %s):
        "stablehlo.return"(%%new) : (%s) -> ()
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = %s, inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : (%s, %s, %s) -> %s`, out.ID, operand.ID, indices.ID, updates.ID, scalar, scalar, scalar, fmtArr(window), operand.Type, indices.Type, updates.Type, out.Type))
	return out
}
