package builder

import (
	"fmt"
	"slices"
)

// BatchedCausalMask returns an i1 mask shaped [batch..., queryLength, keyLength].
// positions and counts are same-shaped static i64 tensors with shape [batch...].
// Entry [batch..., q, k] is true when q < counts[batch...] and
// k-q <= positions[batch...]. Lengths must be positive. Runtime positions and
// counts are not clamped; cache callers must validate their bounds.
// An error leaves the builder unchanged.
func (b *FuncBuilder) BatchedCausalMask(queryLength, keyLength int64, positions, counts Value) (Value, error) {
	if positions.Type.ElementType != "i64" || counts.Type.ElementType != "i64" || !slices.Equal(positions.Type.Shape, counts.Type.Shape) {
		return Value{}, fmt.Errorf("batched causal mask: same-shaped i64 positions and counts required")
	}
	if queryLength <= 0 || keyLength <= 0 {
		return Value{}, fmt.Errorf("batched causal mask: positive lengths required")
	}
	shape := append(slices.Clone(positions.Type.Shape), queryLength, keyLength)
	for _, n := range shape {
		if n < 0 {
			return Value{}, fmt.Errorf("batched causal mask: static shapes required")
		}
	}
	_, known, err := elements(shape)
	if err != nil || !known {
		return Value{}, fmt.Errorf("batched causal mask: static shape with representable element count required")
	}
	return b.batchedCausalMask(shape, positions, counts), nil
}

func (b *FuncBuilder) batchedCausalMask(shape []int64, positions, counts Value) Value {
	typ := Type{Shape: shape, ElementType: "i64"}
	iota := func(axis int) Value {
		v := Value{ID: b.next(), Type: typ}
		b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.iota\"() {iota_dimension = %d : i64} : () -> %s", v.ID, axis, typ))
		return v
	}
	dims := make([]int64, len(shape)-2)
	for i := range dims {
		dims[i] = int64(i)
	}
	expand := func(v Value) Value {
		out := Value{ID: b.next(), Type: typ}
		b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.broadcast_in_dim %s, dims = %s : (%s) -> %s", out.ID, v.ID, fmtArr(dims), v.Type, typ))
		return out
	}
	q, k := iota(len(shape)-2), iota(len(shape)-1)
	// Subtracting nonnegative indices cannot overflow, even for extreme offsets.
	delta, _ := b.Subtract(k, q)
	causal := b.compare(delta, expand(positions), "LE", "SIGNED")
	valid := b.compare(q, expand(counts), "LT", "SIGNED")
	out, _ := b.And(causal, valid)
	return out
}
