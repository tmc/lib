package builder

import (
	"fmt"
	"math"
	"slices"
)

// Softmax normalizes a floating-point tensor along dimension using a stable
// maximum-subtraction formulation. The shape must be static and the selected
// dimension must be nonempty. An error leaves the builder unchanged.
func (b *FuncBuilder) Softmax(v Value, dimension int64) (result Value, err error) {
	if err := staticFloat(v.Type); err != nil {
		return Value{}, fmt.Errorf("softmax: %w", err)
	}
	if dimension < 0 || dimension >= int64(len(v.Type.Shape)) || v.Type.Shape[dimension] == 0 {
		return Value{}, fmt.Errorf("softmax: invalid or empty dimension")
	}
	nops, next := len(b.Ops), b.nextID
	defer func() {
		if err != nil {
			b.Ops, b.nextID = b.Ops[:nops], next
		}
	}()
	return b.softmax(v, dimension, nil)
}

func (b *FuncBuilder) softmax(v Value, dimension int64, mask *Value) (Value, error) {
	scalar := Type{ElementType: v.Type.ElementType}
	var shape, mapping []int64
	for i, n := range v.Type.Shape {
		if int64(i) != dimension {
			shape = append(shape, n)
			mapping = append(mapping, int64(i))
		}
	}
	reduced := Type{Shape: shape, ElementType: v.Type.ElementType}
	initial, err := b.Constant(float32(math.Inf(-1)), scalar)
	if err != nil {
		return Value{}, err
	}
	if mask != nil {
		fill, err := b.BroadcastInDim(initial, nil, v.Type)
		if err != nil {
			return Value{}, err
		}
		v = b.selectValue(*mask, v, fill)
	}
	maximum, err := b.Reduce(v, initial, []int64{dimension}, reduced, "maximum")
	if err != nil {
		return Value{}, err
	}
	maximum, err = b.BroadcastInDim(maximum, mapping, v.Type)
	if err != nil {
		return Value{}, err
	}
	shifted, err := b.Subtract(v, maximum)
	if err != nil {
		return Value{}, err
	}
	exp, err := b.Exponential(shifted)
	if err != nil {
		return Value{}, err
	}
	zero, err := b.Constant(0, scalar)
	if err != nil {
		return Value{}, err
	}
	if mask != nil {
		zeros, err := b.BroadcastInDim(zero, nil, v.Type)
		if err != nil {
			return Value{}, err
		}
		exp = b.selectValue(*mask, exp, zeros)
	}
	sum, err := b.Reduce(exp, zero, []int64{dimension}, reduced, "add")
	if err != nil {
		return Value{}, err
	}
	sum, err = b.BroadcastInDim(sum, mapping, v.Type)
	if err != nil {
		return Value{}, err
	}
	if mask != nil {
		zeros, err := b.Constant(0, v.Type)
		if err != nil {
			return Value{}, err
		}
		ones, err := b.Constant(1, v.Type)
		if err != nil {
			return Value{}, err
		}
		// A fully masked row has zero exponential sum. Divide by one to
		// preserve its zero probabilities instead of producing NaNs.
		sum = b.selectValue(b.compare(sum, zeros, "EQ", "FLOAT"), ones, sum)
	}
	return b.Divide(exp, sum)
}

// ScaledDotProductAttention computes softmax(q*kᵀ/sqrt(d))*v for static f32,
// f16 or bf16 tensors with shape [batch..., sequence, features]. Batch dimensions
// must match; q and k must have the same feature count, and k and v the same sequence
// length. Reduced-precision inputs are converted to f32 for all arithmetic;
// the result is converted back to the input element type. It applies no mask
// or dropout. An error leaves the builder unchanged.
func (b *FuncBuilder) ScaledDotProductAttention(q, k, v Value) (Value, error) {
	return b.scaledAttention(q, k, v, nil)
}

func (b *FuncBuilder) scaledAttention(q, k, v Value, mask *Value) (result Value, err error) {
	if err := sameElements(q.Type, k.Type, v.Type); err != nil {
		return Value{}, fmt.Errorf("attention: %w", err)
	}
	for _, typ := range []Type{q.Type, k.Type, v.Type} {
		if err := staticFloat(typ); err != nil {
			return Value{}, fmt.Errorf("attention: %w", err)
		}
	}
	rank := len(q.Type.Shape)
	if (q.Type.ElementType != "f32" && q.Type.ElementType != "f16" && q.Type.ElementType != "bf16") || rank < 2 || len(k.Type.Shape) != rank || len(v.Type.Shape) != rank {
		return Value{}, fmt.Errorf("attention: matching ranks of at least two and f32, f16 or bf16 elements required")
	}
	for i := 0; i < rank-2; i++ {
		if q.Type.Shape[i] != k.Type.Shape[i] || q.Type.Shape[i] != v.Type.Shape[i] {
			return Value{}, fmt.Errorf("attention: batch dimension %d differs", i)
		}
	}
	if q.Type.Shape[rank-1] <= 0 || q.Type.Shape[rank-1] != k.Type.Shape[rank-1] || k.Type.Shape[rank-2] <= 0 || k.Type.Shape[rank-2] != v.Type.Shape[rank-2] {
		return Value{}, fmt.Errorf("attention: incompatible feature or sequence dimensions")
	}
	nops, next := len(b.Ops), b.nextID
	defer func() {
		if err != nil {
			b.Ops, b.nextID = b.Ops[:nops], next
		}
	}()
	if mask != nil {
		shape := slices.Clone(q.Type.Shape)
		shape[rank-1] = k.Type.Shape[rank-2]
		expanded, err := b.broadcastMask(*mask, shape)
		if err != nil {
			return Value{}, err
		}
		mask = &expanded
	}
	element := q.Type.ElementType
	if element != "f32" {
		q = b.convert(q, "f32")
		k = b.convert(k, "f32")
		v = b.convert(v, "f32")
	}
	result, err = b.attention(q, k, v, mask)
	if err == nil && element != "f32" {
		result = b.convert(result, element)
	}
	return result, err
}

func (b *FuncBuilder) attention(q, k, v Value, mask *Value) (Value, error) {
	rank := len(q.Type.Shape)
	batch := make([]int64, rank-2)
	for i := range batch {
		batch[i] = int64(i)
	}
	scoresType := Type{Shape: slices.Clone(q.Type.Shape), ElementType: "f32"}
	scoresType.Shape[rank-1] = k.Type.Shape[rank-2]
	scores, err := b.DotGeneral(q, k, batch, batch, []int64{int64(rank - 1)}, []int64{int64(rank - 1)}, scoresType)
	if err != nil {
		return Value{}, err
	}
	scale, err := b.Constant(float32(1/math.Sqrt(float64(q.Type.Shape[rank-1]))), scoresType)
	if err != nil {
		return Value{}, err
	}
	scores, err = b.Multiply(scores, scale)
	if err != nil {
		return Value{}, err
	}
	probabilities, err := b.softmax(scores, int64(rank-1), mask)
	if err != nil {
		return Value{}, err
	}
	resultType := Type{Shape: slices.Clone(q.Type.Shape), ElementType: "f32"}
	resultType.Shape[rank-1] = v.Type.Shape[rank-1]
	return b.DotGeneral(probabilities, v, batch, batch, []int64{int64(rank - 1)}, []int64{int64(rank - 2)}, resultType)
}

func staticFloat(typ Type) error {
	if err := checkType(typ); err != nil {
		return err
	}
	if !floatType(typ.ElementType) {
		return fmt.Errorf("floating-point elements required")
	}
	for _, n := range typ.Shape {
		if n < 0 {
			return fmt.Errorf("static shape required")
		}
	}
	return nil
}
