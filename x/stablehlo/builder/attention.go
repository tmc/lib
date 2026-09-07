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
	return b.softmax(v, dimension)
}

func (b *FuncBuilder) softmax(v Value, dimension int64) (Value, error) {
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
	sum, err := b.Reduce(exp, zero, []int64{dimension}, reduced, "add")
	if err != nil {
		return Value{}, err
	}
	sum, err = b.BroadcastInDim(sum, mapping, v.Type)
	if err != nil {
		return Value{}, err
	}
	return b.Divide(exp, sum)
}

// ScaledDotProductAttention computes softmax(q*kᵀ/sqrt(d))*v for static f32
// tensors with shape [batch..., sequence, features]. Batch dimensions must
// match; q and k must have the same feature count, and k and v the same sequence
// length. It applies no mask or dropout. An error leaves the builder unchanged.
func (b *FuncBuilder) ScaledDotProductAttention(q, k, v Value) (result Value, err error) {
	if err := sameElements(q.Type, k.Type, v.Type); err != nil {
		return Value{}, fmt.Errorf("attention: %w", err)
	}
	for _, typ := range []Type{q.Type, k.Type, v.Type} {
		if err := staticFloat(typ); err != nil {
			return Value{}, fmt.Errorf("attention: %w", err)
		}
	}
	rank := len(q.Type.Shape)
	if q.Type.ElementType != "f32" || rank < 2 || len(k.Type.Shape) != rank || len(v.Type.Shape) != rank {
		return Value{}, fmt.Errorf("attention: matching ranks of at least two and f32 elements required")
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
	return b.attention(q, k, v)
}

func (b *FuncBuilder) attention(q, k, v Value) (Value, error) {
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
	probabilities, err := b.Softmax(scores, int64(rank-1))
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
