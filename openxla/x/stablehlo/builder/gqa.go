package builder

import (
	"fmt"
	"math"
	"slices"
)

// GroupedQueryAttention computes scaled dot-product attention for static
// [batch..., heads, sequence, features] tensors. Query heads must be a positive
// multiple of the matching K/V head count. Query head h uses KV head h/groups,
// where groups is the query-head count divided by the KV-head count. Batch
// dimensions must match; Q/K feature sizes and K/V sequence lengths must match.
// f16 and bf16 inputs compute in f32 and return their original storage type.
// The graph groups queries without explicitly replicating K/V tensors.
// An error leaves the builder unchanged.
func (b *FuncBuilder) GroupedQueryAttention(q, k, v Value) (Value, error) {
	return b.groupedQueryAttention(q, k, v, nil)
}

// MaskedGroupedQueryAttention is GroupedQueryAttention with a boolean mask.
// Its mask follows MaskedScaledDotProductAttention broadcasting over score
// dimensions [batch..., queryHeads, querySequence, keySequence]. Fully masked
// rows return zeros for finite inputs. No dropout is applied.
// An error leaves the builder unchanged.
func (b *FuncBuilder) MaskedGroupedQueryAttention(q, k, v, mask Value) (Value, error) {
	return b.groupedQueryAttention(q, k, v, &mask)
}

func (b *FuncBuilder) groupedQueryAttention(q, k, v Value, mask *Value) (result Value, err error) {
	if err := sameElements(q.Type, k.Type, v.Type); err != nil {
		return Value{}, fmt.Errorf("grouped attention: %w", err)
	}
	for _, typ := range []Type{q.Type, k.Type, v.Type} {
		if err := staticFloat(typ); err != nil {
			return Value{}, fmt.Errorf("grouped attention: %w", err)
		}
	}
	rank := len(q.Type.Shape)
	if rank < 3 || len(k.Type.Shape) != rank || len(v.Type.Shape) != rank {
		return Value{}, fmt.Errorf("grouped attention: matching ranks of at least three required")
	}
	element := q.Type.ElementType
	if element != "f32" && element != "f16" && element != "bf16" {
		return Value{}, fmt.Errorf("grouped attention: f32, f16 or bf16 required")
	}
	head := rank - 3
	qheads, kheads := q.Type.Shape[head], k.Type.Shape[head]
	if qheads <= 0 || kheads <= 0 || qheads%kheads != 0 || v.Type.Shape[head] != kheads {
		return Value{}, fmt.Errorf("grouped attention: query heads must be a positive multiple of matching KV heads")
	}
	if !slices.Equal(q.Type.Shape[:head], k.Type.Shape[:head]) || !slices.Equal(q.Type.Shape[:head], v.Type.Shape[:head]) {
		return Value{}, fmt.Errorf("grouped attention: batch dimensions differ")
	}
	if q.Type.Shape[rank-1] <= 0 || q.Type.Shape[rank-1] != k.Type.Shape[rank-1] || k.Type.Shape[rank-2] <= 0 || k.Type.Shape[rank-2] != v.Type.Shape[rank-2] {
		return Value{}, fmt.Errorf("grouped attention: incompatible feature or sequence dimensions")
	}
	nops, next := len(b.Ops), b.nextID
	defer func() {
		if err != nil {
			b.Ops, b.nextID = b.Ops[:nops], next
		}
	}()
	if element != "f32" {
		q = b.convert(q, "f32")
		k = b.convert(k, "f32")
		v = b.convert(v, "f32")
	}
	result, err = b.groupedAttention(q, k, v, mask)
	if err == nil && element != "f32" {
		result = b.convert(result, element)
	}
	return result, err
}

func (b *FuncBuilder) groupedAttention(q, k, v Value, mask *Value) (Value, error) {
	rank := len(q.Type.Shape)
	head := rank - 3
	groups := q.Type.Shape[head] / k.Type.Shape[head]
	groupedShape := append(slices.Clone(q.Type.Shape[:head]), k.Type.Shape[head], groups, q.Type.Shape[rank-2], q.Type.Shape[rank-1])
	groupedQ, err := b.Reshape(q, Type{Shape: groupedShape, ElementType: "f32"})
	if err != nil {
		return Value{}, err
	}
	batch := make([]int64, head+1)
	for i := range batch {
		batch[i] = int64(i)
	}
	scoreShape := slices.Clone(groupedShape)
	scoreShape[len(scoreShape)-1] = k.Type.Shape[rank-2]
	scores, err := b.DotGeneral(groupedQ, k, batch, batch, []int64{int64(rank)}, []int64{int64(rank - 1)}, Type{Shape: scoreShape, ElementType: "f32"})
	if err != nil {
		return Value{}, err
	}
	flatShape := slices.Clone(q.Type.Shape)
	flatShape[rank-1] = k.Type.Shape[rank-2]
	scores, err = b.Reshape(scores, Type{Shape: flatShape, ElementType: "f32"})
	if err != nil {
		return Value{}, err
	}
	scale, err := b.Constant(float32(1/math.Sqrt(float64(q.Type.Shape[rank-1]))), scores.Type)
	if err != nil {
		return Value{}, err
	}
	scores, err = b.Multiply(scores, scale)
	if err != nil {
		return Value{}, err
	}
	if mask != nil {
		expanded, err := b.broadcastMask(*mask, flatShape)
		if err != nil {
			return Value{}, err
		}
		mask = &expanded
	}
	probabilities, err := b.softmax(scores, int64(rank-1), mask)
	if err != nil {
		return Value{}, err
	}
	probabilities, err = b.Reshape(probabilities, Type{Shape: scoreShape, ElementType: "f32"})
	if err != nil {
		return Value{}, err
	}
	groupedOutput := slices.Clone(groupedShape)
	groupedOutput[len(groupedOutput)-1] = v.Type.Shape[rank-1]
	out, err := b.DotGeneral(probabilities, v, batch, batch, []int64{int64(rank)}, []int64{int64(rank - 2)}, Type{Shape: groupedOutput, ElementType: "f32"})
	if err != nil {
		return Value{}, err
	}
	flatOutput := slices.Clone(q.Type.Shape)
	flatOutput[rank-1] = v.Type.Shape[rank-1]
	return b.Reshape(out, Type{Shape: flatOutput, ElementType: "f32"})
}
