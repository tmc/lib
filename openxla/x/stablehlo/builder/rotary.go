package builder

import (
	"fmt"
	"slices"
)

// RotaryEmbedding applies split-half rotary embeddings to x's last axis.
// cosine and sine are same-shaped f32 tensors with last dimension r. Features
// i and i+r are paired for i<r; any features after 2*r are unchanged. Tables
// broadcast along trailing dimensions of x with its last dimension replaced
// by r. For x shaped [batch, heads, sequence, features], tables may have shape
// [sequence, r] or [batch, 1, sequence, r]. The caller supplies the table values
// for its position, frequency and scaling convention.
//
// x must have f32, f16 or bf16 elements, with rank at least one. All shapes are
// static, r is positive, and 2*r must fit x's last dimension. Rotation computes
// in f32 and converts back once. An error leaves the builder unchanged.
func (b *FuncBuilder) RotaryEmbedding(x, cosine, sine Value) (out Value, err error) {
	if err := activationType(x.Type); err != nil {
		return Value{}, fmt.Errorf("rotary embedding: %w", err)
	}
	rank, tableRank := len(x.Type.Shape), len(cosine.Type.Shape)
	if rank == 0 || tableRank == 0 || tableRank > rank || cosine.Type.ElementType != "f32" || sine.Type.ElementType != "f32" || !slices.Equal(cosine.Type.Shape, sine.Type.Shape) {
		return Value{}, fmt.Errorf("rotary embedding: matching non-scalar f32 tables of at most input rank required")
	}
	r := cosine.Type.Shape[tableRank-1]
	if r <= 0 || r > x.Type.Shape[rank-1]/2 {
		return Value{}, fmt.Errorf("rotary embedding: rotation must fit the feature dimension")
	}
	shape := slices.Clone(x.Type.Shape)
	shape[rank-1] = r
	dims := make([]int64, tableRank)
	for i, n := range cosine.Type.Shape {
		d := rank - tableRank + i
		if n < 0 || (n != 1 && n != shape[d]) {
			return Value{}, fmt.Errorf("rotary embedding: table dimension cannot broadcast to input")
		}
		dims[i] = int64(d)
	}
	nops, next := len(b.Ops), b.nextID
	defer func() {
		if err != nil {
			b.Ops, b.nextID = b.Ops[:nops], next
		}
	}()
	return b.rotaryEmbedding(x, cosine, sine, shape, dims, r)
}

func (b *FuncBuilder) rotaryEmbedding(x, cosine, sine Value, shape, dims []int64, r int64) (Value, error) {
	element := x.Type.ElementType
	x = b.convert(x, "f32")
	partType := Type{Shape: shape, ElementType: "f32"}
	cosine, err := b.BroadcastInDim(cosine, dims, partType)
	if err != nil {
		return Value{}, err
	}
	sine, err = b.BroadcastInDim(sine, dims, partType)
	if err != nil {
		return Value{}, err
	}
	slice := func(start, end int64) (Value, error) {
		lo := make([]int64, len(shape))
		hi := slices.Clone(x.Type.Shape)
		stride := make([]int64, len(shape))
		for i := range stride {
			stride[i] = 1
		}
		lo[len(lo)-1], hi[len(hi)-1] = start, end
		typ := Type{Shape: slices.Clone(x.Type.Shape), ElementType: "f32"}
		typ.Shape[len(shape)-1] = end - start
		return b.Slice(x, lo, hi, stride, typ)
	}
	first, err := slice(0, r)
	if err != nil {
		return Value{}, err
	}
	second, err := slice(r, 2*r)
	if err != nil {
		return Value{}, err
	}
	fc, _ := b.Multiply(first, cosine)
	ss, _ := b.Multiply(second, sine)
	fs, _ := b.Multiply(first, sine)
	sc, _ := b.Multiply(second, cosine)
	left, _ := b.Subtract(fc, ss)
	right, _ := b.Add(fs, sc)
	parts := []Value{left, right}
	if n := x.Type.Shape[len(shape)-1]; 2*r < n {
		tail, err := slice(2*r, n)
		if err != nil {
			return Value{}, err
		}
		parts = append(parts, tail)
	}
	out, err := b.Concatenate(parts, int64(len(shape)-1), x.Type)
	if err != nil {
		return Value{}, err
	}
	return b.convert(out, element), nil
}
