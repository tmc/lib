package builder

import (
	"fmt"
	"math"
	"slices"
)

// MaskedScaledDotProductAttention is ScaledDotProductAttention with a boolean
// mask. True entries attend to a key. The mask has i1 elements and broadcasts
// along trailing score dimensions [batch..., query, key]; dimensions of one
// may expand. For example, [query, key] applies the same mask to every batch,
// and [batch, 1, key] masks padding keys independently for each batch.
// Fully masked rows return zeros for finite inputs. No dropout is applied.
// An error leaves the builder unchanged.
func (b *FuncBuilder) MaskedScaledDotProductAttention(q, k, v, mask Value) (Value, error) {
	return b.scaledAttention(q, k, v, &mask)
}

func (b *FuncBuilder) broadcastMask(mask Value, shape []int64) (Value, error) {
	if mask.Type.ElementType != "i1" || len(mask.Type.Shape) > len(shape) {
		return Value{}, fmt.Errorf("attention: boolean mask of at most score rank required")
	}
	mapping := make([]int64, len(mask.Type.Shape))
	for i, n := range mask.Type.Shape {
		d := len(shape) - len(mask.Type.Shape) + i
		if n < 0 || (n != 1 && n != shape[d]) {
			return Value{}, fmt.Errorf("attention: mask dimension %d cannot broadcast to %d", n, shape[d])
		}
		mapping[i] = int64(d)
	}
	return b.BroadcastInDim(mask, mapping, Type{Shape: slices.Clone(shape), ElementType: "i1"})
}

// CausalMask returns a [queryLength, keyLength] boolean mask. Entry [q, k] is
// true when k <= queryOffset+q. Use offset zero for prefill, or the number of
// cached keys preceding the query for decoding. Lengths must be positive and
// the offset nonnegative. An error leaves the builder unchanged.
func (b *FuncBuilder) CausalMask(queryLength, keyLength, queryOffset int64) (Value, error) {
	if queryLength <= 0 || keyLength <= 0 || queryOffset < 0 || queryOffset > math.MaxInt64-(queryLength-1) {
		return Value{}, fmt.Errorf("causal mask: invalid lengths or query offset")
	}
	if queryLength > math.MaxInt64/keyLength {
		return Value{}, fmt.Errorf("causal mask: element count overflows int64")
	}
	return b.causalMask(queryLength, keyLength, queryOffset), nil
}

func (b *FuncBuilder) causalMask(queryLength, keyLength, queryOffset int64) Value {
	typ := Type{Shape: []int64{queryLength, keyLength}, ElementType: "i64"}
	iota := func(dimension int) Value {
		out := Value{ID: b.next(), Type: typ}
		b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.iota\"() {iota_dimension = %d : i64} : () -> %s", out.ID, dimension, typ))
		return out
	}
	q, k := iota(0), iota(1)
	if queryOffset != 0 {
		offset := Value{ID: b.next(), Type: typ}
		b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%d> : %s", offset.ID, queryOffset, typ))
		shifted := Value{ID: b.next(), Type: typ}
		b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.add %s, %s : %s", shifted.ID, q.ID, offset.ID, typ))
		q = shifted
	}
	return b.compare(k, q, "LE", "SIGNED")
}

// And computes the elementwise conjunction of two boolean masks of equal shape.
// An error leaves the builder unchanged.
func (b *FuncBuilder) And(lhs, rhs Value) (Value, error) {
	if lhs.Type.ElementType != "i1" || rhs.Type.ElementType != "i1" {
		return Value{}, fmt.Errorf("and: boolean operands required")
	}
	return b.binaryOp("stablehlo.and", lhs, rhs)
}

func (b *FuncBuilder) selectValue(pred, yes, no Value) Value {
	out := Value{ID: b.next(), Type: yes.Type}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.select\"(%s, %s, %s) : (%s, %s, %s) -> %s", out.ID, pred.ID, yes.ID, no.ID, pred.Type, yes.Type, no.Type, out.Type))
	return out
}

func (b *FuncBuilder) compare(lhs, rhs Value, direction, kind string) Value {
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(lhs.Type.Shape), ElementType: "i1"}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.compare\"(%s, %s) {comparison_direction = #stablehlo<comparison_direction %s>, compare_type = #stablehlo<comparison_type %s>} : (%s, %s) -> %s", out.ID, lhs.ID, rhs.ID, direction, kind, lhs.Type, rhs.Type, out.Type))
	return out
}
