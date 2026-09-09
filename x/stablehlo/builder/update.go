package builder

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// IntegerConstant emits a splat constant with i8, i16, i32, i64, or ui16 elements and a static
// shape. The value must fit the element type. An error leaves the builder unchanged.
func (b *FuncBuilder) IntegerConstant(value int64, typ Type) (Value, error) {
	if err := checkType(typ); err != nil {
		return Value{}, fmt.Errorf("integer constant: %w", err)
	}
	if typ.ElementType != "i64" && typ.ElementType != "i32" && typ.ElementType != "i16" && typ.ElementType != "i8" && typ.ElementType != "ui16" {
		return Value{}, fmt.Errorf("integer constant: i8, i16, i32, i64, or ui16 elements required")
	}
	if typ.ElementType == "i8" && (value < math.MinInt8 || value > math.MaxInt8) {
		return Value{}, fmt.Errorf("integer constant: value overflows i8")
	}
	if typ.ElementType == "i16" && (value < math.MinInt16 || value > math.MaxInt16) {
		return Value{}, fmt.Errorf("integer constant: value overflows i16")
	}
	if typ.ElementType == "i32" && (value < math.MinInt32 || value > math.MaxInt32) {
		return Value{}, fmt.Errorf("integer constant: value overflows i32")
	}
	if typ.ElementType == "ui16" && (value < 0 || value > math.MaxUint16) {
		return Value{}, fmt.Errorf("integer constant: value overflows ui16")
	}
	for _, n := range typ.Shape {
		if n < 0 {
			return Value{}, fmt.Errorf("integer constant: static shape required")
		}
	}
	if _, _, err := elements(typ.Shape); err != nil {
		return Value{}, fmt.Errorf("integer constant: %w", err)
	}
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(typ.Shape), ElementType: typ.ElementType}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%d> : %s", out.ID, value, out.Type))
	return out, nil
}

// DynamicUpdateSlice replaces a slice of operand with update. Both shapes must
// be static, have equal ranks and element types, and update must fit operand.
// starts contains one scalar integer value per dimension, all of the same type.
// StableHLO clamps each start to [0, operandSize-updateSize]; callers that
// need rejection must validate runtime positions before execution.
// An error leaves the builder unchanged.
func (b *FuncBuilder) DynamicUpdateSlice(operand, update Value, starts []Value) (Value, error) {
	if err := sameElements(operand.Type, update.Type); err != nil {
		return Value{}, fmt.Errorf("dynamic update slice: %w", err)
	}
	rank := len(operand.Type.Shape)
	if len(update.Type.Shape) != rank || len(starts) != rank {
		return Value{}, fmt.Errorf("dynamic update slice: ranks and index count must match")
	}
	var indexType string
	for i, n := range operand.Type.Shape {
		if n < 0 || update.Type.Shape[i] < 0 || update.Type.Shape[i] > n {
			return Value{}, fmt.Errorf("dynamic update slice: update must fit static operand")
		}
		typ := starts[i].Type
		if len(typ.Shape) != 0 || !integerIndexType(typ.ElementType) {
			return Value{}, fmt.Errorf("dynamic update slice: scalar integer indices required")
		}
		if i > 0 && indexType != typ.ElementType {
			return Value{}, fmt.Errorf("dynamic update slice: index types differ")
		}
		indexType = typ.ElementType
	}
	return b.dynamicUpdateSlice(operand, update, starts), nil
}

func (b *FuncBuilder) dynamicUpdateSlice(operand, update Value, starts []Value) Value {
	values := []string{operand.ID, update.ID}
	types := []string{operand.Type.String(), update.Type.String()}
	for _, v := range starts {
		values = append(values, v.ID)
		types = append(types, v.Type.String())
	}
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(operand.Type.Shape), ElementType: operand.Type.ElementType}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.dynamic_update_slice\"(%s) : (%s) -> %s", out.ID, strings.Join(values, ", "), strings.Join(types, ", "), out.Type))
	return out
}

// CausalMaskAt returns a [queryLength, keyLength] boolean mask with a runtime
// scalar i64 offset. Entry [q,k] is true when k-q <= offset. This formulation
// avoids signed overflow for extreme offsets. Negative offsets mask initial
// queries; cache callers should validate their own position bounds.
// Lengths must be positive. An error leaves the builder unchanged.
func (b *FuncBuilder) CausalMaskAt(queryLength, keyLength int64, offset Value) (Value, error) {
	if queryLength <= 0 || keyLength <= 0 || queryLength > math.MaxInt64/keyLength {
		return Value{}, fmt.Errorf("causal mask: invalid lengths")
	}
	if offset.Type.ElementType != "i64" || len(offset.Type.Shape) != 0 {
		return Value{}, fmt.Errorf("causal mask: scalar i64 offset required")
	}
	return b.causalMaskAt(queryLength, keyLength, offset), nil
}

func (b *FuncBuilder) causalMaskAt(queryLength, keyLength int64, offset Value) Value {
	typ := Type{Shape: []int64{queryLength, keyLength}, ElementType: "i64"}
	iota := func(axis int) Value {
		v := Value{ID: b.next(), Type: typ}
		b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.iota\"() {iota_dimension = %d : i64} : () -> %s", v.ID, axis, typ))
		return v
	}
	q, k := iota(0), iota(1)
	delta := Value{ID: b.next(), Type: typ}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.subtract %s, %s : %s", delta.ID, k.ID, q.ID, typ))
	expanded := Value{ID: b.next(), Type: typ}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s", expanded.ID, offset.ID, offset.Type, typ))
	return b.compare(delta, expanded, "LE", "SIGNED")
}
