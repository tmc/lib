package builder

import (
	"fmt"
	"slices"
)

// Convert converts between f16, bf16, f32 and f64 without changing shape.
// An error leaves the builder unchanged.
func (b *FuncBuilder) Convert(v Value, elementType string) (Value, error) {
	if err := checkType(v.Type); err != nil {
		return Value{}, fmt.Errorf("convert: %w", err)
	}
	if !floatType(v.Type.ElementType) || !floatType(elementType) {
		return Value{}, fmt.Errorf("convert: floating-point elements required")
	}
	return b.convert(v, elementType), nil
}

func (b *FuncBuilder) convert(v Value, elementType string) Value {
	if v.Type.ElementType == elementType {
		return v
	}
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(v.Type.Shape), ElementType: elementType}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.convert %s : (%s) -> %s", out.ID, v.ID, v.Type, out.Type))
	return out
}
