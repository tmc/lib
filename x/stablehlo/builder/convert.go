package builder

import (
	"fmt"
	"slices"
)

// Convert converts between f16, bf16, f32, f64, i1, i8, i32, i64, ui8, ui32 and ui64 without changing shape.
// An error leaves the builder unchanged.
func (b *FuncBuilder) Convert(v Value, elementType string) (Value, error) {
	if err := checkType(v.Type); err != nil {
		return Value{}, fmt.Errorf("convert: %w", err)
	}
	if !convertType(v.Type.ElementType) || !convertType(elementType) {
		return Value{}, fmt.Errorf("convert: unsupported element type")
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

func convertType(typ string) bool {
	return floatType(typ) || typ == "i1" || typ == "i8" || typ == "i32" || typ == "i64" || typ == "ui8" || typ == "ui32" || typ == "ui64"
}
