package builder

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// Complex combines matching f32 or f64 real and imaginary tensors.
func (b *FuncBuilder) Complex(real, imag Value) (Value, error) {
	if err := sameElements(real.Type, imag.Type); err != nil {
		return Value{}, fmt.Errorf("complex: %w", err)
	}
	if real.Type.ElementType != "f32" && real.Type.ElementType != "f64" {
		return Value{}, fmt.Errorf("complex: f32 or f64 elements required")
	}
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(real.Type.Shape), ElementType: "complex<" + real.Type.ElementType + ">"}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.complex %s, %s : (%s, %s) -> %s", out.ID, real.ID, imag.ID, real.Type, imag.Type, out.Type))
	return out, nil
}

// Real returns the real component of a complex tensor.
func (b *FuncBuilder) Real(v Value) (Value, error) {
	typ, ok := realElementType(v.Type.ElementType)
	if !ok {
		return Value{}, fmt.Errorf("real: complex elements required")
	}
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(v.Type.Shape), ElementType: typ}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.real %s : (%s) -> %s", out.ID, v.ID, v.Type, out.Type))
	return out, nil
}

// Imag returns the imaginary component of a complex tensor.
func (b *FuncBuilder) Imag(v Value) (Value, error) {
	typ, ok := realElementType(v.Type.ElementType)
	if !ok {
		return Value{}, fmt.Errorf("imag: complex elements required")
	}
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(v.Type.Shape), ElementType: typ}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.imag %s : (%s) -> %s", out.ID, v.ID, v.Type, out.Type))
	return out, nil
}

// Conj returns the conjugate of a complex tensor.
func (b *FuncBuilder) Conj(v Value) (Value, error) {
	real, err := b.Real(v)
	if err != nil {
		return Value{}, err
	}
	imag, err := b.Imag(v)
	if err != nil {
		return Value{}, err
	}
	neg := Value{ID: b.next(), Type: imag.Type}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.negate %s : %s", neg.ID, imag.ID, imag.Type))
	return b.Complex(real, neg)
}

// ComplexConstant emits a complex splat constant with a static shape.
func (b *FuncBuilder) ComplexConstant(real, imag float64, typ Type) (Value, error) {
	if _, ok := realElementType(typ.ElementType); !ok {
		return Value{}, fmt.Errorf("complex constant: complex<f32> or complex<f64> elements required")
	}
	for _, d := range typ.Shape {
		if d < 0 {
			return Value{}, fmt.Errorf("complex constant: static shape required")
		}
	}
	bits := 64
	if typ.ElementType == "complex<f32>" {
		bits = 32
	}
	literal := fmt.Sprintf("(%s, %s)", complexLiteral(real, bits), complexLiteral(imag, bits))
	out := Value{ID: b.next(), Type: Type{Shape: slices.Clone(typ.Shape), ElementType: typ.ElementType}}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", out.ID, literal, out.Type))
	return out, nil
}

func realElementType(typ string) (string, bool) {
	switch typ {
	case "complex<f32>":
		return "f32", true
	case "complex<f64>":
		return "f64", true
	}
	return "", false
}

func complexLiteral(x float64, bits int) string {
	if bits == 32 {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return fmt.Sprintf("0x%08X", math.Float32bits(float32(x)))
		}
		return complexDecimalLiteral(fmt.Sprintf("%g", float32(x)))
	}
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return fmt.Sprintf("0x%016X", math.Float64bits(x))
	}
	return complexDecimalLiteral(fmt.Sprintf("%g", x))
}

func complexDecimalLiteral(s string) string {
	if !strings.ContainsAny(s, ".eE") {
		return s + ".0"
	}
	return s
}
