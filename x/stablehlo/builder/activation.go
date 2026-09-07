package builder

import (
	"fmt"
	"slices"
)

// SiLU computes x/(1+exp(-x)) elementwise. It accepts static f32, f16 and bf16
// tensors. Reduced-precision inputs compute in f32 and convert once at the end.
// An error leaves the builder unchanged.
func (b *FuncBuilder) SiLU(x Value) (Value, error) {
	if err := activationType(x.Type); err != nil {
		return Value{}, fmt.Errorf("silu: %w", err)
	}
	out := b.silu(b.convert(x, "f32"))
	return b.convert(out, x.Type.ElementType), nil
}

// SwiGLU computes SiLU(gate)*up for equal-shaped f32, f16 or bf16 tensors.
// Shapes must be static. Reduced-precision inputs compute in f32, including the
// final multiply, and convert only the result. No physical fusion is promised.
// An error leaves the builder unchanged.
func (b *FuncBuilder) SwiGLU(gate, up Value) (Value, error) {
	if err := activationType(gate.Type); err != nil {
		return Value{}, fmt.Errorf("swiglu: %w", err)
	}
	if gate.Type.ElementType != up.Type.ElementType || !slices.Equal(gate.Type.Shape, up.Type.Shape) {
		return Value{}, fmt.Errorf("swiglu: matching shapes and element types required")
	}
	out := b.silu(b.convert(gate, "f32"))
	out, _ = b.Multiply(out, b.convert(up, "f32"))
	return b.convert(out, gate.Type.ElementType), nil
}

func activationType(typ Type) error {
	if typ.ElementType != "f32" && typ.ElementType != "f16" && typ.ElementType != "bf16" {
		return fmt.Errorf("f32, f16 or bf16 elements required")
	}
	if err := staticFloat(typ); err != nil {
		return err
	}
	_, _, err := elements(typ.Shape)
	return err
}

func (b *FuncBuilder) silu(x Value) Value {
	// Every intermediate has x's validated static f32 type. Using exp(-abs(x))
	// avoids overflow in the negative branch of the sigmoid.
	abs, _ := b.unaryOp("stablehlo.abs", x)
	neg, _ := b.unaryOp("stablehlo.negate", abs)
	e, _ := b.Exponential(neg)
	one, _ := b.Constant(1, x.Type)
	denom, _ := b.Add(one, e)
	positive, _ := b.Divide(one, denom)
	negative, _ := b.Divide(e, denom)
	zero, _ := b.Constant(0, x.Type)
	sigmoid := b.selectValue(b.compare(x, zero, "GE", "FLOAT"), positive, negative)
	out, _ := b.Multiply(x, sigmoid)
	return out
}
