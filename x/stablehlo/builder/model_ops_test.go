package builder

import (
	"math"
	"testing"
)

func TestModelOpsValidation(t *testing.T) {
	f := func(shape ...int64) Type { return Type{Shape: shape, ElementType: "f32"} }
	i := func(shape ...int64) Type { return Type{Shape: shape, ElementType: "i64"} }
	for _, tt := range []struct {
		name  string
		types []Type
		run   func(*FuncBuilder) (Value, error)
	}{
		{"gather scalar operand", []Type{f(), i()}, func(b *FuncBuilder) (Value, error) { return b.GatherRows(b.Arg(0), b.Arg(1)) }},
		{"gather empty rows", []Type{f(0, 2), i(1)}, func(b *FuncBuilder) (Value, error) { return b.GatherRows(b.Arg(0), b.Arg(1)) }},
		{"gather float indices", []Type{f(2, 2), f(1)}, func(b *FuncBuilder) (Value, error) { return b.GatherRows(b.Arg(0), b.Arg(1)) }},
		{"gather dynamic indices", []Type{f(2, 2), i(-1)}, func(b *FuncBuilder) (Value, error) { return b.GatherRows(b.Arg(0), b.Arg(1)) }},
		{"gather dynamic empty", []Type{f(2, 0, -1), i(1)}, func(b *FuncBuilder) (Value, error) { return b.GatherRows(b.Arg(0), b.Arg(1)) }},
		{"gather output overflow", []Type{f(1, 2), i(math.MaxInt64)}, func(b *FuncBuilder) (Value, error) { return b.GatherRows(b.Arg(0), b.Arg(1)) }},
		{"silu type", []Type{i(2)}, func(b *FuncBuilder) (Value, error) { return b.SiLU(b.Arg(0)) }},
		{"silu dynamic", []Type{f(-1)}, func(b *FuncBuilder) (Value, error) { return b.SiLU(b.Arg(0)) }},
		{"silu overflow", []Type{f(math.MaxInt64, 2)}, func(b *FuncBuilder) (Value, error) { return b.SiLU(b.Arg(0)) }},
		{"swiglu shape", []Type{f(2), f(3)}, func(b *FuncBuilder) (Value, error) { return b.SwiGLU(b.Arg(0), b.Arg(1)) }},
		{"swiglu type", []Type{f(2), i(2)}, func(b *FuncBuilder) (Value, error) { return b.SwiGLU(b.Arg(0), b.Arg(1)) }},
		{"rotary scalar", []Type{f(), f(1), f(1)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary type", []Type{i(4), f(2), f(2)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary table type", []Type{f(4), i(2), i(2)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary table mismatch", []Type{f(4), f(2), f(1)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary table rank", []Type{f(4), f(1, 2), f(1, 2)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary empty", []Type{f(4), f(0), f(0)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary too wide", []Type{f(3), f(2), f(2)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary broadcast", []Type{f(2, 4), f(3, 2), f(3, 2)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
		{"rotary dynamic", []Type{f(-1, 4), f(2), f(2)}, func(b *FuncBuilder) (Value, error) { return b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2)) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := NewFunc("main", tt.types...)
			if _, err := tt.run(b); err == nil {
				t.Fatal("accepted invalid operation")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatal("error changed builder")
			}
		})
	}
}
