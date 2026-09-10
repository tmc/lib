package builder

import (
	"slices"
	"testing"
)

func TestComplex(t *testing.T) {
	for _, element := range []string{"f32", "f64"} {
		t.Run(element, func(t *testing.T) {
			b := NewFunc("main", Type{Shape: []int64{2}, ElementType: element}, Type{Shape: []int64{2}, ElementType: element})
			z, err := b.Complex(b.Arg(0), b.Arg(1))
			if err != nil {
				t.Fatal(err)
			}
			r, err := b.Real(z)
			if err != nil {
				t.Fatal(err)
			}
			i, err := b.Imag(z)
			if err != nil {
				t.Fatal(err)
			}
			c, err := b.Conj(z)
			if err != nil {
				t.Fatal(err)
			}
			if r.Type.ElementType != element || i.Type.ElementType != element || c.Type.ElementType != "complex<"+element+">" || !slices.Equal(c.Type.Shape, []int64{2}) {
				t.Fatalf("unexpected result types: %v %v %v", r.Type, i.Type, c.Type)
			}
		})
	}
}

func TestComplexValidation(t *testing.T) {
	for _, tt := range []struct {
		name string
		a, b Type
	}{
		{"half", Type{ElementType: "f16"}, Type{ElementType: "f16"}},
		{"integer", Type{ElementType: "i32"}, Type{ElementType: "i32"}},
		{"different", Type{ElementType: "f32"}, Type{ElementType: "f64"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := NewFunc("main", tt.a, tt.b)
			if _, err := b.Complex(b.Arg(0), b.Arg(1)); err == nil {
				t.Fatal("accepted invalid complex inputs")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatal("invalid complex changed builder")
			}
		})
	}
	b := NewFunc("main", Type{ElementType: "f32"})
	for _, op := range []func(Value) (Value, error){b.Real, b.Imag, b.Conj} {
		if _, err := op(b.Arg(0)); err == nil {
			t.Fatal("accepted non-complex input")
		}
	}
}
