package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Expm1(v compute.Value) (compute.Value, error) {
	return f.unaryFloat(v, "exponential_minus_one")
}
func (f *function) Log1p(v compute.Value) (compute.Value, error) {
	return f.unaryFloat(v, "log_plus_one")
}
func (f *function) Ceil(v compute.Value) (compute.Value, error)  { return f.unaryFloat(v, "ceil") }
func (f *function) Floor(v compute.Value) (compute.Value, error) { return f.unaryFloat(v, "floor") }
func (f *function) Round(v compute.Value) (compute.Value, error) {
	return f.unaryFloat(v, "round_nearest_even")
}
func (f *function) Tanh(v compute.Value) (compute.Value, error) { return f.unaryFloat(v, "tanh") }

func (f *function) Sign(v compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if x.Type.ElementType == "i1" {
		return nil, unsupported
	}

	if elementDType(x.Type.ElementType).IsUnsigned() {
		zero := hlo.Value{ID: fmt.Sprintf("%%signzero%d", len(f.raw.Ops)), Type: x.Type}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<0> : %s", zero.ID, zero.Type))
		test := hlo.Value{ID: fmt.Sprintf("%%signcmp%d", len(f.raw.Ops)), Type: hlo.Type{Shape: x.Type.Shape, ElementType: "i1"}}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.compare NE, %s, %s, UNSIGNED : (%s, %s) -> %s", test.ID, x.ID, zero.ID, x.Type, x.Type, test.Type))
		z, err := f.raw.Convert(test, x.Type.ElementType)
		if err != nil {
			return nil, err
		}
		return &node{f, z}, nil
	}
	z := hlo.Value{ID: fmt.Sprintf("%%sign%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.sign %s : %s", z.ID, x.ID, x.Type))
	return &node{f, z}, nil
}
func (f *function) IsFinite(v compute.Value) (compute.Value, error) { return f.classifyFloat(v, false) }
func (f *function) IsNaN(v compute.Value) (compute.Value, error)    { return f.classifyFloat(v, true) }
func (f *function) classifyFloat(v compute.Value, nan bool) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if !elementDType(x.Type.ElementType).IsFloat() {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%classify%d", len(f.raw.Ops)), Type: hlo.Type{Shape: x.Type.Shape, ElementType: "i1"}}
	code := fmt.Sprintf("%s = stablehlo.is_finite %s : (%s) -> %s", z.ID, x.ID, x.Type, z.Type)
	if nan {
		code = fmt.Sprintf("%s = stablehlo.compare NE, %s, %s, FLOAT : (%s, %s) -> %s", z.ID, x.ID, x.ID, x.Type, x.Type, z.Type)
	}
	f.raw.Ops = append(f.raw.Ops, code)
	return &node{f, z}, nil
}
