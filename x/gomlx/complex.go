package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Complex(real, imag compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(real)
	if err != nil {
		return nil, err
	}
	y, err := f.value(imag)
	if err != nil {
		return nil, err
	}
	if x.Type.ElementType != y.Type.ElementType || x.Type.ElementType != "f32" && x.Type.ElementType != "f64" {
		return nil, fmt.Errorf("complex: matching f32 or f64 inputs required")
	}
	x, y, err = f.broadcastPair(x, y)
	if err != nil {
		return nil, err
	}
	z, err := f.raw.Complex(x, y)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}

func (f *function) Conj(v compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if !elementDType(x.Type.ElementType).IsComplex() {
		return nil, fmt.Errorf("conj: complex input required")
	}
	z, err := f.raw.Conj(x)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}

func (f *function) Real(v compute.Value) (compute.Value, error) {
	return f.realImag(v, false)
}

func (f *function) Imag(v compute.Value) (compute.Value, error) {
	return f.realImag(v, true)
}

func (f *function) realImag(v compute.Value, imag bool) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if !elementDType(x.Type.ElementType).IsComplex() {
		if imag {
			return nil, fmt.Errorf("imag: complex input required")
		}
		return nil, fmt.Errorf("real: complex input required")
	}
	var z hlo.Value
	if imag {
		z, err = f.raw.Imag(x)
	} else {
		z, err = f.raw.Real(x)
	}
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
