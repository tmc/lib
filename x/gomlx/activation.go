package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Logistic(v compute.Value) (compute.Value, error) {
	return f.unary(v, "logistic")
}

func (f *function) Erf(v compute.Value) (compute.Value, error) {
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
	z := hlo.Value{ID: fmt.Sprintf("%%erf%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "chlo.erf"(%s) : (%s) -> %s`, z.ID, x.ID, x.Type, z.Type))
	return &node{f, z}, nil
}
