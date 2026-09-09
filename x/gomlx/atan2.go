package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

// Atan2 returns element-wise the arc tangent of y/x (lhs is y, rhs is x).
func (f *function) Atan2(lhs, rhs compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	sLHS, err := f.Shape(lhs)
	if err != nil {
		return nil, err
	}
	sRHS, err := f.Shape(rhs)
	if err != nil {
		return nil, err
	}
	if !sLHS.DType.IsFloat() || !sRHS.DType.IsFloat() {
		return nil, unsupported
	}
	if sLHS.DType != sRHS.DType {
		return nil, fmt.Errorf("atan2 operand dtypes mismatch: %v vs %v", sLHS.DType, sRHS.DType)
	}
	return f.binary(lhs, rhs, func(y, x hlo.Value) (hlo.Value, error) {
		z := hlo.Value{ID: fmt.Sprintf("%%atan2_%d", len(f.raw.Ops)), Type: y.Type}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.atan2 %s, %s : %s", z.ID, y.ID, x.ID, y.Type))
		return z, nil
	})
}
