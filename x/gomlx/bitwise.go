package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) BitwiseAnd(a, b compute.Value) (compute.Value, error) {
	return f.bitwise(a, b, "and", false)
}
func (f *function) BitwiseOr(a, b compute.Value) (compute.Value, error) {
	return f.bitwise(a, b, "or", false)
}
func (f *function) BitwiseXor(a, b compute.Value) (compute.Value, error) {
	return f.bitwise(a, b, "xor", false)
}
func (f *function) LogicalXor(a, b compute.Value) (compute.Value, error) {
	return f.bitwise(a, b, "xor", true)
}

func (f *function) bitwise(a, b compute.Value, op string, logical bool) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	for _, v := range []compute.Value{a, b} {
		s, err := f.Shape(v)
		if err != nil {
			return nil, err
		}
		if logical && s.DType != dtypes.Bool || !logical && !s.DType.IsInt() {
			return nil, unsupported
		}
	}
	return f.binary(a, b, func(x, y hlo.Value) (hlo.Value, error) {
		z := hlo.Value{ID: fmt.Sprintf("%%bits%d", len(f.raw.Ops)), Type: x.Type}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.%s %s, %s : %s", z.ID, op, x.ID, y.ID, x.Type))
		return z, nil
	})
}

func (f *function) BitCount(v compute.Value) (compute.Value, error) { return f.countBits(v, "popcnt") }
func (f *function) Clz(v compute.Value) (compute.Value, error) {
	return f.countBits(v, "count_leading_zeros")
}
func (f *function) countBits(v compute.Value, op string) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if !elementDType(x.Type.ElementType).IsInt() {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%bits%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.%s %s : %s", z.ID, op, x.ID, x.Type))
	return &node{f, z}, nil
}
