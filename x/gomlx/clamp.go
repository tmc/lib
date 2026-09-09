package gomlx

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Clamp(min, input, max compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(input)
	if err != nil {
		return nil, err
	}
	if s.DType == dtypes.Bool {
		return nil, unsupported
	}
	for _, bound := range []compute.Value{min, max} {
		bs, err := f.Shape(bound)
		if err != nil {
			return nil, err
		}
		if bs.DType != s.DType || bs.Rank() != 0 && !slices.Equal(bs.Dimensions, s.Dimensions) {
			return nil, fmt.Errorf("clamp: bounds must be scalar or match the input shape and dtype")
		}
	}
	lo, err := f.value(min)
	if err != nil {
		return nil, err
	}
	x, err := f.value(input)
	if err != nil {
		return nil, err
	}
	hi, err := f.value(max)
	if err != nil {
		return nil, err
	}
	z := hlo.Value{ID: fmt.Sprintf("%%clamp%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.clamp\"(%s, %s, %s) : (%s, %s, %s) -> %s", z.ID, lo.ID, x.ID, hi.ID, lo.Type, x.Type, hi.Type, z.Type))
	return &node{f, z}, nil
}
