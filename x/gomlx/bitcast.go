package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Bitcast(input compute.Value, dtype dtypes.DType) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	info, ok := element(dtype)
	if !ok {
		return nil, unsupported
	}
	s, err := f.Shape(input)
	if err != nil {
		return nil, err
	}
	// The upstream inference helper indexes the last axis when widening.
	if s.Rank() == 0 && dtype.Bits() > s.DType.Bits() {
		return nil, fmt.Errorf("bitcast: widening requires a trailing axis")
	}
	out, err := shapeinference.Bitcast(s, dtype)
	if err != nil {
		return nil, err
	}
	if err := validate(out); err != nil {
		return nil, err
	}
	x, err := f.value(input)
	if err != nil {
		return nil, err
	}
	if s.DType == dtype {
		return &node{f, x}, nil
	}
	// StableHLO boolean (i1) is 1-bit, while GoMLX dtypes.Bool is treated as 8-bit in compute.
	// Cross-type bitcast between Bool and other dtypes is not natively supported by the PJRT plugin.
	if s.DType == dtypes.Bool || dtype == dtypes.Bool {
		return nil, fmt.Errorf("bitcast: boolean bitcast conversion is unsupported: %w", unsupported)
	}
	z := hlo.Value{ID: fmt.Sprintf("%%bitcast%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(out.Dimensions), ElementType: info.hlo}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.bitcast_convert"(%s) : (%s) -> %s`, z.ID, x.ID, x.Type, z.Type))
	return &node{f, z}, nil
}
