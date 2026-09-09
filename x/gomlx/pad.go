package gomlx

import (
	"fmt"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Pad(input, fill compute.Value, config ...compute.PadAxis) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(input)
	if err != nil {
		return nil, err
	}
	fs, err := f.Shape(fill)
	if err != nil {
		return nil, err
	}
	if fs.Rank() != 0 || fs.DType != s.DType {
		return nil, fmt.Errorf("pad: fill must be a scalar with the input dtype")
	}
	out, err := shapeinference.Pad(s, config...)
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
	v, err := f.value(fill)
	if err != nil {
		return nil, err
	}
	if s.Rank() == 0 {
		return &node{f, x}, nil
	}
	low, high, interior := make([]int, s.Rank()), make([]int, s.Rank()), make([]int, s.Rank())
	for i, c := range config {
		low[i], high[i], interior[i] = c.Start, c.End, c.Interior
	}
	z := hlo.Value{ID: fmt.Sprintf("%%pad%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(out.Dimensions), ElementType: x.Type.ElementType}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.pad"(%s, %s) {edge_padding_low = array<i64: %s>, edge_padding_high = array<i64: %s>, interior_padding = array<i64: %s>} : (%s, %s) -> %s`, z.ID, x.ID, v.ID, indexValues(low), indexValues(high), indexValues(interior), x.Type, v.Type, z.Type))
	return &node{f, z}, nil
}
