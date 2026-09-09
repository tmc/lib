package gomlx

import (
	"fmt"
	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Reverse(input compute.Value, axes ...int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(input)
	if err != nil {
		return nil, err
	}
	seen := make([]bool, s.Rank())
	for _, axis := range axes {
		if axis < 0 || axis >= len(seen) || seen[axis] {
			return nil, fmt.Errorf("reverse: invalid axis %d", axis)
		}
		seen[axis] = true
	}
	x, err := f.value(input)
	if err != nil {
		return nil, err
	}
	if len(axes) == 0 {
		return &node{f, x}, nil
	}
	z := hlo.Value{ID: fmt.Sprintf("%%reverse%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.reverse"(%s) {dimensions = array<i64: %s>} : (%s) -> %s`, z.ID, x.ID, indexValues(axes), x.Type, z.Type))
	return &node{f, z}, nil
}
