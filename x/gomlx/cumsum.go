package gomlx

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) CumSum(input compute.Value, axis int, options compute.CumSumOptions) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(input)
	if err != nil {
		return nil, err
	}
	if _, err := shapeinference.CumSum(s, axis); err != nil {
		return nil, err
	}
	x, err := f.value(input)
	if err != nil {
		return nil, err
	}
	scalar := hlo.Type{ElementType: x.Type.ElementType}
	initial := hlo.Value{ID: fmt.Sprintf("%%scaninit%d", len(f.raw.Ops)), Type: scalar}
	literal := "0"
	if s.DType.IsFloat() {
		literal = "0.0"
	}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", initial.ID, literal, scalar))
	windows := make([]int, s.Rank())
	padding := make([]string, s.Rank())
	for i := range windows {
		windows[i] = 1
		padding[i] = "[0, 0]"
	}
	windows[axis] = s.Dimensions[axis]
	low, high := windows[axis]-1, 0
	if options.Exclusive {
		low++
		high--
	}
	if options.Reverse {
		low, high = high, low
	}
	padding[axis] = fmt.Sprintf("[%d, %d]", low, high)
	z := hlo.Value{ID: fmt.Sprintf("%%scan%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.reduce_window"(%s, %s) ({
 ^bb0(%%lhs: %s, %%rhs: %s):
 %%sum = stablehlo.add %%lhs, %%rhs : %s
 stablehlo.return %%sum : %s
 }) {window_dimensions = array<i64: %s>, padding = dense<[%s]> : tensor<%dx2xi64>} : (%s, %s) -> %s`, z.ID, x.ID, initial.ID, scalar, scalar, scalar, scalar, indexValues(windows), strings.Join(padding, ", "), s.Rank(), x.Type, scalar, z.Type))
	return &node{f, z}, nil
}
