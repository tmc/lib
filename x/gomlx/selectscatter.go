package gomlx

import (
	"fmt"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
	"strings"
)

func (f *function) SelectAndScatterMax(input, source compute.Value, windows, strides []int, padding [][2]int) (compute.Value, error) {
	return f.selectScatter(input, source, windows, strides, padding, "GE")
}
func (f *function) SelectAndScatterMin(input, source compute.Value, windows, strides []int, padding [][2]int) (compute.Value, error) {
	return f.selectScatter(input, source, windows, strides, padding, "LE")
}

func (f *function) selectScatter(input, source compute.Value, windows, strides []int, padding [][2]int, direction string) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	s, err := f.Shape(input)
	if err != nil {
		return nil, err
	}
	ss, err := f.Shape(source)
	if err != nil {
		return nil, err
	}
	if s.DType == dtypes.Bool {
		return nil, unsupported
	}
	if _, err := shapeinference.SelectAndScatter(s, ss, windows, strides, padding); err != nil {
		return nil, err
	}
	x, err := f.value(input)
	if err != nil {
		return nil, err
	}
	src, err := f.value(source)
	if err != nil {
		return nil, err
	}
	if s.Rank() == 0 {
		return &node{f, src}, nil
	}
	scalar := hlo.Type{ElementType: x.Type.ElementType}
	initial := hlo.Value{ID: fmt.Sprintf("%%selectinit%d", len(f.raw.Ops)), Type: scalar}
	literal := "0"
	kind := "SIGNED"
	if s.DType.IsFloat() {
		literal = "0.0"
		kind = "FLOAT"
	} else if strings.HasPrefix(x.Type.ElementType, "ui") {
		kind = "UNSIGNED"
	}
	wd, st := make([]int, s.Rank()), make([]int, s.Rank())
	pad := make([]string, s.Rank())
	for i := range wd {
		wd[i] = 1
		if len(windows) > 0 {
			wd[i] = windows[i]
		}
		st[i] = wd[i]
		if len(strides) > 0 {
			st[i] = strides[i]
		}
		lo, hi := 0, 0
		if len(padding) > 0 {
			lo, hi = padding[i][0], padding[i][1]
		}
		pad[i] = fmt.Sprintf("[%d, %d]", lo, hi)
	}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", initial.ID, literal, scalar))
	z := hlo.Value{ID: fmt.Sprintf("%%selectscatter%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.select_and_scatter"(%s, %s, %s) ({
 ^bb0(%%lhs: %s, %%rhs: %s):
 %%selected = "stablehlo.compare"(%%lhs, %%rhs) {comparison_direction = #stablehlo<comparison_direction %s>, compare_type = #stablehlo<comparison_type %s>} : (%s, %s) -> tensor<i1>
 stablehlo.return %%selected : tensor<i1>
 }, {
 ^bb0(%%old: %s, %%update: %s):
 %%sum = stablehlo.add %%old, %%update : %s
 stablehlo.return %%sum : %s
 }) {window_dimensions = array<i64: %s>, window_strides = array<i64: %s>, padding = dense<[%s]> : tensor<%dx2xi64>} : (%s, %s, %s) -> %s`, z.ID, x.ID, src.ID, initial.ID, scalar, scalar, direction, kind, scalar, scalar, scalar, scalar, scalar, scalar, indexValues(wd), indexValues(st), strings.Join(pad, ", "), s.Rank(), x.Type, src.Type, scalar, z.Type))
	return &node{f, z}, nil
}
