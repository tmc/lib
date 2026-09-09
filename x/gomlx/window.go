package gomlx

import (
	"fmt"
	"math"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) ReduceWindow(input compute.Value, reduction compute.ReduceOpType, windows, strides, inputDilations, windowDilations []int, paddings [][2]int) (compute.Value, error) {
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
	op := ""
	switch reduction {
	case compute.ReduceOpSum:
		op = "add"
	case compute.ReduceOpProduct:
		op = "multiply"
	case compute.ReduceOpMax:
		op = "maximum"
	case compute.ReduceOpMin:
		op = "minimum"
	default:
		return nil, fmt.Errorf("invalid window reduction")
	}
	out, err := shapeinference.ReduceWindow(s, windows, strides, inputDilations, windowDilations, paddings)
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
	// Scalar windows have no spatial dimensions.
	if s.Rank() == 0 {
		return &node{f, x}, nil
	}
	scalar := hlo.Type{ElementType: x.Type.ElementType}
	literal := "0"
	if reduction == compute.ReduceOpProduct {
		literal = "1"
	}
	var initial hlo.Value
	if s.DType.IsFloat() {
		v := float32(0)
		switch reduction {
		case compute.ReduceOpProduct:
			v = 1
		case compute.ReduceOpMax:
			v = float32(math.Inf(-1))
		case compute.ReduceOpMin:
			v = float32(math.Inf(1))
		}
		initial, err = f.raw.Constant(v, scalar)
		if err != nil {
			return nil, err
		}
	} else {
		bits := 64
		switch s.DType {
		case dtypes.Int8, dtypes.Uint8:
			bits = 8
		case dtypes.Int32, dtypes.Uint32:
			bits = 32
		}
		unsigned := strings.HasPrefix(x.Type.ElementType, "ui")
		if reduction == compute.ReduceOpMin {
			if unsigned {
				literal = fmt.Sprint(uint64(math.MaxUint64) >> (64 - bits))
			} else {
				literal = fmt.Sprint(int64(math.MaxInt64) >> (64 - bits))
			}
		} else if reduction == compute.ReduceOpMax && !unsigned {
			literal = fmt.Sprint(int64(-1) << (bits - 1))
		}
		initial = hlo.Value{ID: fmt.Sprintf("%%windowinit%d", len(f.raw.Ops)), Type: scalar}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", initial.ID, literal, scalar))
	}
	wd, st, bd, kd := make([]int, s.Rank()), make([]int, s.Rank()), make([]int, s.Rank()), make([]int, s.Rank())
	pad := make([]string, s.Rank())
	for i := range wd {
		wd[i], bd[i], kd[i] = 1, 1, 1
		if len(windows) > 0 {
			wd[i] = windows[i]
		}
		st[i] = wd[i]
		if len(strides) > 0 {
			st[i] = strides[i]
		}
		if inputDilations != nil {
			bd[i] = inputDilations[i]
		}
		if windowDilations != nil {
			kd[i] = windowDilations[i]
		}
		lo, hi := 0, 0
		if len(paddings) > 0 {
			lo, hi = paddings[i][0], paddings[i][1]
		}
		pad[i] = fmt.Sprintf("[%d, %d]", lo, hi)
	}
	z := hlo.Value{ID: fmt.Sprintf("%%window%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(out.Dimensions), ElementType: x.Type.ElementType}}
	code := fmt.Sprintf(`%s = "stablehlo.reduce_window"(%s, %s) ({
 ^bb0(%%lhs: %s, %%rhs: %s):
 %%result = stablehlo.%s %%lhs, %%rhs : %s
 stablehlo.return %%result : %s
 }) {window_dimensions = array<i64: %s>, window_strides = array<i64: %s>, base_dilations = array<i64: %s>, window_dilations = array<i64: %s>, padding = dense<[%s]> : tensor<%dx2xi64>} : (%s, %s) -> %s`, z.ID, x.ID, initial.ID, scalar, scalar, op, scalar, scalar, indexValues(wd), indexValues(st), indexValues(bd), indexValues(kd), strings.Join(pad, ", "), s.Rank(), x.Type, scalar, z.Type)
	f.raw.Ops = append(f.raw.Ops, code)
	return &node{f, z}, nil
}
