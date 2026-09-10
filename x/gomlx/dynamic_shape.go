package gomlx

import (
	"fmt"
	"math"
	"slices"

	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) valueShape(v hlo.Value) shapes.Shape {
	if s, ok := f.valueShapes[v.ID]; ok {
		return s
	}
	return shapes.Shape{DType: elementDType(v.Type.ElementType), Dimensions: ints(v.Type.Shape)}
}

// broadcastShape aligns trailing axes, preserving the name of the axis that
// survives broadcasting. Two non-unit dynamic axes must have the same name.
func broadcastShape(a, b shapes.Shape) (shapes.Shape, error) {
	if a.DType != b.DType {
		return shapes.Shape{}, fmt.Errorf("operand dtype mismatch")
	}
	rank := max(a.Rank(), b.Rank())
	dims, names := make([]int, rank), make([]string, rank)
	for axis := range rank {
		x, y := 1, 1
		an, bn := "", ""
		if i := axis - rank + a.Rank(); i >= 0 {
			x, an = a.Dimensions[i], a.AxisName(i)
		}
		if i := axis - rank + b.Rank(); i >= 0 {
			y, bn = b.Dimensions[i], b.AxisName(i)
		}
		switch {
		case x == 1:
			dims[axis], names[axis] = y, bn
		case y == 1:
			dims[axis], names[axis] = x, an
		case x != y:
			return shapes.Shape{}, fmt.Errorf("incompatible broadcast dimensions")
		default:
			if x == shapes.DynamicDim && (an == "" || !shapes.AxisNameEqual(an, bn)) {
				return shapes.Shape{}, fmt.Errorf("incompatible dynamic broadcast names %q and %q", an, bn)
			}
			name, err := shapes.UnifyAxisName(an, bn)
			if err != nil {
				return shapes.Shape{}, err
			}
			dims[axis], names[axis] = x, name
		}
	}
	if !slices.ContainsFunc(names, func(s string) bool { return s != "" }) {
		names = nil
	}
	return shapes.Shape{DType: a.DType, Dimensions: dims, AxisNames: names}, nil
}

func sliceShape(input shapes.Shape, starts, limits, strides []int) (shapes.Shape, []int, error) {
	rank := input.Rank()
	if len(starts) != rank || len(limits) != rank || len(strides) != rank {
		return shapes.Shape{}, nil, fmt.Errorf("slice: bounds and strides must match rank")
	}
	out := input.Clone()
	ends := slices.Clone(limits)
	for i, n := range input.Dimensions {
		start, end, step := starts[i], limits[i], strides[i]
		if start < 0 || step <= 0 {
			return shapes.Shape{}, nil, fmt.Errorf("slice: invalid start or stride at axis %d", i)
		}
		if n == shapes.DynamicDim && end == shapes.DynamicDim {
			if start != 0 || step != 1 {
				return shapes.Shape{}, nil, fmt.Errorf("slice: dynamic full axis requires zero start and unit stride")
			}
			continue
		}
		if end == shapes.DynamicDim {
			end = n
		}
		if end < start || n >= 0 && end > n {
			return shapes.Shape{}, nil, fmt.Errorf("slice: invalid bounds at axis %d", i)
		}
		ends[i] = end
		out.Dimensions[i] = 0
		if end > start {
			out.Dimensions[i] = 1 + (end-start-1)/step
		}
		if n == shapes.DynamicDim && out.AxisNames != nil {
			out.AxisNames[i] = ""
		}
	}
	return out, ends, nil
}

func concatenateType(values []hlo.Value, axis int) (hlo.Type, error) {
	typ := values[0].Type
	typ.Shape = slices.Clone(typ.Shape)
	typ.Shape[axis] = 0
	for _, v := range values {
		n := v.Type.Shape[axis]
		if n < 0 || typ.Shape[axis] < 0 {
			typ.Shape[axis] = -1
			continue
		}
		if n > math.MaxInt64-typ.Shape[axis] {
			return hlo.Type{}, fmt.Errorf("concatenate: dimension overflow")
		}
		typ.Shape[axis] += n
	}
	return typ, nil
}
