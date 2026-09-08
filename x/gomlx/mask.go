package gomlx

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Equal(a, b compute.Value) (compute.Value, error) {
	return f.compare(a, b, "EQ")
}
func (f *function) LessThan(a, b compute.Value) (compute.Value, error) {
	return f.compare(a, b, "LT")
}
func (f *function) compare(a, b compute.Value, direction string) (compute.Value, error) {
	return f.binary(a, b, func(x, y hlo.Value) (hlo.Value, error) {
		kind := "SIGNED"
		if x.Type.ElementType == "f32" {
			kind = "FLOAT"
		}
		if x.Type.ElementType == "i1" || x.Type.ElementType == "ui8" || x.Type.ElementType == "ui32" || x.Type.ElementType == "ui64" {
			kind = "UNSIGNED"
		}
		z := hlo.Value{ID: fmt.Sprintf("%%compare%d", len(f.raw.Ops)), Type: hlo.Type{Shape: slices.Clone(x.Type.Shape), ElementType: "i1"}}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.compare\"(%s, %s) {comparison_direction = #stablehlo<comparison_direction %s>, compare_type = #stablehlo<comparison_type %s>} : (%s, %s) -> %s", z.ID, x.ID, y.ID, direction, kind, x.Type, y.Type, z.Type))
		return z, nil
	})
}
func (f *function) LogicalAnd(a, b compute.Value) (compute.Value, error) {
	return f.binary(a, b, f.raw.And)
}
func (f *function) Where(condition, onTrue, onFalse compute.Value) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	c, err := f.value(condition)
	if err != nil {
		return nil, err
	}
	x, err := f.value(onTrue)
	if err != nil {
		return nil, err
	}
	y, err := f.value(onFalse)
	if err != nil {
		return nil, err
	}
	if c.Type.ElementType != "i1" || x.Type.ElementType != y.Type.ElementType {
		return nil, fmt.Errorf("where: boolean condition and matching branch dtypes required")
	}
	var dims []int64
	for _, v := range []hlo.Value{c, x, y} {
		if len(v.Type.Shape) == 0 {
			continue
		}
		if dims != nil && !slices.Equal(dims, v.Type.Shape) {
			return nil, fmt.Errorf("where: non-scalar shapes must match")
		}
		dims = v.Type.Shape
	}
	values := []*hlo.Value{&c, &x, &y}
	for _, v := range values {
		if len(v.Type.Shape) == 0 && len(dims) > 0 {
			*v, err = f.raw.BroadcastInDim(*v, nil, hlo.Type{Shape: slices.Clone(dims), ElementType: v.Type.ElementType})
			if err != nil {
				return nil, err
			}
		}
	}
	z := hlo.Value{ID: fmt.Sprintf("%%select%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.select\"(%s, %s, %s) : (%s, %s, %s) -> %s", z.ID, c.ID, x.ID, y.ID, c.Type, x.Type, y.Type, z.Type))
	return &node{f, z}, nil
}

func (f *function) LessOrEqual(a, b compute.Value) (compute.Value, error) {
	return f.compare(a, b, "LE")
}
