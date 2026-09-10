package gomlx

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Transpose(v compute.Value, permutation ...int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if len(permutation) != len(x.Type.Shape) {
		return nil, fmt.Errorf("transpose: permutation rank mismatch")
	}
	dims := make([]int64, len(permutation))
	seen := make([]bool, len(permutation))
	for i, axis := range permutation {
		if axis < 0 || axis >= len(dims) || seen[axis] {
			return nil, fmt.Errorf("transpose: invalid or repeated axis %d", axis)
		}
		seen[axis] = true
		dims[i] = x.Type.Shape[axis]
	}
	z, err := f.raw.Transpose(x, int64s(permutation), hlo.Type{Shape: dims, ElementType: x.Type.ElementType})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}

func (f *function) Iota(shape shapes.Shape, axis int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if err := validate(shape); err != nil {
		return nil, err
	}
	if shape.DType == dtypes.Bool {
		return nil, unsupported
	}
	if axis < 0 || axis >= len(shape.Dimensions) {
		return nil, fmt.Errorf("iota: invalid axis %d", axis)
	}
	info, _ := element(shape.DType)
	z := hlo.Value{ID: fmt.Sprintf("%%iota%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(shape.Dimensions), ElementType: info.hlo}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.iota\"() {iota_dimension = %d : i64} : () -> %s", z.ID, axis, z.Type))
	return &node{f, z}, nil
}

func (f *function) Slice(v compute.Value, starts, limits, strides []int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	shape := f.valueShape(x)
	steps := slices.Clone(strides)
	if len(steps) == 0 {
		steps = make([]int, shape.Rank())
		for i := range steps {
			steps[i] = 1
		}
	}
	out, ends, err := sliceShape(shape, starts, limits, steps)
	if err != nil {
		return nil, err
	}
	op := dynamicOp{kind: "slice", operand: x, starts: slices.Clone(starts), limits: slices.Clone(limits), strides: steps}
	if shape.IsDynamic() {
		return f.recordDynamic(op, out)
	}
	z, err := f.raw.Slice(x, int64s(starts), int64s(ends), int64s(steps), hlo.Type{Shape: int64s(out.Dimensions), ElementType: x.Type.ElementType})
	if err != nil {
		return nil, err
	}
	op.result = z
	f.dynamicOps = append(f.dynamicOps, op)
	f.valueShapes[z.ID] = out
	return &node{f, z}, nil
}

func (f *function) Concatenate(axis int, operands ...compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if len(operands) == 0 {
		return nil, fmt.Errorf("concatenate: no operands")
	}
	values := make([]hlo.Value, len(operands))
	ss := make([]shapes.Shape, len(operands))
	for i, v := range operands {
		x, err := f.value(v)
		if err != nil {
			return nil, err
		}
		values[i] = x
		ss[i] = f.valueShape(x)
	}
	if axis < 0 || axis >= ss[0].Rank() {
		return nil, fmt.Errorf("concatenate: invalid axis %d", axis)
	}
	for _, s := range ss {
		if s.Rank() != ss[0].Rank() {
			return nil, fmt.Errorf("concatenate: rank mismatch")
		}
	}
	// Check accumulation before calling upstream shape inference, which uses int sums.
	typ, err := concatenateType(values, axis)
	if err != nil {
		return nil, err
	}
	out, err := shapeinference.Concatenate(ss, axis)
	if err != nil {
		return nil, err
	}
	typ.Shape = int64s(out.Dimensions)
	z, err := f.raw.Concatenate(values, int64(axis), typ)
	if err != nil {
		return nil, err
	}
	f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: "concatenate", operands: values, axis: axis, result: z})
	f.valueShapes[z.ID] = out
	return &node{f, z}, nil
}

func (f *function) DynamicUpdateSlice(operand, update compute.Value, starts []compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(operand)
	if err != nil {
		return nil, err
	}
	y, err := f.value(update)
	if err != nil {
		return nil, err
	}
	indices := make([]hlo.Value, len(starts))
	for i, v := range starts {
		indices[i], err = f.value(v)
		if err != nil {
			return nil, err
		}
	}
	z, err := f.raw.DynamicUpdateSlice(x, y, indices)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
