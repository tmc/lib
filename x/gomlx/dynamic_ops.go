package gomlx

import (
	"fmt"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/sets"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

type dimensionSpec struct {
	static int
	name   string
	value  hlo.Value
}

type padSpec struct {
	start, end, interior dimensionSpec
}

func (f *function) dimensionValue(v compute.Value) (hlo.Value, error) {
	if v == nil {
		return hlo.Value{}, nil
	}
	x, err := f.value(v)
	if err != nil {
		return hlo.Value{}, err
	}
	if len(x.Type.Shape) != 0 || !elementDType(x.Type.ElementType).IsInt() {
		return hlo.Value{}, fmt.Errorf("dynamic dimension: value must be an integer scalar")
	}
	return x, nil
}

func (f *function) dimensionSpecs(specs []compute.DynamicDimensionSpec) ([]dimensionSpec, error) {
	out := make([]dimensionSpec, len(specs))
	for i, spec := range specs {
		v, err := f.dimensionValue(spec.Value)
		if err != nil {
			return nil, err
		}
		out[i] = dimensionSpec{spec.Static, spec.Name, v}
	}
	return out, nil
}

func (f *function) knownAxisNames() sets.Set[string] {
	names := sets.Make[string]()
	for _, s := range f.valueShapes {
		for _, name := range s.AxisNames {
			if name != "" && name != shapes.AnonymousAxis {
				names.Insert(name)
			}
		}
	}
	return names
}

func (f *function) recordDynamic(op dynamicOp, shape shapes.Shape) (compute.Value, error) {
	if f.returned || f.owner.closed {
		return nil, fmt.Errorf("function closed")
	}
	info, ok := element(shape.DType)
	if !ok {
		return nil, unsupported
	}
	op.result = hlo.Value{ID: fmt.Sprintf("%%dynamic_op%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(shape.Dimensions), ElementType: info.hlo}}
	// This marker is never compiled; structured replay emits the concrete operation.
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("// %s: %s", op.result.ID, op.kind))
	f.dynamicOps = append(f.dynamicOps, op)
	f.valueShapes[op.result.ID] = shape.Clone()
	return &node{f, op.result}, nil
}

func (f *function) DynamicReshape(operand compute.Value, specs ...compute.DynamicDimensionSpec) (compute.Value, error) {
	s, err := f.Shape(operand)
	if err != nil {
		return nil, err
	}
	out, err := shapeinference.DynamicReshape(s, specs)
	if err != nil {
		return nil, err
	}
	dims, err := f.dimensionSpecs(specs)
	if err != nil {
		return nil, err
	}
	x, _ := f.value(operand)
	return f.recordDynamic(dynamicOp{kind: "reshape_dynamic", operand: x, dimensions: dims}, out)
}

func (f *function) DynamicBroadcastInDim(operand compute.Value, axes []int, specs ...compute.DynamicDimensionSpec) (compute.Value, error) {
	s, err := f.Shape(operand)
	if err != nil {
		return nil, err
	}
	out, err := shapeinference.DynamicBroadcastInDim(s, axes, specs, f.knownAxisNames())
	if err != nil {
		return nil, err
	}
	dims, err := f.dimensionSpecs(specs)
	if err != nil {
		return nil, err
	}
	x, _ := f.value(operand)
	return f.recordDynamic(dynamicOp{kind: "broadcast_dynamic", operand: x, axes: slices.Clone(axes), dimensions: dims}, out)
}

func (f *function) DynamicIota(dtype dtypes.DType, axis int, specs ...compute.DynamicDimensionSpec) (compute.Value, error) {
	if dtype == dtypes.Bool || dtype.IsComplex() {
		return nil, unsupported
	}
	out, err := shapeinference.DynamicIota(dtype, axis, specs, f.knownAxisNames())
	if err != nil {
		return nil, err
	}
	dims, err := f.dimensionSpecs(specs)
	if err != nil {
		return nil, err
	}
	return f.recordDynamic(dynamicOp{kind: "iota_dynamic", axis: axis, dimensions: dims}, out)
}

func (f *function) DynamicPad(operand, fill compute.Value, configs ...compute.DynamicPadAxis) (compute.Value, error) {
	s, err := f.Shape(operand)
	if err != nil {
		return nil, err
	}
	fs, err := f.Shape(fill)
	if err != nil {
		return nil, err
	}
	if fs.Rank() != 0 || fs.DType != s.DType {
		return nil, fmt.Errorf("dynamic pad: fill must be a scalar with the input dtype")
	}
	out, err := shapeinference.DynamicPad(s, configs...)
	if err != nil {
		return nil, err
	}
	pads := make([]padSpec, s.Rank())
	for i, c := range configs {
		start, err := f.dimensionValue(c.StartValue)
		if err != nil {
			return nil, err
		}
		end, err := f.dimensionValue(c.EndValue)
		if err != nil {
			return nil, err
		}
		interior, err := f.dimensionValue(c.InteriorValue)
		if err != nil {
			return nil, err
		}
		pads[i] = padSpec{dimensionSpec{static: c.Start, value: start}, dimensionSpec{static: c.End, value: end}, dimensionSpec{static: c.Interior, value: interior}}
	}
	x, _ := f.value(operand)
	v, _ := f.value(fill)
	return f.recordDynamic(dynamicOp{kind: "pad_dynamic", operand: x, operands: []hlo.Value{v}, pads: pads}, out)
}
