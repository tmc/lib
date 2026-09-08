package gomlx

import (
	"fmt"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) DotGeneral(a compute.Value, ac, ab []int, b compute.Value, bc, bb []int, c compute.DotGeneralConfig) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}

	x, err := f.value(a)
	if err != nil {
		return nil, err
	}
	y, err := f.value(b)
	if err != nil {
		return nil, err
	}
	shape, err := contractionShape(x.Type.Shape, y.Type.Shape, ac, ab, bc, bb)
	if err != nil {
		return nil, err
	}

	if x.Type.ElementType != y.Type.ElementType {
		return nil, fmt.Errorf("dot: input dtypes must match")
	}
	if c == (compute.DotGeneralConfig{}) {
		z, err := f.raw.DotGeneral(x, y, int64s(ab), int64s(bb), int64s(ac), int64s(bc), hlo.Type{Shape: shape, ElementType: x.Type.ElementType})
		if err != nil {
			return nil, err
		}
		return &node{f, z}, nil
	}
	input := elementDType(x.Type.ElementType)
	acc := c.AccumulatorDType
	if acc == dtypes.InvalidDType {
		acc = input
	}
	// Match the executable upstream backends: an omitted output dtype retains
	// the input dtype, even when an accumulator dtype is supplied.
	output := c.OutputDType
	if output == dtypes.InvalidDType {
		output = input
	}
	ai, aok := element(acc)
	oi, ook := element(output)
	if !aok || !ook || !input.IsFloat() || !acc.IsFloat() || !output.IsFloat() {
		return nil, unsupported
	}
	algorithm := ""
	if acc != input {
		if acc == dtypes.Float32 {
			algorithm = ", algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>"
		} else {
			x, err = f.raw.Convert(x, ai.hlo)
			if err != nil {
				return nil, err
			}
			y, err = f.raw.Convert(y, ai.hlo)
			if err != nil {
				return nil, err
			}
		}
	}
	z := hlo.Value{ID: fmt.Sprintf("%%dot%d", len(f.raw.Ops)), Type: hlo.Type{Shape: shape, ElementType: oi.hlo}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = \"stablehlo.dot_general\"(%s, %s) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = %s, rhs_batching_dimensions = %s, lhs_contracting_dimensions = %s, rhs_contracting_dimensions = %s>%s} : (%s, %s) -> %s", z.ID, x.ID, y.ID, indexArray(ab), indexArray(bb), indexArray(ac), indexArray(bc), algorithm, x.Type, y.Type, z.Type))
	return &node{f, z}, nil
}
