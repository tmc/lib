package gomlx

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) BatchNormForInference(x, scale, offset, mean, variance compute.Value, epsilon float32, axis int) (compute.Value, error) {
	out, err := f.batchNorm("inference", epsilon, axis, x, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

func (f *function) BatchNormForTraining(x, scale, offset compute.Value, epsilon float32, axis int) (compute.Value, compute.Value, compute.Value, error) {
	out, err := f.batchNorm("training", epsilon, axis, x, scale, offset)
	if err != nil {
		return nil, nil, nil, err
	}
	return out[0], out[1], out[2], nil
}

func (f *function) BatchNormGradient(x, scale, mean, variance, grad compute.Value, epsilon float32, axis int) (compute.Value, compute.Value, compute.Value, error) {
	out, err := f.batchNorm("grad", epsilon, axis, x, scale, mean, variance, grad)
	if err != nil {
		return nil, nil, nil, err
	}
	return out[0], out[1], out[2], nil
}

func (f *function) batchNorm(op string, epsilon float32, axis int, args ...compute.Value) ([]compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	values := make([]hlo.Value, len(args))
	for i, v := range args {
		x, err := f.value(v)
		if err != nil {
			return nil, err
		}
		values[i] = x
	}
	typ := values[0].Type
	if !elementDType(typ.ElementType).IsFloat() {
		return nil, unsupported
	}
	if axis < 0 {
		axis += len(typ.Shape)
	}
	if axis < 0 || axis >= len(typ.Shape) {
		return nil, fmt.Errorf("batch norm: invalid feature axis")
	}
	feature := hlo.Type{Shape: []int64{typ.Shape[axis]}, ElementType: typ.ElementType}
	for i, x := range values[1:] {
		want := feature
		if op == "grad" && i == 3 {
			want = typ
		}
		if x.Type.ElementType != want.ElementType || !slices.Equal(x.Type.Shape, want.Shape) {
			return nil, fmt.Errorf("batch norm: argument %d has incompatible shape or dtype", i+1)
		}
	}
	count := 3
	if op == "inference" {
		count = 1
	}
	out := make([]compute.Value, count)
	names, outputTypes := make([]string, count), make([]string, count)
	for i := range out {
		t := typ
		if i > 0 {
			t = feature
		}
		v := hlo.Value{ID: fmt.Sprintf("%%batchnorm%d_%d", len(f.raw.Ops), i), Type: t}
		out[i] = &node{f, v}
		names[i] = v.ID
		outputTypes[i] = t.String()
	}
	inputs, inputTypes := make([]string, len(values)), make([]string, len(values))
	for i, v := range values {
		inputs[i] = v.ID
		inputTypes[i] = v.Type.String()
	}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.batch_norm_%s"(%s) {epsilon = 0x%08X : f32, feature_index = %d : i64} : (%s) -> (%s)`, strings.Join(names, ", "), op, strings.Join(inputs, ", "), math.Float32bits(epsilon), axis, strings.Join(inputTypes, ", "), strings.Join(outputTypes, ", ")))
	return out, nil
}
