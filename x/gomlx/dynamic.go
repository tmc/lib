package gomlx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

type dynamicOp struct {
	kind     string
	operands []hlo.Value
	operand  hlo.Value
	axis     int
	axes     []int
	result   hlo.Value
}

const (
	dynamicDimensionSize       = "dimension_size"
	dynamicShape               = "shape"
	dynamicSubtract            = "subtract"
	dynamicMultiply            = "multiply"
	dynamicReduceSum           = "reduce_sum"
	dynamicSqrt                = "sqrt"
	dynamicIntegerConstant     = "integer_constant"
	dynamicInt32VectorConstant = "int32_vector_constant"
)

func (f *function) DynamicDimensionSize(operand compute.Value, axis int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(operand)
	if err != nil {
		return nil, err
	}
	if axis < 0 {
		axis += len(x.Type.Shape)
	}
	if axis < 0 || axis >= len(x.Type.Shape) {
		return nil, fmt.Errorf("dynamic dimension size: axis %d out of range for rank %d", axis, len(x.Type.Shape))
	}
	if x.Type.Shape[axis] >= 0 {
		z, err := f.raw.IntegerConstant(x.Type.Shape[axis], hlo.Type{ElementType: "i32"})
		if err != nil {
			return nil, err
		}
		f.valueShapes[z.ID] = shapes.Make(dtypes.Int32)
		f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: dynamicIntegerConstant, axis: int(x.Type.Shape[axis]), result: z})
		return &node{f, z}, nil
	}
	z := hlo.Value{ID: fmt.Sprintf("%%dynamic_dimension%d", len(f.raw.Ops)), Type: hlo.Type{ElementType: "i32"}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.get_dimension_size"(%s) {dimension = %d : i64} : (%s) -> %s`, z.ID, x.ID, axis, x.Type, z.Type))
	f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: dynamicDimensionSize, operand: x, axis: axis, result: z})
	f.valueShapes[z.ID] = shapes.Make(dtypes.Int32)
	return &node{f, z}, nil
}

func (f *function) DynamicShape(operand compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(operand)
	if err != nil {
		return nil, err
	}
	static := true
	for _, dim := range x.Type.Shape {
		static = static && dim >= 0
	}
	if static {
		values := make([]int32, len(x.Type.Shape))
		for i, dim := range x.Type.Shape {
			values[i] = int32(dim)
		}
		z, err := f.raw.Int32VectorConstant(values)
		if err != nil {
			return nil, err
		}
		f.valueShapes[z.ID] = shapes.Make(dtypes.Int32, len(values))
		f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: dynamicInt32VectorConstant, axes: ints32(values), result: z})
		return &node{f, z}, nil
	}
	z := hlo.Value{ID: fmt.Sprintf("%%dynamic_shape%d", len(f.raw.Ops)), Type: hlo.Type{Shape: []int64{int64(len(x.Type.Shape))}, ElementType: "i32"}}
	// The symbolic operation is retained for inspection. Recompiling specialization
	// replaces it structurally with the concrete shape constant.
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf(`%s = "stablehlo.custom_call"(%s) {call_target_name = "tmc.dynamic_shape", has_side_effect = false} : (%s) -> %s`, z.ID, x.ID, x.Type, z.Type))
	f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: dynamicShape, operand: x, result: z})
	f.valueShapes[z.ID] = shapes.Make(dtypes.Int32, len(x.Type.Shape))
	return &node{f, z}, nil
}

func (f *function) checkDynamicSlice() error {
	if len(f.owner.functions) != 1 || f.name != "main" {
		return fmt.Errorf("dynamic specialization: functions are not supported")
	}
	wantRawOps := 0
	for _, op := range f.dynamicOps {
		switch op.kind {
		case dynamicDimensionSize, dynamicShape, dynamicSubtract, dynamicMultiply, dynamicSqrt, dynamicIntegerConstant, dynamicInt32VectorConstant:
			wantRawOps++
		case dynamicReduceSum:
			wantRawOps += 2
		default:
			return fmt.Errorf("dynamic specialization: unsupported operation %q", op.kind)
		}
	}
	if wantRawOps != len(f.raw.Ops) {
		return fmt.Errorf("dynamic specialization: operation outside supported slice")
	}
	return nil
}

func (f *function) specialize(bindings shapes.AxisBindings) (string, []shapes.Shape, []shapes.Shape, error) {
	if err := f.checkDynamicSlice(); err != nil {
		return "", nil, nil, err
	}
	inputs := make([]shapes.Shape, len(f.inputs))
	concrete := make(map[string]hlo.Value, len(f.raw.Args))
	raw := hlo.NewFunc("main")
	for i, shape := range f.inputs {
		resolved, err := shape.Resolve(bindings)
		if err != nil {
			return "", nil, nil, err
		}
		inputs[i] = resolved
		info, _ := element(resolved.DType)
		t := hlo.Type{Shape: int64s(resolved.Dimensions), ElementType: info.hlo}
		v := hlo.Value{ID: f.raw.Args[i].ID, Type: t}
		raw.Args = append(raw.Args, v)
		concrete[f.raw.Args[i].ID] = v
	}
	for _, op := range f.dynamicOps {
		operand := concrete[op.operand.ID]
		var result hlo.Value
		var err error
		switch op.kind {
		case dynamicDimensionSize:
			result, err = raw.IntegerConstant(operand.Type.Shape[op.axis], hlo.Type{ElementType: "i32"})
		case dynamicShape:
			values := make([]int32, len(operand.Type.Shape))
			for j, dim := range operand.Type.Shape {
				values[j] = int32(dim)
			}
			result, err = raw.Int32VectorConstant(values)
		case dynamicSubtract, dynamicMultiply:
			a, aok := concrete[op.operands[0].ID]
			b, bok := concrete[op.operands[1].ID]
			if !aok || !bok {
				return "", nil, nil, fmt.Errorf("dynamic specialization: unknown binary operand")
			}
			if op.kind == dynamicSubtract {
				result, err = raw.Subtract(a, b)
			} else {
				result, err = raw.Multiply(a, b)
			}
		case dynamicReduceSum:
			zero, zerr := raw.Constant(float32(0), hlo.Type{ElementType: operand.Type.ElementType})
			if zerr != nil {
				err = zerr
				break
			}
			seen := make([]bool, len(operand.Type.Shape))
			for _, axis := range op.axes {
				seen[axis] = true
			}
			var dims []int64
			for i, dim := range operand.Type.Shape {
				if !seen[i] {
					dims = append(dims, dim)
				}
			}
			result, err = raw.Reduce(operand, zero, int64s(op.axes), hlo.Type{Shape: dims, ElementType: operand.Type.ElementType}, "add")
		case dynamicSqrt:
			result, err = raw.Sqrt(operand)
		case dynamicIntegerConstant:
			result, err = raw.IntegerConstant(int64(op.axis), hlo.Type{ElementType: "i32"})
		case dynamicInt32VectorConstant:
			values := make([]int32, len(op.axes))
			for i, value := range op.axes {
				values[i] = int32(value)
			}
			result, err = raw.Int32VectorConstant(values)
		default:
			return "", nil, nil, fmt.Errorf("dynamic specialization: unsupported operation %s", op.kind)
		}
		if err != nil {
			return "", nil, nil, err
		}
		concrete[op.result.ID] = result
	}
	outputs := make([]shapes.Shape, len(f.raw.Returns))
	returns := make([]hlo.Value, len(f.raw.Returns))
	for i, old := range f.raw.Returns {
		result, ok := concrete[old.ID]
		if !ok {
			return "", nil, nil, fmt.Errorf("dynamic specialization: unsupported result %s", old.ID)
		}
		returns[i] = result
		dims := ints(result.Type.Shape)
		outputs[i] = shapes.Make(elementDType(result.Type.ElementType), dims...)
	}
	raw.Return(returns...)
	var text strings.Builder
	text.WriteString("module {\n")
	if err := raw.Emit(&text); err != nil {
		return "", nil, nil, err
	}
	text.WriteString("}\n")
	return text.String(), inputs, outputs, nil
}

func ints32(values []int32) []int {
	out := make([]int, len(values))
	for i, value := range values {
		out[i] = int(value)
	}
	return out
}

func ints(values []int64) []int {
	result := make([]int, len(values))
	for i, value := range values {
		result[i] = int(value)
	}
	return result
}

func (f *function) recordBinaryShape(result, left, right hlo.Value) {
	a, aok := f.valueShapes[left.ID]
	b, bok := f.valueShapes[right.ID]
	if !aok || !bok || len(a.Dimensions) != len(result.Type.Shape) || len(b.Dimensions) != len(result.Type.Shape) {
		return
	}
	names, err := shapes.UnifyAxisNames(a, b)
	if err != nil {
		return
	}
	dims := ints(result.Type.Shape)
	if slices.Contains(dims, shapes.DynamicDim) {
		f.valueShapes[result.ID] = shapes.MakeDynamic(elementDType(result.Type.ElementType), dims, names)
	} else {
		f.valueShapes[result.ID] = shapes.Make(elementDType(result.Type.ElementType), dims...)
	}
}

func (f *function) recordReductionShape(result, operand hlo.Value, axes []int) {
	input, ok := f.valueShapes[operand.ID]
	if !ok {
		return
	}
	reduced := make(map[int]bool, len(axes))
	for _, axis := range axes {
		reduced[axis] = true
	}
	var dims []int
	var names []string
	for axis, dim := range input.Dimensions {
		if !reduced[axis] {
			dims = append(dims, dim)
			names = append(names, input.AxisName(axis))
		}
	}
	if slices.Contains(dims, shapes.DynamicDim) {
		f.valueShapes[result.ID] = shapes.MakeDynamic(elementDType(result.Type.ElementType), dims, names)
	} else {
		f.valueShapes[result.ID] = shapes.Make(elementDType(result.Type.ElementType), dims...)
	}
}
