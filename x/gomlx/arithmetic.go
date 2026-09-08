package gomlx

import (
	"fmt"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"math"

	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Mul(a, b compute.Value) (compute.Value, error) {
	if f.owner.returned {
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
	x, y, err = f.broadcastPair(x, y)
	if err != nil {
		return nil, err
	}
	z, err := f.raw.Multiply(x, y)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) BroadcastInDim(v compute.Value, s shapes.Shape, axes []int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	if err := validate(s); err != nil {
		return nil, err
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if s.DType != elementDType(x.Type.ElementType) {
		return nil, fmt.Errorf("broadcast dtype mismatch")
	}
	z, err := f.raw.BroadcastInDim(x, int64s(axes), hlo.Type{Shape: int64s(s.Dimensions), ElementType: x.Type.ElementType})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) Reshape(v compute.Value, dims ...int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if err := validate(shapes.Shape{DType: elementDType(x.Type.ElementType), Dimensions: dims}); err != nil {
		return nil, err
	}
	z, err := f.raw.Reshape(x, hlo.Type{Shape: int64s(dims), ElementType: x.Type.ElementType})
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) ReduceSum(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduce(v, "add", axes)
}
func (f *function) ReduceMax(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduce(v, "maximum", axes)
}
func (f *function) reduce(v compute.Value, op string, axes []int) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if len(axes) == 0 {
		axes = make([]int, len(x.Type.Shape))
		for i := range axes {
			axes[i] = i
		}
	}
	seen := make([]bool, len(x.Type.Shape))
	for _, a := range axes {
		if a < 0 || a >= len(seen) || seen[a] {
			return nil, fmt.Errorf("invalid reduction axis %d", a)
		}
		seen[a] = true
	}
	var dims []int64
	for i, d := range x.Type.Shape {
		if !seen[i] {
			dims = append(dims, d)
		}
	}
	var zero hlo.Value
	if x.Type.ElementType == "f32" {
		initial := float32(0)
		if op == "maximum" {
			initial = float32(math.Inf(-1))
		}
		zero, err = f.raw.Constant(initial, hlo.Type{ElementType: "f32"})
	} else if x.Type.ElementType == "ui32" || x.Type.ElementType == "ui64" {
		zero = hlo.Value{ID: fmt.Sprintf("%%zero%d", len(f.raw.Ops)), Type: hlo.Type{ElementType: x.Type.ElementType}}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<0> : %s", zero.ID, zero.Type))
	} else {
		initial := int64(0)
		if op == "maximum" {
			initial = math.MinInt64
			if x.Type.ElementType == "i32" {
				initial = math.MinInt32
			}
		}
		zero, err = f.raw.IntegerConstant(initial, hlo.Type{ElementType: x.Type.ElementType})
	}
	if err != nil {
		return nil, err
	}
	z, err := f.raw.Reduce(x, zero, int64s(axes), hlo.Type{Shape: dims, ElementType: x.Type.ElementType}, op)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}

func (f *function) binary(a, b compute.Value, emit func(hlo.Value, hlo.Value) (hlo.Value, error)) (compute.Value, error) {
	if f.owner.returned {
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
	x, y, err = f.broadcastPair(x, y)
	if err != nil {
		return nil, err
	}
	z, err := emit(x, y)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) Sub(a, b compute.Value) (compute.Value, error) {
	return f.binary(a, b, f.raw.Subtract)
}
func (f *function) Div(a, b compute.Value) (compute.Value, error) {
	return f.binary(a, b, f.raw.Divide)
}
func (f *function) ConvertDType(v compute.Value, dtype dtypes.DType) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	info, ok := element(dtype)
	if !ok {
		return nil, unsupported
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	z, err := f.raw.Convert(x, info.hlo)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}
func (f *function) Sqrt(v compute.Value) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if x.Type.ElementType != "f32" {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%sqrt%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.sqrt %s : %s", z.ID, x.ID, x.Type))
	return &node{f, z}, nil
}

func (f *function) Pow(a, b compute.Value) (compute.Value, error) {
	return f.binary(a, b, func(x, y hlo.Value) (hlo.Value, error) {
		if x.Type.ElementType != "f32" {
			return hlo.Value{}, unsupported
		}
		z := hlo.Value{ID: fmt.Sprintf("%%pow%d", len(f.raw.Ops)), Type: x.Type}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.power %s, %s : %s", z.ID, x.ID, y.ID, x.Type))
		return z, nil
	})
}

func (f *function) Exp(v compute.Value) (compute.Value, error) {
	return f.unaryFloat(v, "exponential")
}
func (f *function) Log(v compute.Value) (compute.Value, error) {
	return f.unaryFloat(v, "log")
}
func (f *function) unaryFloat(v compute.Value, op string) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if x.Type.ElementType != "f32" {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%unary%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.%s %s : %s", z.ID, op, x.ID, x.Type))
	return &node{f, z}, nil
}

func (f *function) Identity(v compute.Value) (compute.Value, error) {
	if f.owner.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	return &node{f, x}, nil
}

func (f *function) Neg(v compute.Value) (compute.Value, error) {
	return f.unaryFloat(v, "negate")
}
