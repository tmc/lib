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
	if f.returned {
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
	if f.returned {
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
func (f *function) ReduceProduct(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduce(v, "multiply", axes)
}
func (f *function) ReduceMax(v compute.Value, axes ...int) (compute.Value, error) {
	return f.reduce(v, "maximum", axes)
}
func (f *function) reduce(v compute.Value, op string, axes []int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if x.Type.ElementType == "i1" && op != "and" && op != "or" && op != "xor" {
		return nil, unsupported
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
	if x.Type.ElementType == "i1" {
		literal := "false"
		if op == "and" {
			literal = "true"
		}
		zero = hlo.Value{ID: fmt.Sprintf("%%zero%d", len(f.raw.Ops)), Type: hlo.Type{ElementType: "i1"}}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", zero.ID, literal, zero.Type))
	} else if elementDType(x.Type.ElementType).IsFloat() {
		initial := float32(0)
		if op == "multiply" {
			initial = 1
		}
		if op == "maximum" {
			initial = float32(math.Inf(-1))
		}
		if op == "minimum" {
			initial = float32(math.Inf(1))
		}
		zero, err = f.raw.Constant(initial, hlo.Type{ElementType: x.Type.ElementType})
	} else if x.Type.ElementType == "ui8" || x.Type.ElementType == "ui32" || x.Type.ElementType == "ui64" {
		zero = hlo.Value{ID: fmt.Sprintf("%%zero%d", len(f.raw.Ops)), Type: hlo.Type{ElementType: x.Type.ElementType}}
		initial := uint64(0)
		if op == "minimum" || op == "and" {
			initial = math.MaxUint64
			if x.Type.ElementType == "ui8" {
				initial = math.MaxUint8
			}
			if x.Type.ElementType == "ui32" {
				initial = math.MaxUint32
			}
		}
		if op == "multiply" {
			initial = 1
		}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%d> : %s", zero.ID, initial, zero.Type))
	} else {
		initial := int64(0)
		if op == "and" {
			initial = -1
		}
		if op == "minimum" {
			initial = math.MaxInt64
			if x.Type.ElementType == "i8" {
				initial = math.MaxInt8
			}
			if x.Type.ElementType == "i32" {
				initial = math.MaxInt32
			}
		}
		if op == "multiply" {
			initial = 1
		}
		if op == "maximum" {
			initial = math.MinInt64
			if x.Type.ElementType == "i8" {
				initial = math.MinInt8
			}
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
	if f.returned {
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
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if !elementDType(x.Type.ElementType).IsFloat() {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%sqrt%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.sqrt %s : %s", z.ID, x.ID, x.Type))
	return &node{f, z}, nil
}

func (f *function) Pow(a, b compute.Value) (compute.Value, error) {
	return f.binary(a, b, func(x, y hlo.Value) (hlo.Value, error) {
		if x.Type.ElementType == "i1" {
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
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if !elementDType(x.Type.ElementType).IsFloat() {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%unary%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.%s %s : %s", z.ID, op, x.ID, x.Type))
	return &node{f, z}, nil
}

func (f *function) Identity(v compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	return &node{f, x}, nil
}

func (f *function) Neg(v compute.Value) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	if x.Type.ElementType == "i1" {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%neg%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.negate %s : %s", z.ID, x.ID, x.Type))
	return &node{f, z}, nil
}

func (f *function) Sin(v compute.Value) (compute.Value, error) { return f.unaryFloat(v, "sine") }

func (f *function) Rem(a, b compute.Value) (compute.Value, error) {
	return f.binary(a, b, func(x, y hlo.Value) (hlo.Value, error) {
		if x.Type.ElementType == "i1" {
			return hlo.Value{}, unsupported
		}
		z := hlo.Value{ID: fmt.Sprintf("%%rem%d", len(f.raw.Ops)), Type: x.Type}
		f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.remainder %s, %s : %s", z.ID, x.ID, y.ID, x.Type))
		return z, nil
	})
}

func (f *function) Rsqrt(v compute.Value) (compute.Value, error) {
	return f.unaryFloat(v, "rsqrt")
}

func (f *function) LogicalNot(v compute.Value) (compute.Value, error) {
	return f.invert(v, true)
}

func (f *function) BitwiseNot(v compute.Value) (compute.Value, error) {
	return f.invert(v, false)
}

func (f *function) invert(v compute.Value, logical bool) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(v)
	if err != nil {
		return nil, err
	}
	dtype := elementDType(x.Type.ElementType)
	if logical && dtype != dtypes.Bool || !logical && !dtype.IsInt() {
		return nil, unsupported
	}
	z := hlo.Value{ID: fmt.Sprintf("%%not%d", len(f.raw.Ops)), Type: x.Type}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.not %s : %s", z.ID, x.ID, x.Type))
	return &node{f, z}, nil
}
