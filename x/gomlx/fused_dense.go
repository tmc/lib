package gomlx

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
)

// FusedDense implements the compute FusedDense contract by composing a
// dot, optional bias broadcast, and the requested activation.
func (f *function) FusedDense(x, weight, bias compute.Value, options compute.DenseConfig) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	xs, err := f.Shape(x)
	if err != nil {
		return nil, fmt.Errorf("fused dense input: %w", err)
	}
	ws, err := f.Shape(weight)
	if err != nil {
		return nil, fmt.Errorf("fused dense weight: %w", err)
	}
	if len(xs.Dimensions) < 1 || len(ws.Dimensions) < 2 {
		return nil, fmt.Errorf("fused dense requires input rank >= 1 and weight rank >= 2")
	}
	if !denseDType(xs.DType) {
		return nil, unsupported
	}
	if ws.DType != xs.DType {
		return nil, fmt.Errorf("fused dense requires matching f32, f64, f16, or bf16 inputs")
	}
	if options.WeightLayout != compute.DenseLayoutInputOutputs && options.WeightLayout != compute.DenseLayoutOutputsInput {
		return nil, fmt.Errorf("fused dense: unsupported weight layout %v", options.WeightLayout)
	}
	contractAxis := 0
	featureDims := ws.Dimensions[1:]
	if options.WeightLayout == compute.DenseLayoutOutputsInput {
		contractAxis = len(ws.Dimensions) - 1
		featureDims = ws.Dimensions[:len(ws.Dimensions)-1]
	}
	if xs.Dimensions[len(xs.Dimensions)-1] != ws.Dimensions[contractAxis] {
		return nil, fmt.Errorf("fused dense: input feature dimension %d does not match weight dimension %d", xs.Dimensions[len(xs.Dimensions)-1], ws.Dimensions[contractAxis])
	}
	if bias != nil {
		bs, e := f.Shape(bias)
		if e != nil {
			return nil, fmt.Errorf("fused dense bias: %w", e)
		}
		if bs.DType != xs.DType || len(bs.Dimensions) != len(featureDims) {
			return nil, fmt.Errorf("fused dense: bias must have dtype %v and shape %v", xs.DType, featureDims)
		}
		for i, d := range featureDims {
			if d < 0 || bs.Dimensions[i] < 0 {
				return nil, unsupported
			}
			if bs.Dimensions[i] != d {
				return nil, fmt.Errorf("fused dense: bias shape %v does not match %v", bs.Dimensions, featureDims)
			}
		}
	}
	switch options.Activation {
	case compute.ActivationNone, compute.ActivationGelu, compute.ActivationRelu, compute.ActivationSilu, compute.ActivationHardSwish, compute.ActivationTanh:
	default:
		return nil, fmt.Errorf("fused dense: unsupported activation %v", options.Activation)
	}
	for _, d := range xs.Dimensions {
		if d < 0 {
			return nil, unsupported
		}
	}
	for _, d := range ws.Dimensions {
		if d < 0 {
			return nil, unsupported
		}
	}

	contracted := []int{len(xs.Dimensions) - 1}
	result, err := f.DotGeneral(x, contracted, nil, weight, []int{contractAxis}, nil, compute.DotGeneralConfig{})
	if err != nil {
		return nil, fmt.Errorf("fused dense dot: %w", err)
	}
	outDims := append([]int(nil), xs.Dimensions[:len(xs.Dimensions)-1]...)
	outDims = append(outDims, featureDims...)
	outShape := shapes.Shape{DType: xs.DType, Dimensions: outDims}
	if bias != nil {
		axes := make([]int, len(featureDims))
		for i := range axes {
			axes[i] = len(xs.Dimensions) - 1 + i
		}
		bv, e := f.BroadcastInDim(bias, outShape, axes)
		if e != nil {
			return nil, fmt.Errorf("fused dense bias broadcast: %w", e)
		}
		result, e = f.Add(result, bv)
		if e != nil {
			return nil, fmt.Errorf("fused dense bias add: %w", e)
		}
	}
	return f.denseActivation(result, options.Activation, xs.DType)
}

func denseDType(dtype dtypes.DType) bool {
	return dtype == dtypes.Float32 || dtype == dtypes.Float64 || dtype == dtypes.Float16 || dtype == dtypes.BFloat16
}

func (f *function) denseScalar(dtype dtypes.DType, value float64) (compute.Value, error) {
	switch dtype {
	case dtypes.Float32:
		return f.Constant([]float32{float32(value)})
	case dtypes.Float64:
		return f.Constant([]float64{value})
	case dtypes.Float16:
		return f.Constant([]float16.Float16{float16.FromFloat32(float32(value))})
	case dtypes.BFloat16:
		return f.Constant([]bfloat16.BFloat16{bfloat16.FromFloat32(float32(value))})
	default:
		return nil, fmt.Errorf("fused dense: unsupported scalar dtype %v", dtype)
	}
}

func (f *function) denseActivation(v compute.Value, activation compute.ActivationType, dtype dtypes.DType) (compute.Value, error) {
	if activation == compute.ActivationNone {
		return v, nil
	}
	zero, err := f.denseScalar(dtype, 0)
	if err != nil {
		return nil, err
	}
	switch activation {
	case compute.ActivationRelu:
		return f.Max(v, zero)
	case compute.ActivationTanh:
		return f.Tanh(v)
	case compute.ActivationSilu:
		gate, e := f.Logistic(v)
		if e != nil {
			return nil, e
		}
		return f.Mul(v, gate)
	case compute.ActivationGelu:
		sqrt2, e := f.denseScalar(dtype, math.Sqrt2)
		if e != nil {
			return nil, e
		}
		half, e := f.denseScalar(dtype, 0.5)
		if e != nil {
			return nil, e
		}
		one, e := f.denseScalar(dtype, 1)
		if e != nil {
			return nil, e
		}
		x, e := f.Div(v, sqrt2)
		if e != nil {
			return nil, e
		}
		x, e = f.Erf(x)
		if e != nil {
			return nil, e
		}
		x, e = f.Add(x, one)
		if e != nil {
			return nil, e
		}
		x, e = f.Mul(x, half)
		if e != nil {
			return nil, e
		}
		return f.Mul(v, x)
	case compute.ActivationHardSwish:
		six, e := f.denseScalar(dtype, 6)
		if e != nil {
			return nil, e
		}
		half, e := f.denseScalar(dtype, 0.5)
		if e != nil {
			return nil, e
		}
		one, e := f.denseScalar(dtype, 1)
		if e != nil {
			return nil, e
		}
		x, e := f.Div(v, six)
		if e != nil {
			return nil, e
		}
		x, e = f.Add(x, half)
		if e != nil {
			return nil, e
		}
		x, e = f.Max(x, zero)
		if e != nil {
			return nil, e
		}
		x, e = f.Min(x, one)
		if e != nil {
			return nil, e
		}
		return f.Mul(v, x)
	default:
		return nil, fmt.Errorf("fused dense: unsupported activation %v", activation)
	}
}
