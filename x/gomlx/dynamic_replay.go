package gomlx

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

// scalarEvaluator evaluates the concrete prefix that computes a dimension.
// Evaluation runs on the backend, preserving the graph's integer semantics.
type scalarEvaluator func(*hlo.FuncBuilder, hlo.Value) (int64, error)

func (b *Backend) evaluateDimension(raw *hlo.FuncBuilder, value hlo.Value, inputs []*xla.Buffer) (result int64, err error) {
	prefix := *raw
	prefix.Return(value)
	program, err := emitDynamicProgram(&prefix)
	if err != nil {
		return 0, err
	}
	compiled, err := b.compileProgram([]byte(program), "mlir", nil)
	if err != nil {
		return 0, fmt.Errorf("dynamic dimension compile: %w", err)
	}
	outputs, err := compiled.Execute(context.Background(), inputs)
	if err != nil {
		for _, output := range outputs {
			err = errors.Join(err, output.Close())
		}
		return 0, errors.Join(fmt.Errorf("dynamic dimension execute: %w", err), compiled.Close())
	}
	buffers := make([]dimensionBuffer, len(outputs))
	for i, output := range outputs {
		buffers[i] = output
	}
	return readDimension(value, buffers, compiled.Close)
}

type dimensionBuffer interface {
	ToHost([]byte) error
	Close() error
}

// readDimension consumes the prefix outputs and closes the executable after
// closing its outputs, retaining every cleanup error.
func readDimension(value hlo.Value, outputs []dimensionBuffer, closeExecutable func() error) (result int64, err error) {
	defer func() {
		for _, output := range outputs {
			err = errors.Join(err, output.Close())
		}
		err = errors.Join(err, closeExecutable())
	}()
	if len(outputs) != 1 {
		return 0, fmt.Errorf("dynamic dimension: output count mismatch")
	}
	dtype := elementDType(value.Type.ElementType)
	info, _ := element(dtype)
	data := make([]byte, info.size)
	if err := outputs[0].ToHost(data); err != nil {
		return 0, err
	}
	var n uint64
	switch len(data) {
	case 1:
		n = uint64(data[0])
	case 2:
		n = uint64(binary.LittleEndian.Uint16(data))
	case 4:
		n = uint64(binary.LittleEndian.Uint32(data))
	case 8:
		n = binary.LittleEndian.Uint64(data)
	default:
		return 0, fmt.Errorf("dynamic dimension: unsupported integer width")
	}
	if dtype.IsUnsigned() {
		if n > math.MaxInt64 {
			return 0, fmt.Errorf("dynamic dimension: integer exceeds int64")
		}
		return int64(n), nil
	}
	shift := 64 - len(data)*8
	return int64(n<<shift) >> shift, nil
}

func emitDynamicProgram(raw *hlo.FuncBuilder) (string, error) {
	var text strings.Builder
	text.WriteString("module {\n")
	if err := raw.Emit(&text); err != nil {
		return "", err
	}
	text.WriteString("}\n")
	return text.String(), nil
}

func dimensionProduct(dims []int64) (int64, error) {
	product := int64(1)
	for _, dim := range dims {
		if dim < 0 {
			return 0, fmt.Errorf("dynamic dimension: negative size %d", dim)
		}
		if dim != 0 && product > math.MaxInt64/dim {
			return 0, fmt.Errorf("dynamic dimension: size overflow")
		}
		product *= dim
	}
	return product, nil
}

func resolveDimensions(specs []dimensionSpec, operand hlo.Value, reshape bool, bindings shapes.AxisBindings, scalar func(hlo.Value) (int64, error)) ([]int64, error) {
	dims := make([]int64, len(specs))
	inferred := -1
	for i, spec := range specs {
		switch {
		case spec.value.ID != "":
			n, err := scalar(spec.value)
			if err != nil {
				return nil, err
			}
			dims[i] = n
		case spec.name == "":
			dims[i] = int64(spec.static)
		case reshape:
			if inferred >= 0 {
				return nil, fmt.Errorf("dynamic reshape: multiple inferred axes")
			}
			inferred, dims[i] = i, 1
		default:
			n, ok := bindings[spec.name]
			if !ok {
				return nil, fmt.Errorf("dynamic dimension: unknown axis %q", spec.name)
			}
			dims[i] = int64(n)
		}
	}
	product, err := dimensionProduct(dims)
	if err != nil {
		return nil, err
	}
	if reshape {
		size, err := dimensionProduct(operand.Type.Shape)
		if err != nil {
			return nil, err
		}
		if inferred >= 0 {
			if product == 0 || size%product != 0 {
				return nil, fmt.Errorf("dynamic reshape: cannot infer size %d from product %d", size, product)
			}
			dims[inferred] = size / product
		} else if size != product {
			return nil, fmt.Errorf("dynamic reshape: element count mismatch %d != %d", size, product)
		}
	}
	return dims, nil
}

func replayPad(raw *hlo.FuncBuilder, operand, fill hlo.Value, pads []padSpec, scalar func(hlo.Value) (int64, error)) (hlo.Value, error) {
	if len(pads) == 0 {
		return operand, nil
	}
	low, high, interior := make([]int, len(pads)), make([]int, len(pads)), make([]int, len(pads))
	dims := make([]int64, len(pads))
	for i, pad := range pads {
		values := []*int{&low[i], &high[i], &interior[i]}
		for j, spec := range []dimensionSpec{pad.start, pad.end, pad.interior} {
			n := int64(spec.static)
			if spec.value.ID != "" {
				var err error
				n, err = scalar(spec.value)
				if err != nil {
					return hlo.Value{}, err
				}
			}
			*values[j] = int(n)
		}
		if interior[i] < 0 {
			return hlo.Value{}, fmt.Errorf("dynamic pad: negative interior padding")
		}
		// Use checked additions, including negative edge cropping.
		n := operand.Type.Shape[i]
		gap := int64(0)
		if n > 0 {
			if int64(interior[i]) != 0 && n-1 > math.MaxInt64/int64(interior[i]) {
				return hlo.Value{}, fmt.Errorf("dynamic pad: size overflow")
			}
			gap = (n - 1) * int64(interior[i])
		}
		for _, add := range []int64{gap, int64(low[i]), int64(high[i])} {
			if add > 0 && n > math.MaxInt64-add || add < 0 && n < math.MinInt64-add {
				return hlo.Value{}, fmt.Errorf("dynamic pad: size overflow")
			}
			n += add
		}
		if n < 0 {
			return hlo.Value{}, fmt.Errorf("dynamic pad: negative output size")
		}
		dims[i] = n
	}
	if _, err := dimensionProduct(dims); err != nil {
		return hlo.Value{}, err
	}
	z := hlo.Value{ID: fmt.Sprintf("%%replay_pad%d", len(raw.Ops)), Type: hlo.Type{Shape: dims, ElementType: operand.Type.ElementType}}
	raw.Ops = append(raw.Ops, fmt.Sprintf(`%s = "stablehlo.pad"(%s, %s) {edge_padding_low = array<i64: %s>, edge_padding_high = array<i64: %s>, interior_padding = array<i64: %s>} : (%s, %s) -> %s`, z.ID, operand.ID, fill.ID, indexValues(low), indexValues(high), indexValues(interior), operand.Type, fill.Type, z.Type))
	return z, nil
}
