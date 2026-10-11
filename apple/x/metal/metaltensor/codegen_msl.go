package metaltensor

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalir"
)

func tensorMatMulMSL(expr *egraph.Expr, opts egraph.CodegenOptions) (string, error) {
	output, err := tensorExprMetadataOf(expr, nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL matmul: output metadata: %w", err)
	}
	if err := validateTensorMSLSort(expr, output); err != nil {
		return "", fmt.Errorf("tensor MSL matmul: output: %w", err)
	}
	if output.attributes != (metalir.ScalarAttributes{}) {
		return "", fmt.Errorf("tensor MSL matmul: scalar attributes are unsupported")
	}
	if len(expr.Children) != 2 {
		return "", fmt.Errorf("tensor MSL matmul: wants 2 inputs, got %d", len(expr.Children))
	}
	left, err := tensorExprMetadataOf(expr.Children[0], nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL matmul: left input: %w", err)
	}
	if err := validateTensorMSLSort(expr.Children[0], left); err != nil {
		return "", fmt.Errorf("tensor MSL matmul: left input: %w", err)
	}
	right, err := tensorExprMetadataOf(expr.Children[1], nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL matmul: right input: %w", err)
	}
	if err := validateTensorMSLSort(expr.Children[1], right); err != nil {
		return "", fmt.Errorf("tensor MSL matmul: right input: %w", err)
	}
	if err := validateTensorStageShape("matmul", output.shape, []tensorExprMetadata{left, right}); err != nil {
		return "", err
	}
	if left.dtype != output.dtype || right.dtype != output.dtype {
		return "", fmt.Errorf("tensor MSL matmul: input types must match output type")
	}
	if err := validateTensorMSLShape(output.shape); err != nil {
		return "", err
	}
	if err := validateTensorMSLShape(left.shape); err != nil {
		return "", fmt.Errorf("tensor MSL matmul: left input: %w", err)
	}
	if err := validateTensorMSLShape(right.shape); err != nil {
		return "", fmt.Errorf("tensor MSL matmul: right input: %w", err)
	}
	precision, err := tensorPrecision(output.dtype)
	if err != nil {
		return "", err
	}
	return generateMatMulStageMSL(opts.KernelName, precision, left.shape, right.shape, output.shape), nil
}

func tensorRMSNormMSL(expr *egraph.Expr, opts egraph.CodegenOptions) (string, error) {
	output, err := tensorExprMetadataOf(expr, nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL rmsnorm: output metadata: %w", err)
	}
	if err := validateTensorMSLSort(expr, output); err != nil {
		return "", fmt.Errorf("tensor MSL rmsnorm: output: %w", err)
	}
	if len(expr.Children) != 2 {
		return "", fmt.Errorf("tensor MSL rmsnorm: wants 2 inputs, got %d", len(expr.Children))
	}
	input, err := tensorExprMetadataOf(expr.Children[0], nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL rmsnorm: input: %w", err)
	}
	if err := validateTensorMSLSort(expr.Children[0], input); err != nil {
		return "", fmt.Errorf("tensor MSL rmsnorm: input: %w", err)
	}
	weight, err := tensorExprMetadataOf(expr.Children[1], nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL rmsnorm: weight: %w", err)
	}
	if err := validateTensorMSLSort(expr.Children[1], weight); err != nil {
		return "", fmt.Errorf("tensor MSL rmsnorm: weight: %w", err)
	}
	if err := validateTensorStageShape("rmsnorm", output.shape, []tensorExprMetadata{input, weight}); err != nil {
		return "", err
	}
	if input.dtype != output.dtype || weight.dtype != output.dtype {
		return "", fmt.Errorf("tensor MSL rmsnorm: input types must match output type")
	}
	attrs := output.attributes
	if !attrs.HasEpsilon || attrs.HasTheta || attrs.Theta != 0 || !positiveFinite(float64(attrs.Epsilon)) {
		return "", fmt.Errorf("tensor MSL rmsnorm: invalid epsilon")
	}
	if err := validateTensorMSLShape(output.shape); err != nil {
		return "", err
	}
	precision, err := tensorPrecision(output.dtype)
	if err != nil {
		return "", err
	}
	return generateRMSNormStageMSL(opts.KernelName, precision, input.shape, attrs.Epsilon), nil
}

func tensorRoPEMSL(expr *egraph.Expr, opts egraph.CodegenOptions) (string, error) {
	output, err := tensorExprMetadataOf(expr, nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL rope: output metadata: %w", err)
	}
	if err := validateTensorMSLSort(expr, output); err != nil {
		return "", fmt.Errorf("tensor MSL rope: output: %w", err)
	}
	if len(expr.Children) != 2 {
		return "", fmt.Errorf("tensor MSL rope: wants 2 inputs, got %d", len(expr.Children))
	}
	input, err := tensorExprMetadataOf(expr.Children[0], nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL rope: data input: %w", err)
	}
	if err := validateTensorMSLSort(expr.Children[0], input); err != nil {
		return "", fmt.Errorf("tensor MSL rope: data input: %w", err)
	}
	position, err := tensorExprMetadataOf(expr.Children[1], nil)
	if err != nil {
		return "", fmt.Errorf("tensor MSL rope: position input: %w", err)
	}
	if err := validateTensorMSLSort(expr.Children[1], position); err != nil {
		return "", fmt.Errorf("tensor MSL rope: position input: %w", err)
	}
	attrs := output.attributes
	if !attrs.HasTheta || attrs.HasEpsilon || attrs.Epsilon != 0 || !positiveFinite(float64(attrs.Theta)) {
		return "", fmt.Errorf("tensor MSL rope: invalid theta")
	}
	if !sameShape(output.shape, input.shape) || output.dtype != input.dtype {
		return "", fmt.Errorf("tensor MSL rope: output must preserve data input shape and type")
	}
	if len(input.shape) == 0 || input.shape[len(input.shape)-1] < 2 || input.shape[len(input.shape)-1]%2 != 0 {
		return "", fmt.Errorf("tensor MSL rope: last data dimension must be positive and even")
	}
	positionCount, err := tensorElementCount(position.shape)
	if err != nil || positionCount != 1 {
		return "", fmt.Errorf("tensor MSL rope: position must contain exactly one element")
	}
	positionType, err := ropePositionType(position.dtype)
	if err != nil {
		return "", fmt.Errorf("tensor MSL rope: %w", err)
	}
	if err := validateTensorMSLShape(input.shape); err != nil {
		return "", err
	}
	precision, err := tensorPrecision(output.dtype)
	if err != nil {
		return "", err
	}
	return generateRoPEStageMSL(opts.KernelName, precision, positionType, input.shape, attrs.Theta), nil
}

func validateTensorMSLSort(expr *egraph.Expr, metadata tensorExprMetadata) error {
	if expr.Sort != metadata.dtype.String() {
		return fmt.Errorf("sort %q does not match tensor type %s", expr.Sort, metadata.dtype)
	}
	return nil
}

func ropePositionType(dtype metalir.DataType) (string, error) {
	switch dtype {
	case metalir.Int32:
		return "int", nil
	case metalir.Int64:
		return "long", nil
	default:
		return "", fmt.Errorf("position type must be int32 or int64")
	}
}

func generateRoPEStageMSL(kernelName, precision, positionType string, shape []int, theta float32) string {
	width := shape[len(shape)-1]
	count, _ := tensorElementCount(shape)
	pairs := count / 2
	return fmt.Sprintf(`#include <metal_stdlib>
using namespace metal;

kernel void %s(
    device const %s* input [[buffer(0)]],
    device const %s* position [[buffer(1)]],
    device %s* out [[buffer(2)]],
    uint pair_id [[thread_position_in_grid]]) {
    if (pair_id >= %d) return;
    const uint width = %d;
    const uint pair_in_row = pair_id %% (width / 2);
    const uint row = pair_id / (width / 2);
    const uint first = row * width + pair_in_row;
    const uint second = first + width / 2;
    const float exponent = float(2 * pair_in_row) / float(width);
    const float angle = float(position[0]) / pow(%.9g, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const float x0 = float(input[first]);
    const float x1 = float(input[second]);
    out[first] = %s(x0 * cosine - x1 * sine);
    out[second] = %s(x0 * sine + x1 * cosine);
}
`, kernelName, precision, positionType, precision, pairs, width, theta, precision, precision)
}
