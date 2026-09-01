//go:build darwin

package aneegraph

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

// ANEOptions configures Apple Neural Engine MIL IR code generation.
type ANEOptions struct {
	BatchSize        int
	SpatialDim       int
	HiddenDim        int
	IntermediateDim  int
	WeightBlobOffset uint64
}

// GenerateMIL generates a complete Model Intermediate Language (MIL) program for ANE compilation from an E-Graph Expr AST.
func GenerateMIL(expr *egraph.Expr, opts ANEOptions) (string, error) {
	if expr == nil {
		return "", fmt.Errorf("aneegraph: nil expr passed to GenerateMIL")
	}
	if opts.SpatialDim <= 0 {
		opts.SpatialDim = 1
	}
	if opts.HiddenDim <= 0 {
		opts.HiddenDim = 64
	}
	if opts.IntermediateDim <= 0 {
		opts.IntermediateDim = 128
	}

	buildInfo := `[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]`

	var sb strings.Builder
	sb.WriteString("program(1.3)\n")
	sb.WriteString(buildInfo + "\n")
	sb.WriteString("{\n")

	switch expr.Op {
	case "fused_swiglu_gemv":
		sb.WriteString(generateMILSwiGLU(opts))
	case "fused_rmsnorm_gemv":
		sb.WriteString(generateMILRMSNormGEMV(opts))
	default:
		sb.WriteString(generateMILGEMV(opts))
	}

	sb.WriteString("}\n")
	return sb.String(), nil
}

func generateMILSwiGLU(opts ANEOptions) string {
	return fmt.Sprintf(`    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];

        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];

        // 1. Gate projection GEMV
        tensor<fp16, [%d, %d, 1, 1]> W_gate = const()[name = string("W_gate"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(%d)))];
        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_gate, x = x16)[name = string("conv_gate")];

        // 2. Up projection GEMV
        tensor<fp16, [%d, %d, 1, 1]> W_up = const()[name = string("W_up"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(%d)))];
        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_up, x = x16)[name = string("conv_up")];

        // 3. Fused SiLU activation and elementwise product on ANE
        tensor<fp16, [1, %d, 1, %d]> silu_gate = sigmoid(x = gate)[name = string("sig_gate")];
        tensor<fp16, [1, %d, 1, %d]> act = mul(x = gate, y = silu_gate)[name = string("swish_gate")];
        tensor<fp16, [1, %d, 1, %d]> ffn_mid = mul(x = act, y = up)[name = string("swiglu_product")];

        // 4. Down projection GEMV
        tensor<fp16, [%d, %d, 1, 1]> W_down = const()[name = string("W_down"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(%d)))];
        tensor<fp16, [1, %d, 1, %d]> out16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W_down, x = ffn_mid)[name = string("conv_down")];

        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = out16)[name = string("cast_out")];
    } -> (y);
`,
		opts.HiddenDim, opts.SpatialDim,
		opts.HiddenDim, opts.SpatialDim,
		opts.IntermediateDim, opts.HiddenDim, opts.IntermediateDim, opts.HiddenDim, opts.WeightBlobOffset,
		opts.IntermediateDim, opts.SpatialDim,
		opts.IntermediateDim, opts.HiddenDim, opts.IntermediateDim, opts.HiddenDim, opts.WeightBlobOffset+uint64(opts.IntermediateDim*opts.HiddenDim*2),
		opts.IntermediateDim, opts.SpatialDim,
		opts.IntermediateDim, opts.SpatialDim,
		opts.IntermediateDim, opts.SpatialDim,
		opts.IntermediateDim, opts.SpatialDim,
		opts.HiddenDim, opts.IntermediateDim, opts.HiddenDim, opts.IntermediateDim, opts.WeightBlobOffset+uint64(opts.IntermediateDim*opts.HiddenDim*4),
		opts.HiddenDim, opts.SpatialDim,
		opts.HiddenDim, opts.SpatialDim,
	)
}

func generateMILRMSNormGEMV(opts ANEOptions) string {
	return fmt.Sprintf(`    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];

        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(%d)))];
        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    } -> (y);
`,
		opts.HiddenDim, opts.SpatialDim,
		opts.HiddenDim, opts.SpatialDim,
		opts.HiddenDim, opts.HiddenDim, opts.HiddenDim, opts.HiddenDim, opts.WeightBlobOffset,
		opts.HiddenDim, opts.SpatialDim,
		opts.HiddenDim, opts.SpatialDim,
	)
}

func generateMILGEMV(opts ANEOptions) string {
	return generateMILRMSNormGEMV(opts)
}
