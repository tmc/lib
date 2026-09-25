package advise

import (
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func isElementwise(ev timeline.Event) bool {
	name := strings.ToLower(ev.Name)
	if strings.Contains(name, "gemm") || strings.Contains(name, "matmul") ||
		strings.Contains(name, "matvec") || strings.Contains(name, "gemv") ||
		strings.Contains(name, "attention") || strings.Contains(name, "conv") ||
		strings.Contains(name, "allreduce") || strings.Contains(name, "reduce") ||
		strings.Contains(name, "reduction") || strings.Contains(name, "norm") ||
		strings.Contains(name, "softmax") || strings.Contains(name, "gather") ||
		strings.Contains(name, "scatter") || strings.Contains(name, "scan") {
		return false
	}
	keywords := []string{
		"swiglu", "silu", "gelu", "relu", "sigmoid", "tanh", "activation",
		"quant", "quantize", "dequant", "pointwise", "elementwise",
		"add", "mul", "scale", "bias", "dropout", "cast", "convert",
	}
	for _, kw := range keywords {
		if strings.Contains(name, kw) {
			return true
		}
	}
	return false
}

func isNormKernel(name string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "gemm") || strings.Contains(n, "gemv") || strings.Contains(n, "matvec") || strings.Contains(n, "matmul") {
		return false
	}
	return strings.Contains(n, "rms") || strings.Contains(n, "layernorm") || strings.Contains(n, "norm")
}

func isQuantKernel(name string) bool {
	n := strings.ToLower(name)
	if strings.Contains(n, "gemm") || strings.Contains(n, "gemv") || strings.Contains(n, "matvec") || strings.Contains(n, "matmul") {
		return false
	}
	return strings.Contains(n, "quant") || strings.Contains(n, "quantize")
}

func isGEMVKernel(name string) bool {
	n := strings.ToLower(name)
	return strings.Contains(n, "matvec") || strings.Contains(n, "gemv")
}

func matchingShapesOrBuffers(ev1, ev2 timeline.Event) bool {
	if ev1.Shape != "" && ev2.Shape != "" {
		return ev1.Shape == ev2.Shape
	}
	if ev1.Buffer != "" && ev2.Buffer != "" {
		return ev1.Buffer == ev2.Buffer
	}
	if ev1.Grid != [3]int{} && ev1.Grid == ev2.Grid && ev1.Block != [3]int{} && ev1.Block == ev2.Block {
		return true
	}
	return false
}

func cleanKernelName(name string) string {
	name = strings.TrimSuffix(name, ".SASS")
	name = strings.TrimSuffix(name, ".PTX")
	name = strings.TrimSuffix(name, ".TILEIR")
	nameLower := strings.ToLower(name)
	switch {
	case strings.Contains(nameLower, "swiglu"):
		return "SwiGLU"
	case strings.Contains(nameLower, "quant"):
		return "quant"
	case strings.Contains(nameLower, "gelu"):
		return "GELU"
	case strings.Contains(nameLower, "silu"):
		return "SiLU"
	case strings.Contains(nameLower, "relu"):
		return "ReLU"
	case strings.Contains(nameLower, "rmsnorm") || strings.Contains(nameLower, "rms"):
		return "RMSNorm"
	case strings.Contains(nameLower, "layernorm"):
		return "LayerNorm"
	default:
		return name
	}
}
