package cudafuse

import (
	"fmt"
	"strings"
)

// compileSumKernel uses the same checked indexed expression as FuseLazy.
// FP32 expression values accumulate sequentially in double precision, then
// convert once to FP32, matching the reference sum policy.
func compileSumKernel(inputs []*LazyNode, expression, output *LazyNode, name string) (FusedKernel, error) {
	// Validation also supplies collision checks and exact ordered parameters.
	kernel, err := compileIndexedLazyRegion(inputs, []*LazyNode{expression}, name)
	if err != nil {
		return FusedKernel{}, err
	}
	count, err := indexedElements(expression)
	if err != nil {
		return FusedKernel{}, err
	}
	if output == nil || output.Op != "sum" || !identifier(output.Name()) {
		return FusedKernel{}, fmt.Errorf("cudafuse: missing sum output")
	}
	kernel.Outputs = []string{output.Name()}
	var source strings.Builder
	fmt.Fprintln(&source, "#include <cuda_runtime.h>")
	fmt.Fprintf(&source, "extern \"C\" __global__ void %s(\n", name)
	for _, in := range inputs {
		fmt.Fprintf(&source, "    const float* __restrict__ %s,\n", in.Name())
	}
	fmt.Fprintf(&source, "    float* __restrict__ out_%s,\n    int n) {\n", output.Name())
	fmt.Fprintln(&source, "    if (n != 1 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;")
	fmt.Fprintln(&source, "    if (blockIdx.x != 0 || threadIdx.x != 0) return;")
	fmt.Fprintln(&source, "    double sum_value = 0.0;")
	if count != 0 {
		fmt.Fprintln(&source, "#pragma unroll 1")
		fmt.Fprintf(&source, "    for (unsigned long long idx = 0; idx < %dULL; idx++) {\n", count)
		external := make(map[*LazyNode]bool)
		for _, in := range inputs {
			external[in] = true
		}
		value := newIndexedExpression(&source, external).emit(expression, "idx")
		fmt.Fprintf(&source, "        sum_value += (double)%s;\n", value)
		fmt.Fprintln(&source, "    }")
	}
	fmt.Fprintf(&source, "    out_%s[0] = (float)sum_value;\n}\n", output.Name())
	kernel.Source = source.String()
	return kernel, nil
}

// compileScanKernel emits an ordered inclusive FP32 prefix. The first value is
// copied without adding an identity, preserving its signed zero.
func compileScanKernel(inputs []*LazyNode, expression, output *LazyNode, name string) (FusedKernel, error) {
	kernel, err := compileIndexedLazyRegion(inputs, []*LazyNode{expression}, name)
	if err != nil {
		return FusedKernel{}, err
	}
	count, err := indexedElements(expression)
	if err != nil {
		return FusedKernel{}, err
	}
	if output == nil || output.Op != "scan" || !identifier(output.Name()) {
		return FusedKernel{}, fmt.Errorf("cudafuse: missing scan output")
	}
	kernel.Outputs = []string{output.Name()}
	var source strings.Builder
	fmt.Fprintln(&source, "#include <cuda_runtime.h>")
	fmt.Fprintf(&source, "extern \"C\" __global__ void %s(\n", name)
	for _, in := range inputs {
		fmt.Fprintf(&source, "    const float* __restrict__ %s,\n", in.Name())
	}
	fmt.Fprintf(&source, "    float* __restrict__ out_%s,\n    int n) {\n", output.Name())
	fmt.Fprintf(&source, "    if (n != %d || n <= 0 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;\n", count)
	fmt.Fprintln(&source, "    if (blockIdx.x != 0 || threadIdx.x != 0) return;")
	if count != 0 {
		fmt.Fprintln(&source, "    float scan_value = 0.0f;")
		fmt.Fprintln(&source, "#pragma unroll 1")
		fmt.Fprintf(&source, "    for (unsigned long long idx = 0; idx < %dULL; idx++) {\n", count)
		external := make(map[*LazyNode]bool)
		for _, in := range inputs {
			external[in] = true
		}
		value := newIndexedExpression(&source, external).emit(expression, "idx")
		emitScanUpdate(&source, value)
		fmt.Fprintf(&source, "        out_%s[idx] = scan_value;\n", output.Name())
		fmt.Fprintln(&source, "    }")
	}
	fmt.Fprintln(&source, "}")
	kernel.Source = source.String()
	return kernel, nil
}

func emitScanUpdate(source *strings.Builder, value string) {
	fmt.Fprintf(source, "        if (idx == 0) scan_value = %s;\n", value)
	fmt.Fprintf(source, "        else scan_value = __fadd_rn(scan_value, %s);\n", value)
}
