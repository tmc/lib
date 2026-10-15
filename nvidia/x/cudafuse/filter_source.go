package cudafuse

import (
	"fmt"
	"strings"
)

// The ordered FP32 scan represents every prefix of this boolean domain exactly.
const maxFilterElements = 1 << 24

func filterNames(output *LazyNode) []string {
	return []string{output.Name() + "_data", output.Name() + "_predicate", output.Name() + "_prefix", output.Name() + "_status_count"}
}

func compileFilterCount(inputs []*LazyNode, value, predicate, output *LazyNode, name string) (FusedKernel, error) {
	outputs := uniqueExecutionNodes([]*LazyNode{value, predicate})
	k, err := compileIndexedLazyRegion(inputs, outputs, name)
	if err != nil {
		return FusedKernel{}, err
	}
	n, err := indexedElements(value)
	if err != nil || n > maxFilterElements {
		return FusedKernel{}, fmt.Errorf("cudafuse: Filter count exceeds exact FP32 prefix domain")
	}
	k.Outputs = filterNames(output)
	var source strings.Builder
	fmt.Fprintln(&source, "#include <cuda_runtime.h>")
	fmt.Fprintf(&source, "extern \"C\" __global__ void %s(\n", name)
	for _, in := range inputs {
		fmt.Fprintf(&source, "    const float* __restrict__ %s,\n", in.Name())
	}
	for _, out := range k.Outputs {
		fmt.Fprintf(&source, "    float* __restrict__ out_%s,\n", out)
	}
	fmt.Fprintln(&source, "    int n) {")
	fmt.Fprintf(&source, "    if (n != %d || n <= 0 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;\n", n)
	fmt.Fprintln(&source, "    if (blockIdx.x != 0 || threadIdx.x != 0) return;")
	fmt.Fprintln(&source, "    float scan_value = 0.0f;")
	fmt.Fprintln(&source, "    unsigned int invalid_predicate = 0;")
	fmt.Fprintln(&source, "#pragma unroll 1")
	fmt.Fprintf(&source, "    for (unsigned long long idx = 0; idx < %dULL; idx++) {\n", n)
	external := make(map[*LazyNode]bool)
	for _, in := range inputs {
		external[in] = true
	}
	emitter := newIndexedExpression(&source, external)
	v, p := emitter.emit(value, "idx"), emitter.emit(predicate, "idx")
	fmt.Fprintf(&source, "        out_%s[idx] = %s;\n", k.Outputs[0], v)
	fmt.Fprintf(&source, "        out_%s[idx] = %s;\n", k.Outputs[1], p)
	fmt.Fprintf(&source, "        invalid_predicate |= !(%s == 0.0f || %s == 1.0f);\n", p, p)
	fmt.Fprintf(&source, "        float filter_selected = %s == 1.0f ? 1.0f : 0.0f;\n", p)
	emitScanUpdate(&source, "filter_selected")
	fmt.Fprintf(&source, "        out_%s[idx] = scan_value;\n", k.Outputs[2])
	fmt.Fprintln(&source, "    }")
	fmt.Fprintf(&source, "    out_%s[0] = (float)invalid_predicate;\n", k.Outputs[3])
	fmt.Fprintf(&source, "    out_%s[1] = scan_value;\n}\n", k.Outputs[3])
	k.Source = source.String()
	return k, nil
}

func compileFilterScatter(output *LazyNode, inputCount, selected int, name string) FusedKernel {
	inputs := filterNames(output)[:3]
	var source strings.Builder
	fmt.Fprintln(&source, "#include <cuda_runtime.h>")
	fmt.Fprintf(&source, "extern \"C\" __global__ void %s(\n", name)
	for _, in := range inputs {
		fmt.Fprintf(&source, "    const float* __restrict__ %s,\n", in)
	}
	fmt.Fprintf(&source, "    float* __restrict__ out_%s,\n    int n) {\n", output.Name())
	fmt.Fprintf(&source, "    if (n != %d || n <= 0 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;\n", inputCount)
	fmt.Fprintln(&source, "#pragma unroll 1")
	fmt.Fprintln(&source, "    for (unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x; idx < (unsigned int)n; idx += (unsigned long long)blockDim.x * gridDim.x) {")
	fmt.Fprintf(&source, "        if (%s[idx] == 1.0f) {\n", inputs[1])
	fmt.Fprintf(&source, "            unsigned int destination = (unsigned int)%s[idx] - 1u;\n", inputs[2])
	fmt.Fprintf(&source, "            if (destination < %du) out_%s[destination] = %s[idx];\n", selected, output.Name(), inputs[0])
	fmt.Fprintln(&source, "        }\n    }\n}")
	return FusedKernel{Name: name, Source: source.String(), Inputs: inputs, Outputs: []string{output.Name()}}
}
