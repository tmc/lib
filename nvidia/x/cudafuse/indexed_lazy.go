package cudafuse

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// FuseLazy generates a row-major FP32 kernel for a lazy region. Inputs identify
// external tensors; traversal stops at those nodes. Outputs must have the same
// element count, bounded by int32. Unary and binary operations, reshapes, and
// transposes are evaluated without intermediate buffers.
//
// The caller must use distinct input and output buffers, a one-dimensional
// launch, and the exact compiled element count. LazyNode has no source location,
// so the returned LineMap is empty. This function generates source only.
func FuseLazy(inputs, outputs []*LazyNode, kernelName string) (FusedKernel, error) {
	return compileIndexedLazyRegion(inputs, outputs, kernelName)
}

// compileIndexedLazyRegion evaluates lazy expressions at an output index.
// Transposes change that index for their operands; reshapes preserve it.
func compileIndexedLazyRegion(inputs, outputs []*LazyNode, name string) (FusedKernel, error) {
	var kernel FusedKernel
	if len(outputs) == 0 {
		return kernel, fmt.Errorf("cudafuse: indexed region has no outputs")
	}
	external := make(map[*LazyNode]bool)
	for _, input := range inputs {
		if input == nil || external[input] {
			return kernel, fmt.Errorf("cudafuse: nil or duplicate indexed input")
		}
		external[input] = true
	}
	count, err := indexedElements(outputs[0])
	if err != nil {
		return kernel, err
	}
	state := make(map[*LazyNode]int)
	var nodes []*LazyNode
	var visit func(*LazyNode) error
	visit = func(node *LazyNode) error {
		if node == nil || state[node] == 1 {
			return fmt.Errorf("cudafuse: nil or cyclic indexed node")
		}
		if state[node] == 2 {
			return nil
		}
		n, err := indexedElements(node)
		if err != nil {
			return err
		}
		if n != count || node.DType != cudair.Float32 {
			return fmt.Errorf("cudafuse: indexed node %s requires float32 and %d elements", node.Name(), count)
		}
		if !external[node] {
			if err := validateIndexedOperator(node); err != nil {
				return err
			}
			state[node] = 1
			for _, input := range node.Inputs {
				if err := visit(input); err != nil {
					return err
				}
			}
		}
		state[node] = 2
		nodes = append(nodes, node)
		return nil
	}
	seenOutput := make(map[*LazyNode]bool)
	for _, output := range outputs {
		if seenOutput[output] {
			return kernel, fmt.Errorf("cudafuse: duplicate indexed output")
		}
		seenOutput[output] = true
		if err := visit(output); err != nil {
			return kernel, err
		}
	}
	for _, input := range inputs {
		if state[input] != 2 {
			return kernel, fmt.Errorf("cudafuse: unused indexed input")
		}
	}
	// Reuse the emitter's identifier and generated-name collision checks.
	byNode := make(map[*LazyNode]*cudair.Node)
	var named, namedInputs, namedOutputs []*cudair.Node
	for _, node := range nodes {
		value := &cudair.Node{Name: node.Name()}
		byNode[node] = value
		named = append(named, value)
	}
	for _, input := range inputs {
		namedInputs = append(namedInputs, byNode[input])
	}
	for _, output := range outputs {
		namedOutputs = append(namedOutputs, byNode[output])
	}
	if err := validateNames(name, named, namedInputs, namedOutputs); err != nil {
		return kernel, err
	}
	kernel.Name = name
	kernel.LineMap = make(map[int]cudair.SourceLocation)
	var source strings.Builder
	fmt.Fprintln(&source, "#include <cuda_runtime.h>")
	fmt.Fprintf(&source, "extern \"C\" __global__ void %s(\n", name)
	for _, input := range inputs {
		kernel.Inputs = append(kernel.Inputs, input.Name())
		fmt.Fprintf(&source, "    const float* __restrict__ %s,\n", input.Name())
	}
	for _, output := range outputs {
		kernel.Outputs = append(kernel.Outputs, output.Name())
		fmt.Fprintf(&source, "    float* __restrict__ out_%s,\n", output.Name())
	}
	fmt.Fprintln(&source, "    int n) {")
	fmt.Fprintf(&source, "    if (n != %d || n <= 0 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;\n", count)
	if count != 0 {
		fmt.Fprintln(&source, "#pragma unroll 1")
		fmt.Fprintln(&source, "    for (unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x; idx < (unsigned int)n; idx += (unsigned long long)blockDim.x * gridDim.x) {")
		type valueKey struct {
			node  *LazyNode
			index string
		}
		values := make(map[valueKey]string)
		next := 0
		var emit func(*LazyNode, string) string
		emit = func(node *LazyNode, index string) string {
			key := valueKey{node, index}
			if value := values[key]; value != "" {
				return value
			}
			var expression string
			switch {
			case external[node]:
				expression = fmt.Sprintf("%s[%s]", node.Name(), index)
			case node.Op == "transpose":
				perm := node.Attrs["permutation"].([]int)
				outStride, inStride := rowMajorStrides(node.Shape), rowMajorStrides(node.Inputs[0].Shape)
				var terms []string
				for axis, inputAxis := range perm {
					terms = append(terms, fmt.Sprintf("((%s / %dULL) %% %dULL) * %dULL", index, outStride[axis], node.Shape[axis], inStride[inputAxis]))
				}
				mapped := "0ULL"
				if len(terms) != 0 {
					mapped = strings.Join(terms, " + ")
				}
				next++
				indexName := fmt.Sprintf("index_%d", next)
				fmt.Fprintf(&source, "        unsigned long long %s = %s;\n", indexName, mapped)
				expression = emit(node.Inputs[0], indexName)
			case node.Op == "reshape":
				expression = emit(node.Inputs[0], index)
			default:
				a := emit(node.Inputs[0], index)
				switch node.Op {
				case "relu":
					expression = fmt.Sprintf("(%s > 0.0f ? %s : 0.0f)", a, a)
				case "silu":
					expression = fmt.Sprintf("%s / (1.0f + expf(-%s))", a, a)
				case "exp", "sqrt":
					expression = fmt.Sprintf("%sf(%s)", node.Op, a)
				case "neg":
					expression = "-" + a
				default:
					operator := map[string]string{"add": "+", "sub": "-", "mul": "*", "div": "/"}[node.Op]
					expression = a + " " + operator + " " + emit(node.Inputs[1], index)
				}
			}
			next++
			value := fmt.Sprintf("v_%s_at_%d", node.Name(), next)
			fmt.Fprintf(&source, "        float %s = %s;\n", value, expression)
			values[key] = value
			return value
		}
		for _, output := range outputs {
			value := emit(output, "idx")
			fmt.Fprintf(&source, "        out_%s[idx] = %s;\n", output.Name(), value)
		}
		fmt.Fprintln(&source, "    }")
	}
	fmt.Fprintln(&source, "}")
	kernel.Source = source.String()
	return kernel, nil
}

func indexedElements(node *LazyNode) (uint64, error) {
	if node == nil {
		return 0, fmt.Errorf("cudafuse: nil indexed node")
	}
	if node.Err != nil {
		return 0, fmt.Errorf("cudafuse: indexed node %s: %w", node.Name(), node.Err)
	}
	empty := false
	for _, dimension := range node.Shape {
		if dimension < 0 {
			return 0, fmt.Errorf("cudafuse: negative indexed dimension")
		}
		empty = empty || dimension == 0
	}
	if empty {
		return 0, nil
	}
	count, err := shapeElements(node.Shape)
	if err != nil || count > math.MaxInt32 {
		return 0, fmt.Errorf("cudafuse: indexed extent exceeds int32")
	}
	return count, nil
}

func validateIndexedOperator(node *LazyNode) error {
	arity := 1
	switch node.Op {
	case "relu", "silu", "exp", "sqrt", "neg", "reshape", "transpose":
	case "add", "sub", "mul", "div":
		arity = 2
	default:
		return fmt.Errorf("cudafuse: indexed operator %q: %w", node.Op, ErrUnsupportedOperator)
	}
	if len(node.Inputs) != arity {
		return fmt.Errorf("cudafuse: indexed operator %s requires %d inputs", node.Op, arity)
	}
	for _, input := range node.Inputs {
		if input == nil {
			return fmt.Errorf("cudafuse: indexed operator %s has nil input", node.Op)
		}
	}
	switch node.Op {
	case "transpose":
		perm, ok := node.Attrs["permutation"].([]int)
		if !ok || len(node.Attrs) != 1 {
			return fmt.Errorf("cudafuse: indexed transpose requires permutation")
		}
		shape, err := transposeShape(node.Inputs[0].Shape, perm)
		if err != nil || !slices.Equal(shape, node.Shape) {
			return fmt.Errorf("cudafuse: indexed transpose shape or permutation mismatch")
		}
	case "reshape":
		shape, ok := node.Attrs["target_shape"].([]int)
		if !ok || len(node.Attrs) != 1 || !slices.Equal(shape, node.Shape) {
			return fmt.Errorf("cudafuse: indexed reshape requires target shape")
		}
	default:
		if len(node.Attrs) != 0 {
			return fmt.Errorf("cudafuse: unexpected indexed operator attributes")
		}
		for _, input := range node.Inputs {
			if !slices.Equal(input.Shape, node.Shape) {
				return fmt.Errorf("cudafuse: indexed operator %s requires matching shapes", node.Op)
			}
		}
	}
	return nil
}

func rowMajorStrides(shape []int) []uint64 {
	strides := make([]uint64, len(shape))
	stride := uint64(1)
	for axis := len(shape) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= uint64(shape[axis])
	}
	return strides
}
