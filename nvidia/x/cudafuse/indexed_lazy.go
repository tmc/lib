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
// element count, bounded by int32. Equal-shape or scalar binary operations,
// unary operations, reshapes, transposes, and periodic Cycle views are evaluated
// by their own checked index mappings. A Cycle execution boundary retains its
// owned output buffer when this source is used by the deferred executor.
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
		_, err := indexedElements(node)
		if err != nil {
			return err
		}
		if node.DType != cudair.Float32 {
			return fmt.Errorf("cudafuse: indexed node %s requires float32: %w", node.Name(), cudair.ErrTypeMismatch)
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
		n, err := indexedElements(output)
		if err != nil || n != count {
			return kernel, fmt.Errorf("cudafuse: indexed outputs require the same iteration extent")
		}
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
		emitter := newIndexedExpression(&source, external)
		for _, output := range outputs {
			value := emitter.emit(output, "idx")
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
	count, err := shapeElements(node.Shape)
	if err != nil {
		return 0, fmt.Errorf("cudafuse: indexed shape: %w", err)
	}
	if count > math.MaxInt32 {
		return 0, fmt.Errorf("cudafuse: indexed extent exceeds int32")
	}
	return count, nil
}

func validateIndexedOperator(node *LazyNode) error {
	if err := validateLazyShape(node); err != nil {
		return fmt.Errorf("cudafuse: indexed operator %s: %w", node.Op, err)
	}
	arity := 1
	switch node.Op {
	case "relu", "silu", "exp", "sqrt", "neg", "reshape", "transpose", "cycle":
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
	case "reshape", "cycle":
		shape, ok := node.Attrs["target_shape"].([]int)
		if !ok || len(node.Attrs) != 1 || !slices.Equal(shape, node.Shape) {
			return fmt.Errorf("cudafuse: indexed reshape requires target shape")
		}
	default:
		if len(node.Attrs) != 0 {
			return fmt.Errorf("cudafuse: unexpected indexed operator attributes")
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

type indexedValueKey struct {
	node  *LazyNode
	index string
}

type indexedExpression struct {
	source   *strings.Builder
	external map[*LazyNode]bool
	values   map[indexedValueKey]string
	next     int
}

func newIndexedExpression(source *strings.Builder, external map[*LazyNode]bool) *indexedExpression {
	return &indexedExpression{source: source, external: external, values: make(map[indexedValueKey]string)}
}

func (e *indexedExpression) emit(node *LazyNode, index string) string {
	// Normalize a whole scalar expression, including its views and operands.
	// Non-scalar transpose and Cycle expressions retain their own index domain.
	if count, _ := indexedElements(node); count == 1 {
		index = "0ULL"
	}
	key := indexedValueKey{node, index}
	if value := e.values[key]; value != "" {
		return value
	}
	var expression string
	switch {
	case e.external[node]:
		if index == "0ULL" {
			e.next++
			index = fmt.Sprintf("index_%d", e.next)
			fmt.Fprintf(e.source, "        unsigned long long %s = 0ULL;\n", index)
		}
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
		e.next++
		indexName := fmt.Sprintf("index_%d", e.next)
		fmt.Fprintf(e.source, "        unsigned long long %s = %s;\n", indexName, mapped)
		expression = e.emit(node.Inputs[0], indexName)
	case node.Op == "reshape":
		expression = e.emit(node.Inputs[0], index)
	case node.Op == "cycle":
		count, _ := indexedElements(node.Inputs[0])
		expression = e.emit(node.Inputs[0], fmt.Sprintf("(%s %% %dULL)", index, count))
	default:
		a := e.emit(node.Inputs[0], index)
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
			expression = a + " " + operator + " " + e.emit(node.Inputs[1], index)
		}
	}
	e.next++
	value := fmt.Sprintf("v_%s_at_%d", node.Name(), e.next)
	fmt.Fprintf(e.source, "        float %s = %s;\n", value, expression)
	e.values[key] = value
	return value
}
