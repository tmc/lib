package cudafuse

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// ErrUnsupportedOperator indicates that an operator cannot be lowered by cudafuse.
var ErrUnsupportedOperator = errors.New("cudafuse: unsupported operator for fusion")

// FusedKernel represents a synthesized CUDA C++ kernel with line-level source provenance.
type FusedKernel struct {
	Name    string
	Source  string
	LineMap map[int]cudair.SourceLocation // Maps generated C++ line number -> Go source location
	Inputs  []string
	Outputs []string
}

// Fuse generates a fused CUDA C++ kernel for contiguous row-major float32
// elementwise graphs with equal operand shapes or one-element scalar operands.
// Outputs must share one declared shape. The caller must supply
// distinct input and output buffers and an element count within their extents.
// Launches must use one-dimensional blocks and grids; other launch shapes and
// nonpositive element counts return without accessing buffers.
//
// The source carries #line directives attributing generated lines to the Go
// code they came from, so a binary compiled from it with line information
// disassembles back to the caller's own file and line. Use FuseWithSourceMap
// for source without the directives, along with the mapping they encode.
func Fuse(g *cudair.Graph, kernelName string) (string, error) {
	fk, err := FuseWithSourceMap(g, kernelName)
	if err != nil {
		return "", err
	}
	return fk.SourceWithLineDirectives(), nil
}

// FuseWithSourceMap generates a fused kernel and records exact source mappings from generated lines to Go callsites.
func FuseWithSourceMap(g *cudair.Graph, kernelName string) (FusedKernel, error) {
	var fk FusedKernel
	if g == nil {
		return fk, fmt.Errorf("cudafuse: nil graph")
	}
	fk.Name = kernelName
	fk.LineMap = make(map[int]cudair.SourceLocation)

	nodes := g.Nodes()
	if len(nodes) == 0 {
		return fk, fmt.Errorf("cudafuse: empty graph")
	}
	if err := validateNodes(nodes); err != nil {
		return fk, err
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		return fk, fmt.Errorf("cudafuse: validate graph: %w", err)
	}
	nodes = snapshot.Order()

	var inputs []*cudair.Node
	// Explicit outputs may be consumed inside the region and still needed by
	// another consumer outside it. Graph.Outputs preserves those boundaries and
	// supplies terminal outputs only when the graph has no explicit outputs.
	outputs := snapshot.Outputs()
	if len(outputs) == 0 {
		return fk, fmt.Errorf("cudafuse: graph has no outputs")
	}
	for _, out := range outputs[1:] {
		if !slices.Equal(out.Shape, outputs[0].Shape) {
			return fk, fmt.Errorf("cudafuse: outputs require one declared iteration domain")
		}
	}
	seen := make(map[*cudair.Node]bool)
	for _, out := range outputs {
		if seen[out] {
			return fk, fmt.Errorf("cudafuse: duplicate output %q", out.Name)
		}
		seen[out] = true
	}
	for _, n := range nodes {
		if n.Kind == cudair.KindInput {
			inputs = append(inputs, n)
		}
	}
	if err := validateNames(kernelName, nodes, inputs, outputs); err != nil {
		return fk, err
	}

	for _, in := range inputs {
		fk.Inputs = append(fk.Inputs, in.Name)
	}
	for _, out := range outputs {
		fk.Outputs = append(fk.Outputs, out.Name)
	}

	var lines []string
	lineNo := 0 // number of the last line appended, counting newlines within a chunk
	addLine := func(s string, loc *cudair.SourceLocation) {
		lines = append(lines, s)
		lineNo++
		if loc != nil && (loc.File != "" || loc.Line > 0) {
			fk.LineMap[lineNo] = *loc
		}
		lineNo += strings.Count(s, "\n")
	}

	addLine(`#include <cuda_runtime.h>`, nil)
	addLine(``, nil)
	addLine(`extern "C" __global__ void `+kernelName+`(`, nil)

	// Input parameters
	for _, in := range inputs {
		addLine(fmt.Sprintf("    const float* __restrict__ %s,", in.Name), &in.SourceLoc)
	}
	// Output parameters
	for _, out := range outputs {
		addLine(fmt.Sprintf("    float* __restrict__ out_%s,", out.Name), &out.SourceLoc)
	}
	addLine("    int n\n) {", nil)

	// Grid-stride loop. The index and stride are 64-bit because
	// blockDim.x * blockIdx.x and blockDim.x * gridDim.x can exceed 32 bits
	// even though n cannot. NVRTC unrolls the loop for sm_100 and later and
	// sizes the unrolled loop with a 64-bit division, a __cuda_sm20_div_u64
	// call. On GB10 that left fused_gate executing 91 instructions per warp
	// outside its Go lines, against 29 on them; not unrolled, it executes 31.
	addLine("    if (n <= 0 || blockDim.x == 0 || gridDim.x == 0 || blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1) return;", nil)
	addLine("#pragma unroll 1", nil)
	addLine("    for (unsigned long long idx = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x; idx < (unsigned int)n; idx += (unsigned long long)blockDim.x * gridDim.x) {", nil)

	// Load inputs into local variables
	for _, in := range inputs {
		index := "idx"
		if count, _ := shapeElements(in.Shape); count == 1 {
			index = "0"
		}
		addLine(fmt.Sprintf("        float v_%s = %s[%s];", in.Name, in.Name, index), &in.SourceLoc)
	}

	// Compute intermediate expressions
	for _, n := range nodes {
		if n.Kind == cudair.KindInput {
			continue
		}

		var expr string
		switch {
		case n.Op == "relu":
			expr = fmt.Sprintf("(v_%s > 0.0f) ? v_%s : 0.0f", n.Inputs[0].Name, n.Inputs[0].Name)
		case n.Op == "silu":
			expr = fmt.Sprintf("v_%s / (1.0f + expf(-v_%s))", n.Inputs[0].Name, n.Inputs[0].Name)
		case n.Op == "exp":
			expr = fmt.Sprintf("expf(v_%s)", n.Inputs[0].Name)
		case n.Op == "sqrt":
			expr = fmt.Sprintf("sqrtf(v_%s)", n.Inputs[0].Name)
		case n.Op == "neg":
			expr = fmt.Sprintf("-v_%s", n.Inputs[0].Name)
		case n.Op == "add":
			expr = fmt.Sprintf("v_%s + v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case n.Op == "sub":
			expr = fmt.Sprintf("v_%s - v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case n.Op == "mul":
			expr = fmt.Sprintf("v_%s * v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case n.Op == "div":
			expr = fmt.Sprintf("v_%s / v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		default:
			return fk, fmt.Errorf("%w: %s", ErrUnsupportedOperator, n.Op)
		}

		addLine(fmt.Sprintf("        float v_%s = %s;", n.Name, expr), &n.SourceLoc)
	}

	// Store outputs
	for _, out := range outputs {
		addLine(fmt.Sprintf("        out_%s[idx] = v_%s;", out.Name, out.Name), &out.SourceLoc)
	}

	addLine("    }", nil)
	addLine("}", nil)

	fk.Source = strings.Join(lines, "\n")
	return fk, nil
}

// validateNodes checks the representation consumed by the scalar emitter.
// Snapshot supplies ownership and cycle checks, but permits more operators,
// layouts and types than this emitter implements.
func validateNodes(nodes []*cudair.Node) error {
	for _, n := range nodes {
		if n == nil {
			return fmt.Errorf("cudafuse: nil node")
		}
		arity, kind := 0, cudair.KindInput
		switch n.Op {
		case "input":
		case "relu", "silu", "exp", "sqrt", "neg":
			arity, kind = 1, cudair.KindUnary
		case "add", "sub", "mul", "div":
			arity, kind = 2, cudair.KindBinary
		default:
			detail := ""
			switch {
			case strings.HasPrefix(n.Op, "rmsnorm"):
				detail = " (RMSNorm reduction not implemented)"
			case strings.HasPrefix(n.Op, "rope"):
				detail = " (RoPE transformation not implemented)"
			case n.Op == "matmul":
				detail = " (matmul cannot be lowered as elementwise)"
			}
			return fmt.Errorf("%w: %s%s", ErrUnsupportedOperator, n.Op, detail)
		}
		if n.Kind != kind || len(n.Inputs) != arity {
			return fmt.Errorf("cudafuse: node %q requires kind %s and %d inputs", n.Name, kind, arity)
		}
		for _, in := range n.Inputs {
			if in == nil {
				return fmt.Errorf("cudafuse: node %q has nil input", n.Name)
			}
		}
		if kind == cudair.KindUnary && !slices.Equal(n.Shape, n.Inputs[0].Shape) {
			return fmt.Errorf("%w: unary node %q changes shape", cudair.ErrShapeMismatch, n.Name)
		}
		if kind == cudair.KindBinary {
			shape, err := binaryShape(n.Inputs[0].Shape, n.Inputs[1].Shape)
			if err != nil {
				return err
			}
			if !slices.Equal(n.Shape, shape) {
				return fmt.Errorf("%w: binary node %q output", cudair.ErrShapeMismatch, n.Name)
			}
		}
		if n.Type != cudair.Float32 {
			return fmt.Errorf("cudafuse: node %q requires float32, got %s: %w", n.Name, n.Type, cudair.ErrTypeMismatch)
		}
		if n.Layout != cudashape.LayoutRowMajor || !cudashape.IsContiguousRowMajor(n.Shape, n.Strides) {
			return fmt.Errorf("cudafuse: node %q requires contiguous row-major layout", n.Name)
		}
		if n.Effect != cudair.EffectReadOnly || n.StorageAlias != nil {
			return fmt.Errorf("cudafuse: node %q has unsupported storage effects", n.Name)
		}
	}
	return nil
}

func validateNames(kernel string, nodes, inputs, outputs []*cudair.Node) error {
	// Reserve language tokens and the CUDA/runtime names used by the emitter.
	used := make(map[string]bool)
	for _, name := range strings.Fields(`alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public register reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while xor xor_eq blockDim blockIdx threadIdx gridDim expf sqrtf n idx`) {
		used[name] = true
	}
	claim := func(name string) error {
		if !identifier(name) || used[name] {
			return fmt.Errorf("cudafuse: invalid or conflicting generated identifier %q", name)
		}
		used[name] = true
		return nil
	}
	if err := claim(kernel); err != nil {
		return err
	}
	for _, n := range nodes {
		if !identifier(n.Name) {
			return fmt.Errorf("cudafuse: invalid node name %q", n.Name)
		}
		if err := claim("v_" + n.Name); err != nil {
			return err
		}
	}
	for _, n := range inputs {
		if err := claim(n.Name); err != nil {
			return err
		}
	}
	for _, n := range outputs {
		if err := claim("out_" + n.Name); err != nil {
			return err
		}
	}
	return nil
}

func identifier(s string) bool {
	// Leading underscores and double underscores are reserved to C++/CUDA.
	if s == "" || strings.HasPrefix(s, "_") || strings.Contains(s, "__") {
		return false
	}
	for i, c := range s {
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && (c >= '0' && c <= '9' || c == '_') {
			continue
		}
		return false
	}
	return true
}
