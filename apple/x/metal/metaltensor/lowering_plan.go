package metaltensor

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalir"
	"github.com/tmc/lib/apple/x/metal/metalshape"
)

// TensorBufferKind identifies whether a stage buffer is supplied by the
// caller or produced by an earlier stage.
type TensorBufferKind uint8

const (
	ExternalTensorBuffer TensorBufferKind = iota
	IntermediateTensorBuffer
)

// TensorBufferRef identifies one input or intermediate buffer in a plan.
// External references preserve the input node ID when the source payload has
// one; symbolic bucket payloads encode it in Name. Intermediate IDs are local
// to one plan.
type TensorBufferRef struct {
	Kind TensorBufferKind
	ID   int
	Name string
}

// TensorDispatchSize is a three-dimensional Metal grid or threadgroup size.
type TensorDispatchSize struct {
	Width  uint64
	Height uint64
	Depth  uint64
}

// TensorKernelStage describes one separately compiled and dispatched kernel.
// Inputs and Output identify buffers; on Darwin,
// ExecuteTensorExecutionPlan allocates intermediate outputs and dispatches
// the stages.
type TensorKernelStage struct {
	Op          string
	KernelName  string
	Source      string
	Inputs      []TensorBufferRef
	Output      TensorBufferRef
	Shape       []int
	DataType    metalir.DataType
	Grid        TensorDispatchSize
	Threadgroup TensorDispatchSize
}

// TensorBufferLayout describes the shape and type required for an external
// input buffer in an execution plan.
type TensorBufferLayout struct {
	Buffer   TensorBufferRef
	Shape    []int
	DataType metalir.DataType
}

// TensorExecutionPlan is a sequence of kernel stages with caller-supplied
// external input buffers. On Darwin, ExecuteTensorExecutionPlan allocates
// intermediate outputs and dispatches the stages. The current lowering
// supports row-major rank-2 matmul, row-wise RMSNorm, split-half RoPE,
// and contiguous elementwise add, mul, and silu. Other operators fail closed.
// Zero-element outputs have a zero grid
// and are skipped by ExecuteTensorExecutionPlan. Shapes whose element count
// or dimensions exceed MSL's uint indexing range are rejected.
type TensorExecutionPlan struct {
	Stages       []TensorKernelStage
	Inputs       []TensorBufferRef
	InputLayouts []TensorBufferLayout
	Output       TensorBufferRef
}

// LowerTensorExecutionPlan lowers an extracted tensor expression into
// separate MSL kernels. dimensions binds symbolic shape expressions in
// bucket-specialized candidates. Each generated stage is one dispatch; the
// returned plan does not imply a single-kernel implementation.
func LowerTensorExecutionPlan(expr *egraph.Expr, dimensions map[string]int64) (*TensorExecutionPlan, error) {
	if expr == nil {
		return nil, fmt.Errorf("tensor lowering: nil expression")
	}
	state := tensorPlanBuilder{
		dimensions: dimensions,
		buffers:    make(map[*egraph.Expr]TensorBufferRef),
		active:     make(map[*egraph.Expr]bool),
		uses:       tensorExprUseCounts(expr),
	}
	output, err := state.lower(expr)
	if err != nil {
		return nil, err
	}
	return &TensorExecutionPlan{
		Stages:       state.stages,
		Inputs:       state.inputs,
		InputLayouts: state.inputLayouts,
		Output:       output,
	}, nil
}

type tensorPlanBuilder struct {
	dimensions   map[string]int64
	stages       []TensorKernelStage
	inputs       []TensorBufferRef
	inputLayouts []TensorBufferLayout
	inputIndex   map[string]int
	buffers      map[*egraph.Expr]TensorBufferRef
	active       map[*egraph.Expr]bool
	uses         map[*egraph.Expr]int
	fusing       map[*egraph.Expr]bool
}

type tensorExprMetadata struct {
	shape      []int
	dtype      metalir.DataType
	attributes metalir.ScalarAttributes
	identity   string
	inputID    int
	inputName  string
	hasInputID bool
}

func (b *tensorPlanBuilder) lower(expr *egraph.Expr) (TensorBufferRef, error) {
	if expr == nil {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: nil child expression")
	}
	if ref, ok := b.buffers[expr]; ok {
		return ref, nil
	}
	if b.active[expr] {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: expression cycle at %q", expr.Op)
	}
	b.active[expr] = true
	defer delete(b.active, expr)

	metadata, err := tensorExprMetadataOf(expr, b.dimensions)
	if err != nil {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s metadata: %w", expr.Op, err)
	}
	if expr.Sort != metadata.dtype.String() {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s sort %q does not match payload type %s", expr.Op, expr.Sort, metadata.dtype)
	}
	if err := validateTensorMSLShape(metadata.shape); err != nil {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s shape: %w", expr.Op, err)
	}
	if expr.Op == "input" || expr.Op == "constant" {
		if len(expr.Children) != 0 {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s leaf has %d children", expr.Op, len(expr.Children))
		}
		ref, err := b.externalBuffer(expr, metadata)
		if err != nil {
			return TensorBufferRef{}, err
		}
		b.buffers[expr] = ref
		return ref, nil
	}

	want := 0
	switch expr.Op {
	case "add", "mul", "matmul", "rmsnorm", "rope":
		want = 2
	case "silu":
		want = 1
	default:
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: operator %q has no stage lowering", expr.Op)
	}
	if len(expr.Children) != want {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s wants %d inputs, got %d", expr.Op, want, len(expr.Children))
	}
	if expr.Op == "rmsnorm" {
		if !metadata.attributes.HasEpsilon || metadata.attributes.HasTheta || metadata.attributes.Theta != 0 || metadata.attributes.Epsilon <= 0 || math.IsNaN(float64(metadata.attributes.Epsilon)) || math.IsInf(float64(metadata.attributes.Epsilon), 0) {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: rmsnorm has invalid epsilon")
		}
	} else if expr.Op == "rope" {
		if !metadata.attributes.HasTheta || metadata.attributes.HasEpsilon || metadata.attributes.Epsilon != 0 || !positiveFinite(float64(metadata.attributes.Theta)) {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: rope has invalid theta")
		}
	} else if metadata.attributes != (metalir.ScalarAttributes{}) {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s has unsupported scalar attributes", expr.Op)
	}

	inputRefs := make([]TensorBufferRef, want)
	inputMetadata := make([]tensorExprMetadata, want)
	for i, child := range expr.Children {
		if expr.Op == "matmul" || expr.Op == "rmsnorm" || expr.Op == "rope" {
			ref, err := b.lower(child)
			if err != nil {
				return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s input %d: %w", expr.Op, i, err)
			}
			inputRefs[i] = ref
		}
		inputMetadata[i], err = tensorExprMetadataOf(child, b.dimensions)
		if err != nil {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s input %d metadata: %w", expr.Op, i, err)
		}
		if child.Sort != inputMetadata[i].dtype.String() || inputMetadata[i].dtype != metadata.dtype && !(expr.Op == "rope" && i == 1) {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s input %d has incompatible type", expr.Op, i)
		}
		inputCount, err := tensorElementCount(inputMetadata[i].shape)
		if err != nil {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s input %d size: %w", expr.Op, i, err)
		}
		if inputCount > math.MaxUint32 {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s input %d exceeds the MSL uint index range", expr.Op, i)
		}
	}
	if err := validateTensorStageShape(expr.Op, metadata.shape, inputMetadata); err != nil {
		return TensorBufferRef{}, err
	}
	if _, err := tensorElementCount(metadata.shape); err != nil {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s output size: %w", expr.Op, err)
	}
	count, _ := tensorElementCount(metadata.shape)
	if count > math.MaxUint32 {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s output exceeds the MSL uint index range", expr.Op)
	}
	if metadata.dtype != metalir.Float32 && metadata.dtype != metalir.Float16 {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s does not support MSL type %s", expr.Op, metadata.dtype)
	}

	index := len(b.stages)
	kernelName := fmt.Sprintf("tensor_stage_%d_%s", index, expr.Op)
	precision, err := tensorPrecision(metadata.dtype)
	if err != nil {
		return TensorBufferRef{}, err
	}
	var source string
	stageInputs := append([]TensorBufferRef(nil), inputRefs...)
	if expr.Op == "matmul" {
		source = generateMatMulStageMSL(kernelName, precision, inputMetadata[0].shape, inputMetadata[1].shape, metadata.shape)
	} else if expr.Op == "rmsnorm" {
		source = generateRMSNormStageMSL(kernelName, precision, inputMetadata[0].shape, metadata.attributes.Epsilon)
	} else if expr.Op == "rope" {
		positionType, err := ropePositionType(inputMetadata[1].dtype)
		if err != nil {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: rope: %w", err)
		}
		source = generateRoPEStageMSL(kernelName, precision, positionType, inputMetadata[0].shape, metadata.attributes.Theta)
	} else {
		stagedExpr, refs, err := b.fusedElementwiseExpr(expr, true)
		if err != nil {
			return TensorBufferRef{}, err
		}
		stageInputs = refs
		registry, err := NewTensorOperatorRegistry()
		if err != nil {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: create operator registry: %w", err)
		}
		source, err = egraph.GenerateMSL(stagedExpr, egraph.CodegenOptions{
			KernelName: kernelName,
			Precision:  precision,
			Registry:   registry,
		})
		if err != nil {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: generate %s stage: %w", expr.Op, err)
		}
	}
	output := TensorBufferRef{Kind: IntermediateTensorBuffer, ID: index, Name: fmt.Sprintf("stage_%d_output", index)}
	grid, group := tensorDispatch(metadata.shape, expr.Op)
	b.stages = append(b.stages, TensorKernelStage{
		Op: expr.Op, KernelName: kernelName, Source: source,
		Inputs: append([]TensorBufferRef(nil), stageInputs...), Output: output,
		Shape: copyShape(metadata.shape), DataType: metadata.dtype,
		Grid: grid, Threadgroup: group,
	})
	b.buffers[expr] = output
	return output, nil
}

// fusedElementwiseExpr inlines elementwise children that have exactly one
// use in the extracted expression. A child used more than once is lowered to
// its own buffer, so a fused stage never duplicates its computation. Other
// operators are boundaries and are lowered by the normal stage path.
func (b *tensorPlanBuilder) fusedElementwiseExpr(expr *egraph.Expr, root bool) (*egraph.Expr, []TensorBufferRef, error) {
	if expr == nil {
		return nil, nil, fmt.Errorf("tensor lowering: nil child expression")
	}
	if b.fusing == nil {
		b.fusing = make(map[*egraph.Expr]bool)
	}
	if b.fusing[expr] {
		return nil, nil, fmt.Errorf("tensor lowering: expression cycle at %q", expr.Op)
	}
	b.fusing[expr] = true
	defer delete(b.fusing, expr)
	if !root && (!isTensorElementwise(expr.Op) || b.uses[expr] != 1) {
		ref, err := b.lower(expr)
		if err != nil {
			return nil, nil, err
		}
		return &egraph.Expr{Op: "input", Sort: expr.Sort, Data: ref}, []TensorBufferRef{ref}, nil
	}
	metadata, err := tensorExprMetadataOf(expr, b.dimensions)
	if err != nil {
		return nil, nil, fmt.Errorf("tensor lowering: %s metadata: %w", expr.Op, err)
	}
	want := 1
	if expr.Op == "add" || expr.Op == "mul" {
		want = 2
	}
	if !isTensorElementwise(expr.Op) || len(expr.Children) != want {
		return nil, nil, fmt.Errorf("tensor lowering: invalid fused elementwise operator %q", expr.Op)
	}
	if expr.Sort != metadata.dtype.String() || metadata.attributes != (metalir.ScalarAttributes{}) {
		return nil, nil, fmt.Errorf("tensor lowering: %s has incompatible sort or unsupported attributes", expr.Op)
	}
	childMetadata := make([]tensorExprMetadata, len(expr.Children))
	for i, child := range expr.Children {
		childMetadata[i], err = tensorExprMetadataOf(child, b.dimensions)
		if err != nil {
			return nil, nil, fmt.Errorf("tensor lowering: %s input %d metadata: %w", expr.Op, i, err)
		}
		if child.Sort != childMetadata[i].dtype.String() || childMetadata[i].dtype != metadata.dtype {
			return nil, nil, fmt.Errorf("tensor lowering: %s input %d has incompatible type", expr.Op, i)
		}
	}
	if err := validateTensorStageShape(expr.Op, metadata.shape, childMetadata); err != nil {
		return nil, nil, err
	}
	result := &egraph.Expr{Op: expr.Op, Sort: expr.Sort, Data: expr.Data}
	refs := make([]TensorBufferRef, 0, len(expr.Children))
	seen := make(map[TensorBufferRef]bool)
	for i, child := range expr.Children {
		childExpr, childRefs, err := b.fusedElementwiseExpr(child, false)
		if err != nil {
			return nil, nil, fmt.Errorf("tensor lowering: %s input %d: %w", expr.Op, i, err)
		}
		result.Children = append(result.Children, childExpr)
		for _, ref := range childRefs {
			if !seen[ref] {
				seen[ref] = true
				refs = append(refs, ref)
			}
		}
	}
	return result, refs, nil
}

func isTensorElementwise(op string) bool {
	switch op {
	case "add", "mul", "silu":
		return true
	default:
		return false
	}
}

func tensorExprUseCounts(root *egraph.Expr) map[*egraph.Expr]int {
	uses := make(map[*egraph.Expr]int)
	visited := make(map[*egraph.Expr]bool)
	var visit func(*egraph.Expr)
	visit = func(expr *egraph.Expr) {
		if expr == nil || visited[expr] {
			return
		}
		visited[expr] = true
		for _, child := range expr.Children {
			if child != nil {
				uses[child]++
				visit(child)
			}
		}
	}
	visit(root)
	return uses
}

func (b *tensorPlanBuilder) externalBuffer(expr *egraph.Expr, metadata tensorExprMetadata) (TensorBufferRef, error) {
	key := metadata.identity
	if key == "" && metadata.hasInputID {
		key = fmt.Sprintf("node-%d:%s", metadata.inputID, metadata.inputName)
	}
	if key == "" {
		return TensorBufferRef{}, fmt.Errorf("tensor lowering: %s leaf has no external input identity", expr.Op)
	}
	if b.inputIndex == nil {
		b.inputIndex = make(map[string]int)
	}
	if index, ok := b.inputIndex[key]; ok {
		layout := b.inputLayouts[index]
		if layout.DataType != metadata.dtype || !sameShape(layout.Shape, metadata.shape) {
			return TensorBufferRef{}, fmt.Errorf("tensor lowering: input %q has conflicting layouts", key)
		}
		return b.inputs[index], nil
	}
	id := -1
	if metadata.hasInputID {
		id = metadata.inputID
	}
	ref := TensorBufferRef{Kind: ExternalTensorBuffer, ID: id, Name: key}
	b.inputIndex[key] = len(b.inputs)
	b.inputs = append(b.inputs, ref)
	b.inputLayouts = append(b.inputLayouts, TensorBufferLayout{
		Buffer: ref, Shape: copyShape(metadata.shape), DataType: metadata.dtype,
	})
	return ref, nil
}

func tensorExprMetadataOf(expr *egraph.Expr, dimensions map[string]int64) (tensorExprMetadata, error) {
	return tensorExprMetadataOfActive(expr, dimensions, make(map[*egraph.Expr]bool))
}

func tensorExprMetadataOfActive(expr *egraph.Expr, dimensions map[string]int64, active map[*egraph.Expr]bool) (tensorExprMetadata, error) {
	if expr == nil {
		return tensorExprMetadata{}, fmt.Errorf("nil tensor expression")
	}
	if active[expr] {
		return tensorExprMetadata{}, fmt.Errorf("expression cycle at %q", expr.Op)
	}
	active[expr] = true
	defer delete(active, expr)
	var metadata tensorExprMetadata
	var shapeExprs []metalshape.Expr
	hasShapeMetadata := false
	data := expr.Data
	if nodeData, ok := data.(egraph.DimensionNodeData); ok {
		shape := nodeData.Shape()
		shapeExprs = shape.Dimensions()
		metadata.identity = shape.Identity()
		if id, name, ok := parseTensorLeafIdentity(metadata.identity); ok {
			metadata.inputID = id
			metadata.inputName = name
			metadata.hasInputID = true
		}
		data = nodeData.Payload()
	}
	switch value := data.(type) {
	case graphInputData:
		metadata.shape = copyShape(value.Shape)
		metadata.dtype = value.Type
		hasShapeMetadata = true
		metadata.inputID = value.ID
		metadata.inputName = value.Name
		metadata.hasInputID = true
	case TensorNodeData:
		metadata.shape = copyShape(value.Shape)
		metadata.dtype = value.Type
		metadata.attributes = value.Attributes
		// Bucket payloads omit concrete shapes because DimensionNodeData
		// carries the authoritative symbolic shape.
		hasShapeMetadata = shapeExprs == nil || len(value.Shape) != 0
	default:
		inferred, err := inferTensorExprMetadata(expr, dimensions, active)
		if err != nil {
			return tensorExprMetadata{}, fmt.Errorf("missing tensor payload (got %T): %w", expr.Data, err)
		}
		metadata = inferred
		hasShapeMetadata = true
	}
	if shapeExprs != nil {
		resolved := make([]int, len(shapeExprs))
		for i, dimension := range shapeExprs {
			value, err := dimension.Eval(dimensions)
			if err != nil {
				return tensorExprMetadata{}, fmt.Errorf("resolve dimension %d: %w", i, err)
			}
			if value > int64(math.MaxInt) || value > math.MaxUint32 {
				return tensorExprMetadata{}, fmt.Errorf("dimension %d does not fit the MSL uint index range", i)
			}
			resolved[i] = int(value)
		}
		if hasShapeMetadata && !sameShape(metadata.shape, resolved) {
			return tensorExprMetadata{}, fmt.Errorf("payload shape %v disagrees with symbolic shape %v", metadata.shape, resolved)
		}
		metadata.shape = resolved
	}
	if metadata.dtype.String() == "unknown" {
		return tensorExprMetadata{}, fmt.Errorf("unknown tensor type")
	}
	return metadata, nil
}

func inferTensorExprMetadata(expr *egraph.Expr, dimensions map[string]int64, active map[*egraph.Expr]bool) (tensorExprMetadata, error) {
	if expr.Op != "add" && expr.Op != "mul" && expr.Op != "silu" && expr.Op != "matmul" && expr.Op != "rmsnorm" && expr.Op != "rope" {
		return tensorExprMetadata{}, fmt.Errorf("operator %q has no inferable tensor payload", expr.Op)
	}
	children := make([]tensorExprMetadata, len(expr.Children))
	for i, child := range expr.Children {
		metadata, err := tensorExprMetadataOfActive(child, dimensions, active)
		if err != nil {
			return tensorExprMetadata{}, fmt.Errorf("input %d: %w", i, err)
		}
		if child.Sort != metadata.dtype.String() {
			return tensorExprMetadata{}, fmt.Errorf("input %d sort %q does not match payload type %s", i, child.Sort, metadata.dtype)
		}
		children[i] = metadata
	}
	want := 2
	if expr.Op == "silu" {
		want = 1
	}
	if len(children) != want {
		return tensorExprMetadata{}, fmt.Errorf("operator %q wants %d inputs, got %d", expr.Op, want, len(children))
	}
	metadata := tensorExprMetadata{dtype: children[0].dtype}
	for i := range children {
		if children[i].dtype != metadata.dtype && !(expr.Op == "rope" && i == 1) {
			return tensorExprMetadata{}, fmt.Errorf("input %d has incompatible type", i)
		}
	}
	switch expr.Op {
	case "add", "mul":
		metadata.shape = copyShape(children[0].shape)
	case "silu":
		metadata.shape = copyShape(children[0].shape)
	case "rmsnorm":
		metadata.shape = copyShape(children[0].shape)
	case "rope":
		metadata.shape = copyShape(children[0].shape)
	case "matmul":
		if len(children[0].shape) != 2 || len(children[1].shape) != 2 || children[0].shape[1] != children[1].shape[0] {
			return tensorExprMetadata{}, fmt.Errorf("matmul requires compatible rank-2 input shapes")
		}
		metadata.shape = []int{children[0].shape[0], children[1].shape[1]}
	}
	if expr.Sort != metadata.dtype.String() {
		return tensorExprMetadata{}, fmt.Errorf("sort %q does not match inferred type %s", expr.Sort, metadata.dtype)
	}
	if err := validateTensorStageShape(expr.Op, metadata.shape, children); err != nil {
		return tensorExprMetadata{}, err
	}
	return metadata, nil
}

func validateTensorStageShape(op string, output []int, inputs []tensorExprMetadata) error {
	same := func(a, b []int) bool { return sameShape(a, b) }
	switch op {
	case "add", "mul":
		if !same(output, inputs[0].shape) || !same(output, inputs[1].shape) {
			return fmt.Errorf("tensor lowering: %s requires equal contiguous shapes", op)
		}
	case "silu":
		if !same(output, inputs[0].shape) {
			return fmt.Errorf("tensor lowering: silu must preserve input shape")
		}
	case "rmsnorm":
		if len(output) == 0 || !same(output, inputs[0].shape) || !same(output, inputs[1].shape) {
			return fmt.Errorf("tensor lowering: rmsnorm requires equal non-scalar input shapes")
		}
	case "rope":
		if len(output) == 0 || !same(output, inputs[0].shape) || output[len(output)-1] < 2 || output[len(output)-1]%2 != 0 {
			return fmt.Errorf("tensor lowering: rope requires an even final data dimension")
		}
		positions, err := tensorElementCount(inputs[1].shape)
		if err != nil || positions != 1 {
			return fmt.Errorf("tensor lowering: rope position must contain one element")
		}
		if _, err := ropePositionType(inputs[1].dtype); err != nil {
			return fmt.Errorf("tensor lowering: rope %w", err)
		}
	case "matmul":
		if len(output) != 2 || len(inputs[0].shape) != 2 || len(inputs[1].shape) != 2 ||
			inputs[0].shape[1] != inputs[1].shape[0] ||
			output[0] != inputs[0].shape[0] || output[1] != inputs[1].shape[1] {
			return fmt.Errorf("tensor lowering: matmul requires compatible rank-2 row-major shapes")
		}
	}
	return nil
}

func tensorDispatch(shape []int, op string) (TensorDispatchSize, TensorDispatchSize) {
	count, _ := tensorElementCount(shape)
	if count == 0 {
		return TensorDispatchSize{Height: 1, Depth: 1}, TensorDispatchSize{Width: 1, Height: 1, Depth: 1}
	}
	if op == "matmul" {
		rows, cols := uint64(shape[0]), uint64(shape[1])
		return TensorDispatchSize{Width: rows * cols, Height: 1, Depth: 1}, TensorDispatchSize{
			Width: min(uint64(256), max(uint64(1), rows*cols)), Height: 1, Depth: 1,
		}
	}
	if op == "rmsnorm" {
		rows := uint64(1)
		for _, dimension := range shape[:len(shape)-1] {
			rows *= uint64(dimension)
		}
		return TensorDispatchSize{Width: rows, Height: 1, Depth: 1}, TensorDispatchSize{
			Width: min(uint64(256), max(uint64(1), rows)), Height: 1, Depth: 1,
		}
	}
	if op == "rope" {
		pairs := count / 2
		return TensorDispatchSize{Width: pairs, Height: 1, Depth: 1}, TensorDispatchSize{
			Width: min(uint64(256), max(uint64(1), pairs)), Height: 1, Depth: 1,
		}
	}
	return TensorDispatchSize{Width: count, Height: 1, Depth: 1}, TensorDispatchSize{
		Width: min(uint64(256), max(uint64(1), count)), Height: 1, Depth: 1,
	}
}

func generateRMSNormStageMSL(kernelName, precision string, shape []int, epsilon float32) string {
	width := shape[len(shape)-1]
	rows := 1
	for _, dimension := range shape[:len(shape)-1] {
		rows *= dimension
	}
	return fmt.Sprintf(`#include <metal_stdlib>
using namespace metal;

kernel void %s(
    device const %s* input [[buffer(0)]],
    device const %s* weight [[buffer(1)]],
    device %s* out [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= %d) return;
    const uint width = %d;
    const uint base = row * width;
    float sum = 0.0f;
    for (uint i = 0; i < width; ++i) {
        const float value = float(input[base + i]);
        sum += value * value;
    }
    const float scale = rsqrt(sum / float(width) + %.9g);
    for (uint i = 0; i < width; ++i) {
        out[base + i] = %s(float(input[base + i]) * scale * float(weight[base + i]));
    }
}
`, kernelName, precision, precision, precision, rows, width, epsilon, precision)
}

func tensorElementCount(shape []int) (uint64, error) {
	count := uint64(1)
	for _, dimension := range shape {
		if dimension < 0 {
			return 0, fmt.Errorf("negative dimension %d", dimension)
		}
		value := uint64(dimension)
		if value != 0 && count > math.MaxUint64/value {
			return 0, fmt.Errorf("element count overflows uint64")
		}
		count *= value
	}
	return count, nil
}

func generateMatMulStageMSL(kernelName, precision string, left, right, output []int) string {
	return fmt.Sprintf(`#include <metal_stdlib>
using namespace metal;

kernel void %s(
    device const %s* lhs [[buffer(0)]],
    device const %s* rhs [[buffer(1)]],
    device %s* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    const uint rows = %d;
    const uint cols = %d;
    if (id >= rows * cols) return;
    const uint row = id / max(cols, 1u);
    const uint col = id %% max(cols, 1u);
    float sum = 0.0f;
    for (uint k = 0; k < %d; ++k) {
        sum += float(lhs[row * %d + k]) * float(rhs[k * %d + col]);
    }
    out[id] = %s(sum);
}
`, kernelName, precision, precision, precision, output[0], output[1], left[1], left[1], right[1], precision)
}

func parseTensorLeafIdentity(identity string) (int, string, bool) {
	if !strings.HasPrefix(identity, "node-") {
		return 0, "", false
	}
	parts := strings.SplitN(strings.TrimPrefix(identity, "node-"), ":", 2)
	if len(parts) != 2 || parts[1] == "" {
		return 0, "", false
	}
	id, err := strconv.Atoi(parts[0])
	if err != nil || id < 0 {
		return 0, "", false
	}
	return id, parts[1], true
}

func validateTensorMSLShape(shape []int) error {
	count, err := tensorElementCount(shape)
	if err != nil {
		return err
	}
	if count > math.MaxUint32 {
		return fmt.Errorf("element count exceeds the MSL uint index range")
	}
	for _, dimension := range shape {
		if uint64(dimension) > math.MaxUint32 {
			return fmt.Errorf("dimension exceeds the MSL uint index range")
		}
	}
	return nil
}
