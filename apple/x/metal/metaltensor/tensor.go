package metaltensor

import (
	"fmt"
	"math"
	"runtime"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalir"
	"github.com/tmc/lib/apple/x/metal/metalshape"
)

// Tensor represents a deferred symbolic or device-backed tensor node in a computation DAG.
type Tensor struct {
	graph     *metalir.Graph
	node      *metalir.Node
	shape     []int
	shapeExpr []metalshape.Expr
	dtype     metalir.DataType
}

func captureSource() metalir.SourceLocation {
	var loc metalir.SourceLocation
	if pc, file, line, ok := runtime.Caller(2); ok {
		loc.File = file
		loc.Line = line
		if fn := runtime.FuncForPC(pc); fn != nil {
			loc.Function = fn.Name()
		}
	}
	return loc
}

// NewInput creates a new input tensor within the given graph.
func NewInput(g *metalir.Graph, name string, shape []int, dtype metalir.DataType) Tensor {
	shape = copyShape(shape)
	n := g.AddInput(name, copyShape(shape), dtype)
	n.SourceLoc = captureSource()
	return Tensor{
		graph:     g,
		node:      n,
		shape:     copyShape(n.Shape),
		shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...),
		dtype:     dtype,
	}
}

// NewInputWithShapeExpr creates an input whose dimensions may depend on runtime values.
func NewInputWithShapeExpr(g *metalir.Graph, name string, shape []metalshape.Expr, dtype metalir.DataType) (Tensor, error) {
	if g == nil {
		return Tensor{}, fmt.Errorf("input: nil graph")
	}
	n, err := g.AddInputWithShapeExpr(name, shape, dtype)
	if err != nil {
		return Tensor{}, fmt.Errorf("input %q: %w", name, err)
	}
	n.SourceLoc = captureSource()
	return Tensor{graph: g, node: n, shape: copyShape(n.Shape), shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...), dtype: dtype}, nil
}

// Graph returns the underlying computation graph.
func (t Tensor) Graph() *metalir.Graph {
	return t.graph
}

// Node returns the underlying IR node.
func (t Tensor) Node() *metalir.Node {
	return t.node
}

// Shape returns the tensor dimensions.
func (t Tensor) Shape() []int {
	return copyShape(t.shape)
}

// ShapeExpr returns the tensor dimensions as immutable expressions.
func (t Tensor) ShapeExpr() []metalshape.Expr {
	if t.shapeExpr != nil {
		return append([]metalshape.Expr(nil), t.shapeExpr...)
	}
	if t.node != nil {
		return append([]metalshape.Expr(nil), t.node.ShapeExpr...)
	}
	return nil
}

// DataType returns the tensor data type.
func (t Tensor) DataType() metalir.DataType {
	return t.dtype
}

// Add computes elementwise addition t + other.
func (t Tensor) Add(other Tensor) Tensor {
	n := t.graph.AddBinary("add", t.node, other.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph:     t.graph,
		node:      n,
		shape:     copyShape(n.Shape),
		shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...),
		dtype:     t.dtype,
	}
}

// Mul computes elementwise multiplication t * other.
func (t Tensor) Mul(other Tensor) Tensor {
	n := t.graph.AddBinary("mul", t.node, other.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph:     t.graph,
		node:      n,
		shape:     copyShape(n.Shape),
		shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...),
		dtype:     t.dtype,
	}
}

// MatMul computes matrix multiplication t x other.
func (t Tensor) MatMul(other Tensor) Tensor {
	n := t.graph.AddMatMul(t.node, other.node)
	n.SourceLoc = captureSource()
	outShape := copyShape(n.Shape)
	return Tensor{
		graph:     t.graph,
		node:      n,
		shape:     copyShape(outShape),
		shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...),
		dtype:     t.dtype,
	}
}

// RMSNorm computes Root Mean Square layer normalization.
func (t Tensor) RMSNorm(weight Tensor, eps float32) Tensor {
	n := t.graph.AddRMSNorm(t.node, weight.node, eps)
	n.SourceLoc = captureSource()
	return Tensor{
		graph:     t.graph,
		node:      n,
		shape:     copyShape(n.Shape),
		shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...),
		dtype:     t.dtype,
	}
}

// Silu computes SiLU (Swish-1) activation.
func (t Tensor) Silu() Tensor {
	n := t.graph.AddUnary("silu", t.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph:     t.graph,
		node:      n,
		shape:     copyShape(n.Shape),
		shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...),
		dtype:     t.dtype,
	}
}

// RoPE computes Rotary Position Embedding.
func (t Tensor) RoPE(pos Tensor, theta float32) Tensor {
	n := t.graph.AddRoPE(t.node, pos.node, theta)
	n.SourceLoc = captureSource()
	return Tensor{
		graph:     t.graph,
		node:      n,
		shape:     copyShape(n.Shape),
		shapeExpr: append([]metalshape.Expr(nil), n.ShapeExpr...),
		dtype:     t.dtype,
	}
}

type graphInputData struct {
	ID    int
	Name  string
	Shape []int
	Type  metalir.DataType
}

// TensorNodeData carries output facts and scalar parameters in an egraph node.
// Its exported fields remain available on extracted Expr.Data values.
type TensorNodeData struct {
	Shape      []int
	Type       metalir.DataType
	Attributes metalir.ScalarAttributes
}

type tensorFact struct {
	shape    []int
	dtype    metalir.DataType
	attrs    metalir.ScalarAttributes
	known    bool
	conflict bool
}

type tensorAnalysis struct{}

func (tensorAnalysis) Make(g *egraph.EGraph, n egraph.ENode) any {
	var fact tensorFact
	data := n.Data
	if shaped, ok := data.(egraph.DimensionNodeData); ok {
		data = shaped.Payload()
	}
	switch data := data.(type) {
	case graphInputData:
		fact = tensorFact{shape: copyShape(data.Shape), dtype: data.Type, known: true}
	case TensorNodeData:
		fact = tensorFact{shape: copyShape(data.Shape), dtype: data.Type, attrs: data.Attributes, known: true}
	default:
		return inferTensorFact(g, n)
	}
	// Custom operators carry an explicit output fact in their payload. Their
	// shape semantics belong to their own analysis or validation layer.
	switch n.Op {
	case "add", "mul", "silu", "rmsnorm", "rope", "matmul":
	default:
		return fact
	}
	childFacts := make([]tensorFact, len(n.Children))
	for i, child := range n.Children {
		value, ok := g.AnalysisData(child)
		childFact, isTensorFact := value.(tensorFact)
		if !ok || !isTensorFact || !childFact.known || childFact.conflict {
			return tensorFact{conflict: true}
		}
		childFacts[i] = childFact
	}
	switch n.Op {
	case "add", "mul", "silu", "rmsnorm":
		want := 2
		if n.Op == "silu" {
			want = 1
		}
		if len(childFacts) != want {
			return tensorFact{conflict: true}
		}
		for _, child := range childFacts {
			if child.dtype != fact.dtype || !sameShape(child.shape, fact.shape) {
				return tensorFact{conflict: true}
			}
		}
	case "rope":
		if len(childFacts) != 2 || childFacts[0].dtype != fact.dtype || !sameShape(childFacts[0].shape, fact.shape) {
			return tensorFact{conflict: true}
		}
	case "matmul":
		if len(childFacts) != 2 || childFacts[0].dtype != fact.dtype || childFacts[1].dtype != fact.dtype ||
			len(childFacts[0].shape) != 2 || len(childFacts[1].shape) != 2 || len(fact.shape) != 2 ||
			childFacts[0].shape[1] != childFacts[1].shape[0] ||
			fact.shape[0] != childFacts[0].shape[0] || fact.shape[1] != childFacts[1].shape[1] {
			return tensorFact{conflict: true}
		}
	}
	return fact
}

func inferTensorFact(g *egraph.EGraph, n egraph.ENode) any {
	if n.Op != "add" && n.Op != "mul" && n.Op != "silu" && n.Op != "matmul" {
		return nil
	}
	children := make([]tensorFact, len(n.Children))
	for i, child := range n.Children {
		value, ok := g.AnalysisData(child)
		fact, isTensorFact := value.(tensorFact)
		if !ok || !isTensorFact || !fact.known || fact.conflict {
			return tensorFact{conflict: true}
		}
		children[i] = fact
	}
	fact := tensorFact{known: true}
	switch n.Op {
	case "add", "mul":
		if len(children) != 2 || children[0].dtype != children[1].dtype || !sameShape(children[0].shape, children[1].shape) {
			return tensorFact{conflict: true}
		}
		fact.dtype = children[0].dtype
		fact.shape = copyShape(children[0].shape)
	case "silu":
		if len(children) != 1 {
			return tensorFact{conflict: true}
		}
		fact.dtype = children[0].dtype
		fact.shape = copyShape(children[0].shape)
	case "matmul":
		if len(children) != 2 || children[0].dtype != children[1].dtype ||
			len(children[0].shape) != 2 || len(children[1].shape) != 2 ||
			children[0].shape[1] != children[1].shape[0] {
			return tensorFact{conflict: true}
		}
		fact.dtype = children[0].dtype
		fact.shape = []int{children[0].shape[0], children[1].shape[1]}
	}
	if fact.dtype.String() != n.Sort {
		return tensorFact{conflict: true}
	}
	return fact
}

func (tensorAnalysis) Merge(a, b any) (any, bool) {
	left, leftOK := a.(tensorFact)
	right, rightOK := b.(tensorFact)
	if !leftOK {
		if rightOK {
			return cloneTensorFact(right), true
		}
		return nil, false
	}
	if !rightOK {
		return cloneTensorFact(left), false
	}
	if left.conflict || right.conflict {
		conflict := tensorFact{conflict: true}
		return conflict, !sameTensorFact(left, conflict)
	}
	if !left.known {
		if !right.known {
			return tensorFact{}, false
		}
		return cloneTensorFact(right), true
	}
	if !right.known {
		return cloneTensorFact(left), false
	}
	if !sameTensorFact(left, right) {
		return tensorFact{conflict: true}, true
	}
	return cloneTensorFact(left), false
}

func cloneTensorFact(fact tensorFact) tensorFact {
	fact.shape = copyShape(fact.shape)
	return fact
}

func sameTensorFact(a, b tensorFact) bool {
	return a.known == b.known && a.conflict == b.conflict && a.dtype == b.dtype &&
		a.attrs == b.attrs && sameShape(a.shape, b.shape)
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func copyShape(shape []int) []int {
	return append([]int(nil), shape...)
}

func positiveFinite(value float64) bool {
	return value > 0 && !math.IsInf(value, 0) && !math.IsNaN(value)
}

// NewTensorOperatorRegistry returns the built-in tensor operators, signatures,
// and operand-commutation rewrites for add and mul. Callers may add rewrites
// with OperatorRegistry.AddRewrites or register additional operators.
func NewTensorOperatorRegistry() (*egraph.OperatorRegistry, error) {
	registry := egraph.NewOperatorRegistry()
	signatures := make(map[string][]egraph.OperatorSignature)
	rewrites := make(map[string][]egraph.OperatorRewrite)
	mslGenerators := map[string]egraph.MSLGenerator{
		"matmul":  tensorMatMulMSL,
		"rmsnorm": tensorRMSNormMSL,
		"rope":    tensorRoPEMSL,
	}
	types := []metalir.DataType{
		metalir.Float32, metalir.Float16, metalir.BFloat16,
		metalir.Int32, metalir.Int64, metalir.Uint8,
	}
	for _, dtype := range types {
		sort := dtype.String()
		for _, op := range []string{"input", "constant"} {
			signatures[op] = append(signatures[op], egraph.OperatorSignature{Result: sort})
		}
		for _, op := range []string{"add", "mul", "matmul", "rmsnorm"} {
			signatures[op] = append(signatures[op], egraph.OperatorSignature{Result: sort, Children: []string{sort, sort}})
		}
		signatures["silu"] = append(signatures["silu"], egraph.OperatorSignature{Result: sort, Children: []string{sort}})
		signatures["rope"] = append(signatures["rope"], egraph.OperatorSignature{Result: sort, Children: []string{sort, ""}})
		for _, op := range []string{"add", "mul"} {
			name, resultSort := op, sort
			rewrites[op] = append(rewrites[op], egraph.OperatorRewrite{
				Phase: "tensor algebra",
				Factory: func() (*egraph.Rewrite, error) {
					variableA := egraph.VarSorted(resultSort, "a")
					variableB := egraph.VarSorted(resultSort, "b")
					return egraph.NewPatternRewrite(
						"tensor-"+name+"-commute-"+resultSort,
						egraph.PatSorted(resultSort, name, variableA, variableB),
						egraph.PatSorted(resultSort, name, variableB, variableA),
					)
				},
			})
		}
	}
	for _, op := range []string{"input", "constant", "add", "mul", "matmul", "rmsnorm", "silu", "rope"} {
		if err := registry.Register(op, egraph.OperatorDefinition{
			Signatures: signatures[op], Rewrites: rewrites[op], MSL: mslGenerators[op],
		}); err != nil {
			return nil, err
		}
	}
	return registry, nil
}

func validateTensorNode(n *metalir.Node) error {
	if n == nil {
		return fmt.Errorf("compile: nil node")
	}
	if n.ValidationError != nil {
		return fmt.Errorf("compile: invalid node %q: %w", n.Name, n.ValidationError)
	}
	switch n.Op {
	case "rmsnorm":
		if !n.Attributes.HasEpsilon || n.Attributes.HasTheta || !positiveFinite(float64(n.Attributes.Epsilon)) {
			return fmt.Errorf("compile: node %q has invalid RMSNorm epsilon", n.Name)
		}
	case "rope":
		if !n.Attributes.HasTheta || n.Attributes.HasEpsilon || !positiveFinite(float64(n.Attributes.Theta)) {
			return fmt.Errorf("compile: node %q has invalid RoPE theta", n.Name)
		}
	default:
		if n.Attributes != (metalir.ScalarAttributes{}) {
			return fmt.Errorf("compile: node %q has scalar attributes unsupported by %s", n.Name, n.Op)
		}
	}
	if _, err := n.ResolveShape(nil); err != nil {
		return fmt.Errorf("compile: node %q has unresolved or inconsistent shape: %w", n.Name, err)
	}
	wantInputs := 0
	switch n.Op {
	case "add", "mul", "rmsnorm", "rope", "matmul":
		wantInputs = 2
	case "silu":
		wantInputs = 1
	default:
		return nil
	}
	if len(n.Inputs) != wantInputs {
		return fmt.Errorf("compile: %s wants %d inputs, got %d", n.Op, wantInputs, len(n.Inputs))
	}
	for i, input := range n.Inputs {
		if input == nil {
			return fmt.Errorf("compile: %s input %d is nil", n.Op, i)
		}
	}
	if n.Op == "matmul" {
		left, right := n.Inputs[0], n.Inputs[1]
		if left.Type != n.Type || right.Type != n.Type {
			return fmt.Errorf("compile: matmul inputs and output must have the same type")
		}
		if len(left.Shape) != 2 || len(right.Shape) != 2 || len(n.Shape) != 2 {
			return fmt.Errorf("compile: matmul requires rank-2 inputs and output")
		}
		if left.Shape[1] != right.Shape[0] {
			return fmt.Errorf("compile: matmul inner dimensions differ: %d and %d", left.Shape[1], right.Shape[0])
		}
		if n.Shape[0] != left.Shape[0] || n.Shape[1] != right.Shape[1] {
			return fmt.Errorf("compile: matmul output shape does not match its inputs")
		}
		return nil
	}
	if n.Op == "rope" {
		input := n.Inputs[0]
		if input.Type != n.Type || !sameShape(input.Shape, n.Shape) {
			return fmt.Errorf("compile: rope output must preserve the data input type and shape")
		}
		return nil
	}
	for i, input := range n.Inputs {
		if input.Type != n.Type {
			return fmt.Errorf("compile: %s input %d has type %s, output has type %s", n.Op, i, input.Type, n.Type)
		}
		if len(input.Shape) != len(n.Shape) {
			return fmt.Errorf("compile: %s input %d has rank %d, output has rank %d", n.Op, i, len(input.Shape), len(n.Shape))
		}
		for dim := range n.Shape {
			if input.Shape[dim] != n.Shape[dim] {
				return fmt.Errorf("compile: %s input %d dimension %d is %d, output dimension is %d", n.Op, i, dim, input.Shape[dim], n.Shape[dim])
			}
		}
	}
	return nil
}

// Compile compiles a fully concrete expression DAG into an MSL shader.
// Symbolic dimensions require CompileWithDimensions.
func (t Tensor) Compile(kernelName string) (string, error) {
	registry, err := NewTensorOperatorRegistry()
	if err != nil {
		return "", fmt.Errorf("compile: register tensor operators: %w", err)
	}
	return t.CompileWithRegistry(kernelName, nil, registry, egraph.SaturationOptions{})
}

// CompileWithDimensions binds symbolic dimensions, validates every graph node,
// and compiles a concrete snapshot. It does not specialize bucket intervals.
func (t Tensor) CompileWithDimensions(kernelName string, dimensions map[string]int64) (string, error) {
	registry, err := NewTensorOperatorRegistry()
	if err != nil {
		return "", fmt.Errorf("compile: register tensor operators: %w", err)
	}
	return t.CompileWithRegistry(kernelName, dimensions, registry, egraph.SaturationOptions{})
}

// CompileWithSaturation binds symbolic dimensions, lowers a concrete graph,
// runs the supplied rewrite phases to a fixed point, and emits MSL. A limit
// stop is an error; partial saturation is never extracted.
func (t Tensor) CompileWithSaturation(kernelName string, dimensions map[string]int64, phases []egraph.SaturationPhase, opts egraph.SaturationOptions) (string, error) {
	registry, err := NewTensorOperatorRegistry()
	if err != nil {
		return "", fmt.Errorf("compile: register tensor operators: %w", err)
	}
	return t.compileWithSaturationRegistry(kernelName, dimensions, registry, phases, opts)
}

// CompileWithRegistry compiles with the supplied operator registry. Its
// registered rewrite factories determine the ordered phase fixed points.
func (t Tensor) CompileWithRegistry(kernelName string, dimensions map[string]int64, registry *egraph.OperatorRegistry, saturationOpts egraph.SaturationOptions) (string, error) {
	if registry == nil {
		return "", fmt.Errorf("compile: nil operator registry")
	}
	schedule, err := registry.NewSaturationRewriteSchedule()
	if err != nil {
		return "", fmt.Errorf("compile: create operator rewrite schedule: %w", err)
	}
	return t.compileWithRewriteSchedule(kernelName, dimensions, registry, schedule, saturationOpts)
}

// CompileWithOperatorSchedule binds dimensions, lowers a concrete graph,
// executes a registry-backed nested rewrite schedule, and emits MSL. Schedule
// runs can be placed inside fixed points, sequences, and bounded repeats.
func (t Tensor) CompileWithOperatorSchedule(kernelName string, dimensions map[string]int64, registry *egraph.OperatorRegistry, schedule *egraph.OperatorSchedule, opts egraph.SaturationOptions) (string, error) {
	if registry == nil {
		return "", fmt.Errorf("compile: nil operator registry")
	}
	resolvedSchedule, err := registry.NewRewriteSchedule(schedule)
	if err != nil {
		return "", fmt.Errorf("compile: create operator rewrite schedule: %w", err)
	}
	return t.compileWithRewriteSchedule(kernelName, dimensions, registry, resolvedSchedule, opts)
}

func (t Tensor) compileWithRewriteSchedule(kernelName string, dimensions map[string]int64, registry *egraph.OperatorRegistry, schedule *egraph.RewriteSchedule, opts egraph.SaturationOptions) (string, error) {
	concrete, err := t.resolveDimensions(dimensions)
	if err != nil {
		return "", err
	}
	eg, root, err := concrete.lowerConcrete(registry)
	if err != nil {
		return "", err
	}
	root, err = saturateTensorGraphWithSchedule(eg, root, schedule, opts)
	if err != nil {
		return "", err
	}
	if err := validateTensorRoot(eg, root, concrete.node); err != nil {
		return "", err
	}
	expr, _, err := egraph.ExtractBest(eg, root, egraph.AstSizeCost{})
	if err != nil {
		return "", fmt.Errorf("compile: %w", err)
	}
	precision, err := tensorPrecision(concrete.node.Type)
	if err != nil {
		return "", err
	}
	return egraph.GenerateMSL(expr, egraph.CodegenOptions{
		KernelName: kernelName,
		Precision:  precision,
		Registry:   registry,
	})
}

// TensorCandidateSpace is a saturated, typed candidate universe for one
// concrete tensor shape. Its snapshot is graph-revision-bound. The registry
// must not be modified while the space is in use.
type TensorCandidateSpace struct {
	snapshot *egraph.CandidateSnapshot
	registry *egraph.OperatorRegistry
	dtype    metalir.DataType
}

// PrepareCandidateSpace resolves dimensions, lowers and saturates the graph,
// validates root analysis, and captures all reachable alternatives without
// selecting a cost-based winner. A nil registry selects built-in operators.
func (t Tensor) PrepareCandidateSpace(dimensions map[string]int64, registry *egraph.OperatorRegistry, opts egraph.SaturationOptions) (*TensorCandidateSpace, error) {
	if registry == nil {
		var err error
		registry, err = NewTensorOperatorRegistry()
		if err != nil {
			return nil, fmt.Errorf("candidate space: create tensor operator registry: %w", err)
		}
	}
	schedule, err := registry.NewSaturationRewriteSchedule()
	if err != nil {
		return nil, fmt.Errorf("candidate space: create operator rewrite schedule: %w", err)
	}
	concrete, err := t.resolveDimensions(dimensions)
	if err != nil {
		return nil, err
	}
	eg, root, err := concrete.lowerConcrete(registry)
	if err != nil {
		return nil, fmt.Errorf("candidate space: lower tensor graph: %w", err)
	}
	root, err = saturateTensorGraphWithSchedule(eg, root, schedule, opts)
	if err != nil {
		return nil, fmt.Errorf("candidate space: %w", err)
	}
	if err := validateTensorRoot(eg, root, concrete.node); err != nil {
		return nil, err
	}
	snapshot, err := egraph.NewCandidateSnapshot(eg, root)
	if err != nil {
		return nil, fmt.Errorf("candidate space: snapshot tensor alternatives: %w", err)
	}
	return &TensorCandidateSpace{snapshot: snapshot, registry: registry, dtype: concrete.node.Type}, nil
}

// Snapshot returns the revision-bound e-graph candidate snapshot.
func (s *TensorCandidateSpace) Snapshot() *egraph.CandidateSnapshot {
	if s == nil {
		return nil
	}
	return s.snapshot
}

// NewSearch creates a caller-measured search over this candidate space.
func (s *TensorCandidateSpace) NewSearch(config egraph.CandidateSearchConfig) (*egraph.CandidateSearch, error) {
	if s == nil || s.snapshot == nil {
		return nil, fmt.Errorf("candidate space: nil snapshot")
	}
	return egraph.NewCandidateSearch(s.snapshot, config)
}

// GenerateMSL emits one externally selected candidate using this space's
// operator registry and root tensor precision.
func (s *TensorCandidateSpace) GenerateMSL(kernelName string, expr *egraph.Expr) (string, error) {
	if s == nil || s.snapshot == nil || s.registry == nil {
		return "", fmt.Errorf("candidate space: incomplete candidate space")
	}
	if expr == nil || expr.Sort != s.dtype.String() {
		return "", fmt.Errorf("candidate space: candidate root sort does not match tensor type %s", s.dtype)
	}
	precision, err := tensorPrecision(s.dtype)
	if err != nil {
		return "", err
	}
	return egraph.GenerateMSL(expr, egraph.CodegenOptions{KernelName: kernelName, Precision: precision, Registry: s.registry})
}

func (t Tensor) compileWithSaturationRegistry(kernelName string, dimensions map[string]int64, registry *egraph.OperatorRegistry, phases []egraph.SaturationPhase, opts egraph.SaturationOptions) (string, error) {
	concrete, err := t.resolveDimensions(dimensions)
	if err != nil {
		return "", err
	}
	return concrete.compileConcrete(kernelName, registry, phases, opts)
}

func (t Tensor) resolveDimensions(dimensions map[string]int64) (Tensor, error) {
	if t.graph == nil || t.node == nil {
		return Tensor{}, fmt.Errorf("compile: tensor has no graph node")
	}
	clones := make(map[*metalir.Node]*metalir.Node, len(t.graph.Nodes))
	for _, node := range t.graph.Nodes {
		if node == nil {
			return Tensor{}, fmt.Errorf("compile: graph contains a nil node")
		}
		shape, err := node.ResolveShape(dimensions)
		if err != nil {
			return Tensor{}, fmt.Errorf("compile: resolve node %q shape: %w", node.Name, err)
		}
		clone := *node
		clone.Shape = shape
		clone.ShapeExpr = nil
		clone.Inputs = nil
		clones[node] = &clone
	}
	for _, node := range t.graph.Nodes {
		clone := clones[node]
		clone.Inputs = make([]*metalir.Node, len(node.Inputs))
		for i, input := range node.Inputs {
			mapped, ok := clones[input]
			if !ok {
				return Tensor{}, fmt.Errorf("compile: node %q input %d is outside its graph", node.Name, i)
			}
			clone.Inputs[i] = mapped
		}
	}
	root, ok := clones[t.node]
	if !ok {
		return Tensor{}, fmt.Errorf("compile: tensor node is outside its graph")
	}
	return Tensor{graph: t.graph, node: root, shape: copyShape(root.Shape), dtype: t.dtype}, nil
}

func (t Tensor) compileConcrete(kernelName string, registry *egraph.OperatorRegistry, phases []egraph.SaturationPhase, opts egraph.SaturationOptions) (string, error) {
	eg, root, err := t.lowerConcrete(registry)
	if err != nil {
		return "", err
	}
	root, err = saturateTensorGraph(eg, root, phases, opts)
	if err != nil {
		return "", err
	}
	if err := validateTensorRoot(eg, root, t.node); err != nil {
		return "", err
	}
	expr, _, err := egraph.ExtractBest(eg, root, egraph.AstSizeCost{})
	if err != nil {
		return "", fmt.Errorf("compile: %w", err)
	}
	precision, err := tensorPrecision(t.node.Type)
	if err != nil {
		return "", err
	}
	return egraph.GenerateMSL(expr, egraph.CodegenOptions{
		KernelName: kernelName,
		Precision:  precision,
		Registry:   registry,
	})
}

func (t Tensor) lowerConcrete(registry *egraph.OperatorRegistry) (*egraph.EGraph, egraph.Id, error) {
	eg, err := registry.NewEGraphWithAnalysis(tensorAnalysis{})
	if err != nil {
		return nil, egraph.InvalidId, fmt.Errorf("compile: create typed tensor e-graph: %w", err)
	}
	classes := make(map[*metalir.Node]egraph.Id)
	visiting := make(map[*metalir.Node]bool)
	type lowerFrame struct {
		node     *metalir.Node
		children []egraph.Id
		next     int
	}
	ready := func(n *metalir.Node) (egraph.Id, bool, error) {
		if n == nil {
			return egraph.InvalidId, false, fmt.Errorf("compile: graph contains a nil input")
		}
		if id, ok := classes[n]; ok {
			return id, true, nil
		}
		if visiting[n] {
			return egraph.InvalidId, false, fmt.Errorf("compile: graph contains a cycle at node %q", n.Name)
		}
		if err := validateTensorNode(n); err != nil {
			return egraph.InvalidId, false, err
		}
		if n.Kind == metalir.KindInput || n.Kind == metalir.KindConstant {
			id, err := eg.AddChecked(egraph.ENode{
				Op: n.Op, Sort: n.Type.String(),
				Data: graphInputData{
					ID: n.ID, Name: n.Name, Shape: copyShape(n.Shape), Type: n.Type,
				},
			})
			if err != nil {
				return egraph.InvalidId, false, fmt.Errorf("compile: add input %q: %w", n.Name, err)
			}
			classes[n] = id
			return id, true, nil
		}
		return egraph.InvalidId, false, nil
	}

	root, done, err := ready(t.node)
	if err != nil {
		return nil, egraph.InvalidId, err
	}
	if !done {
		visiting[t.node] = true
		stack := []lowerFrame{{node: t.node, children: make([]egraph.Id, len(t.node.Inputs))}}
		for len(stack) > 0 {
			current := &stack[len(stack)-1]
			if current.next < len(current.node.Inputs) {
				input := current.node.Inputs[current.next]
				id, done, err := ready(input)
				if err != nil {
					return nil, egraph.InvalidId, err
				}
				if done {
					current.children[current.next] = id
					current.next++
					continue
				}
				visiting[input] = true
				stack = append(stack, lowerFrame{node: input, children: make([]egraph.Id, len(input.Inputs))})
				continue
			}

			id, err := eg.AddChecked(egraph.ENode{
				Op:       current.node.Op,
				Sort:     current.node.Type.String(),
				Data:     TensorNodeData{Shape: copyShape(current.node.Shape), Type: current.node.Type, Attributes: current.node.Attributes},
				Children: current.children,
			})
			if err != nil {
				return nil, egraph.InvalidId, fmt.Errorf("compile: add node %q: %w", current.node.Name, err)
			}
			classes[current.node] = id
			delete(visiting, current.node)
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				root = id
				break
			}
			parent := &stack[len(stack)-1]
			parent.children[parent.next] = id
			parent.next++
		}
	}
	return eg, root, nil
}

func saturateTensorGraph(eg *egraph.EGraph, root egraph.Id, phases []egraph.SaturationPhase, opts egraph.SaturationOptions) (egraph.Id, error) {
	saturation, err := egraph.SaturateSchedule(eg, phases, opts)
	if err != nil {
		return egraph.InvalidId, fmt.Errorf("compile: saturate tensor e-graph: %w", err)
	}
	if saturation.StopReason != egraph.SaturationFixedPoint && saturation.StopReason != egraph.SaturationScheduleDone {
		return egraph.InvalidId, fmt.Errorf("compile: saturation stopped with %s", saturation.StopReason)
	}
	root, err = eg.FindChecked(root)
	if err != nil {
		return egraph.InvalidId, fmt.Errorf("compile: canonicalize tensor root: %w", err)
	}
	return root, nil
}

func saturateTensorGraphWithSchedule(eg *egraph.EGraph, root egraph.Id, schedule *egraph.RewriteSchedule, opts egraph.SaturationOptions) (egraph.Id, error) {
	result, err := egraph.ExecuteSchedule(eg, schedule, opts)
	if err != nil {
		return egraph.InvalidId, fmt.Errorf("compile: execute tensor rewrite schedule: %w", err)
	}
	if result.StopReason != egraph.SaturationScheduleDone && result.StopReason != egraph.SaturationFixedPoint {
		return egraph.InvalidId, fmt.Errorf("compile: rewrite schedule stopped with %s", result.StopReason)
	}
	root, err = eg.FindChecked(root)
	if err != nil {
		return egraph.InvalidId, fmt.Errorf("compile: canonicalize tensor root: %w", err)
	}
	return root, nil
}

func validateTensorRoot(eg *egraph.EGraph, root egraph.Id, node *metalir.Node) error {
	value, ok := eg.AnalysisData(root)
	fact, isTensorFact := value.(tensorFact)
	if !ok || !isTensorFact || !fact.known || fact.conflict ||
		fact.dtype != node.Type || !sameShape(fact.shape, node.Shape) {
		return fmt.Errorf("compile: inconsistent tensor shape or type analysis at root")
	}
	return nil
}

func tensorPrecision(dtype metalir.DataType) (string, error) {
	switch dtype {
	case metalir.Float32:
		return "float", nil
	case metalir.Float16:
		return "half", nil
	default:
		return "", fmt.Errorf("compile: unsupported MSL data type %s", dtype)
	}
}
