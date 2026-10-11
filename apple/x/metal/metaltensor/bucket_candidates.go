package metaltensor

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalir"
	"github.com/tmc/lib/apple/x/metal/metalshape"
)

// TensorBucketCandidateSpace is one independently saturated candidate space
// for a single symbolic-dimension bucket combination. Scores measured at a
// representative point only rank candidates. Interval analysis rejects
// disjoint shape bounds, but overlapping bounds do not prove shape equality.
// Rewrite guards must prove their conditions for the whole bucket, and custom
// operator shape semantics remain the caller's responsibility. Callers must
// bind runtime dimensions and hard-validate the whole bucket before promotion.
// The registry must not be modified while a space is in use.
type TensorBucketCandidateSpace struct {
	context  egraph.BucketContext
	snapshot *egraph.CandidateSnapshot
	registry *egraph.OperatorRegistry
	dtype    metalir.DataType
}

// BucketContext returns this candidate space's bucket indices and intersected
// intervals as defensive copies.
func (s *TensorBucketCandidateSpace) BucketContext() egraph.BucketContext {
	if s == nil {
		return egraph.BucketContext{}
	}
	return egraph.BucketContext{
		Index:         s.context.Index,
		BucketIndices: cloneBucketIndices(s.context.BucketIndices),
		Intervals:     cloneBucketIntervals(s.context.Intervals),
	}
}

// Snapshot returns the revision-bound candidate snapshot for this bucket.
func (s *TensorBucketCandidateSpace) Snapshot() *egraph.CandidateSnapshot {
	if s == nil {
		return nil
	}
	return s.snapshot
}

// NewSearch creates a caller-measured search over this bucket's alternatives.
func (s *TensorBucketCandidateSpace) NewSearch(config egraph.CandidateSearchConfig) (*egraph.CandidateSearch, error) {
	if s == nil || s.snapshot == nil {
		return nil, fmt.Errorf("tensor bucket: nil candidate snapshot")
	}
	return egraph.NewCandidateSearch(s.snapshot, config)
}

// GenerateMSL emits an externally selected expression with this bucket's
// operator registry and root tensor precision.
func (s *TensorBucketCandidateSpace) GenerateMSL(kernelName string, expr *egraph.Expr) (string, error) {
	if s == nil || s.snapshot == nil || s.registry == nil {
		return "", fmt.Errorf("tensor bucket: incomplete candidate space")
	}
	if expr == nil || expr.Sort != s.dtype.String() {
		return "", fmt.Errorf("tensor bucket: candidate root sort does not match tensor type %s", s.dtype)
	}
	precision, err := tensorPrecision(s.dtype)
	if err != nil {
		return "", err
	}
	return egraph.GenerateMSL(expr, egraph.CodegenOptions{
		KernelName: kernelName,
		Precision:  precision,
		Registry:   s.registry,
	})
}

// PrepareBucketCandidateSpaces specializes a symbolic tensor DAG under each
// bucket interval, then strictly type-checks, saturates, and snapshots one
// candidate graph per combination. Base intervals are intersected by
// BuildDimensionSearchSpace. Incomplete saturation is always rejected. Its
// interval analysis detects disjoint shapes, not equality: overlapping bounds
// alone cannot justify a rewrite that requires equal dimensions. Rewrite
// guards and custom operator shape contracts must be valid throughout each
// bucket. This method prepares candidates; it does not prove them valid for
// every runtime shape in the bucket.
func (t Tensor) PrepareBucketCandidateSpaces(
	buckets egraph.DimensionBuckets,
	base map[string]egraph.DimensionInterval,
	opts egraph.SearchSpaceOptions,
	registry *egraph.OperatorRegistry,
) ([]*TensorBucketCandidateSpace, error) {
	if registry == nil {
		var err error
		registry, err = NewTensorOperatorRegistry()
		if err != nil {
			return nil, fmt.Errorf("tensor bucket: create operator registry: %w", err)
		}
	}
	if t.graph == nil || t.node == nil {
		return nil, fmt.Errorf("tensor bucket: tensor has no graph node")
	}
	nodes, shapes, err := symbolicGraphInputs(t)
	if err != nil {
		return nil, err
	}
	schedule, err := registry.NewSaturationRewriteSchedule()
	if err != nil {
		return nil, fmt.Errorf("tensor bucket: create operator rewrite schedule: %w", err)
	}
	opts.AllowIncomplete = false
	space, err := egraph.BuildDimensionSearchSpaceWithSchedule(shapes, buckets, base, opts, func(
		context egraph.BucketContext,
		graph *egraph.EGraph,
		specialized map[string]egraph.DimensionShape,
	) ([]egraph.Id, error) {
		if err := registry.RegisterOperators(graph); err != nil {
			return nil, fmt.Errorf("register strict operator signatures: %w", err)
		}
		ids := make(map[*metalir.Node]egraph.Id, len(nodes))
		for _, node := range nodes {
			children := make([]egraph.Id, len(node.Inputs))
			for i, child := range node.Inputs {
				id, ok := ids[child]
				if !ok {
					return nil, fmt.Errorf("node %q input %d was not lowered", node.Name, i)
				}
				children[i] = id
			}
			shape := specialized[tensorNodeKey(node)]
			shapePayload := egraph.NewDimensionShape(shape.Dimensions())
			if node.Kind == metalir.KindInput || node.Kind == metalir.KindConstant {
				shapePayload = egraph.NewDimensionShape(shape.Dimensions(), tensorLeafIdentity(node))
			}
			payload := egraph.NewDimensionNodeData(shapePayload, TensorNodeData{Type: node.Type, Attributes: node.Attributes})
			id, err := graph.AddChecked(egraph.ENode{
				Op: node.Op, Sort: node.Type.String(), Data: payload, Children: children,
			})
			if err != nil {
				return nil, fmt.Errorf("lower node %q: %w", node.Name, err)
			}
			ids[node] = id
		}
		root, ok := ids[t.node]
		if !ok {
			return nil, fmt.Errorf("tensor root was not lowered")
		}
		return []egraph.Id{root}, nil
	}, schedule)
	if err != nil {
		return nil, fmt.Errorf("tensor bucket: build dimension search space: %w", err)
	}
	built := space.Buckets()
	out := make([]*TensorBucketCandidateSpace, len(built))
	for i, bucket := range built {
		if len(bucket.Roots) != 1 {
			return nil, fmt.Errorf("tensor bucket %d: expected one root, got %d", i, len(bucket.Roots))
		}
		root := bucket.Roots[0]
		value, ok, err := bucket.Graph.AnalysisDataChecked(root)
		if err != nil {
			return nil, fmt.Errorf("tensor bucket %d: read root dimension fact: %w", i, err)
		}
		fact, factOK := value.(egraph.DimensionFact)
		if !ok || !factOK || !fact.Known() || fact.ConflictOrInvalid() {
			return nil, fmt.Errorf("tensor bucket %d: root has no valid dimension fact", i)
		}
		snapshot, err := egraph.NewCandidateSnapshot(bucket.Graph, root)
		if err != nil {
			return nil, fmt.Errorf("tensor bucket %d: capture candidates: %w", i, err)
		}
		out[i] = &TensorBucketCandidateSpace{
			context: egraph.BucketContext{
				Index: bucket.Index, BucketIndices: cloneBucketIndices(bucket.BucketIndices),
				Intervals: cloneBucketIntervals(bucket.Intervals),
			},
			snapshot: snapshot, registry: registry, dtype: t.node.Type,
		}
	}
	return out, nil
}

func symbolicGraphInputs(t Tensor) ([]*metalir.Node, map[string]egraph.DimensionShape, error) {
	if t.graph == nil || t.node == nil {
		return nil, nil, fmt.Errorf("tensor bucket: tensor has no graph node")
	}
	known := make(map[*metalir.Node]bool, len(t.graph.Nodes))
	ids := make(map[int]bool, len(t.graph.Nodes))
	for _, node := range t.graph.Nodes {
		if node == nil || known[node] || ids[node.ID] {
			return nil, nil, fmt.Errorf("tensor bucket: graph contains nil or duplicate node")
		}
		known[node] = true
		ids[node.ID] = true
	}
	if !known[t.node] {
		return nil, nil, fmt.Errorf("tensor bucket: tensor node is outside its graph")
	}
	state := make(map[*metalir.Node]uint8, len(known))
	ordered := make([]*metalir.Node, 0, len(known))
	var visit func(*metalir.Node) error
	visit = func(node *metalir.Node) error {
		if !known[node] {
			return fmt.Errorf("node %q references a node outside its graph", node.Name)
		}
		if state[node] == 1 {
			return fmt.Errorf("graph contains a cycle at node %q", node.Name)
		}
		if state[node] == 2 {
			return nil
		}
		shape, err := tensorShapeExpressions(node)
		if err != nil {
			return err
		}
		if err := validateSymbolicTensorNode(node, shape); err != nil {
			return err
		}
		state[node] = 1
		for _, child := range node.Inputs {
			if err := visit(child); err != nil {
				return err
			}
		}
		state[node] = 2
		ordered = append(ordered, node)
		return nil
	}
	// Include disconnected graph nodes in validation and lowering so malformed
	// or unresolved shapes cannot be hidden outside the requested root.
	for _, node := range t.graph.Nodes {
		if err := visit(node); err != nil {
			return nil, nil, fmt.Errorf("tensor bucket: validate symbolic graph: %w", err)
		}
	}
	shapes := make(map[string]egraph.DimensionShape, len(ordered))
	for _, node := range ordered {
		expressions, err := tensorShapeExpressions(node)
		if err != nil {
			return nil, nil, err
		}
		shapes[tensorNodeKey(node)] = egraph.NewDimensionShape(expressions)
	}
	return ordered, shapes, nil
}

func tensorShapeExpressions(node *metalir.Node) ([]metalshape.Expr, error) {
	if node == nil {
		return nil, fmt.Errorf("tensor bucket: nil node")
	}
	if node.ValidationError != nil {
		return nil, fmt.Errorf("node %q is invalid: %w", node.Name, node.ValidationError)
	}
	if node.ShapeExpr != nil {
		return append([]metalshape.Expr(nil), node.ShapeExpr...), nil
	}
	shape := make([]metalshape.Expr, len(node.Shape))
	for i, dimension := range node.Shape {
		value, err := metalshape.DimensionConstant(int64(dimension))
		if err != nil {
			return nil, fmt.Errorf("node %q dimension %d: %w", node.Name, i, err)
		}
		shape[i] = value
	}
	return shape, nil
}

func validateSymbolicTensorNode(node *metalir.Node, shape []metalshape.Expr) error {
	if node.Type.String() == "unknown" {
		return fmt.Errorf("node %q has an unknown data type", node.Name)
	}
	switch node.Op {
	case "rmsnorm":
		if !node.Attributes.HasEpsilon || node.Attributes.HasTheta || !positiveFinite(float64(node.Attributes.Epsilon)) {
			return fmt.Errorf("node %q has invalid RMSNorm epsilon", node.Name)
		}
	case "rope":
		if !node.Attributes.HasTheta || node.Attributes.HasEpsilon || !positiveFinite(float64(node.Attributes.Theta)) {
			return fmt.Errorf("node %q has invalid RoPE theta", node.Name)
		}
	default:
		if node.Attributes != (metalir.ScalarAttributes{}) {
			return fmt.Errorf("node %q has scalar attributes unsupported by %s", node.Name, node.Op)
		}
	}
	shapeEqual := func(other *metalir.Node) bool {
		otherShape, err := tensorShapeExpressions(other)
		if err != nil || len(shape) != len(otherShape) {
			return false
		}
		for i := range shape {
			if !shape[i].Equal(otherShape[i]) {
				return false
			}
		}
		return true
	}
	wrongArity := func(want int) error {
		if len(node.Inputs) != want {
			return fmt.Errorf("node %q (%s) wants %d inputs, got %d", node.Name, node.Op, want, len(node.Inputs))
		}
		return nil
	}
	switch node.Op {
	case "input", "constant":
		if len(node.Inputs) != 0 {
			return fmt.Errorf("node %q (%s) must be a leaf", node.Name, node.Op)
		}
	case "silu":
		if err := wrongArity(1); err != nil {
			return err
		}
		if node.Inputs[0].Type != node.Type || !shapeEqual(node.Inputs[0]) {
			return fmt.Errorf("node %q (silu) must preserve its input type and shape", node.Name)
		}
	case "add", "mul", "rmsnorm":
		if err := wrongArity(2); err != nil {
			return err
		}
		for i, input := range node.Inputs {
			if input.Type != node.Type || !shapeEqual(input) {
				return fmt.Errorf("node %q (%s) input %d has incompatible type or shape", node.Name, node.Op, i)
			}
		}
	case "rope":
		if err := wrongArity(2); err != nil {
			return err
		}
		if node.Inputs[0].Type != node.Type || !shapeEqual(node.Inputs[0]) {
			return fmt.Errorf("node %q (rope) must preserve its data input type and shape", node.Name)
		}
	case "matmul":
		if err := wrongArity(2); err != nil {
			return err
		}
		left, right := node.Inputs[0], node.Inputs[1]
		leftShape, leftErr := tensorShapeExpressions(left)
		rightShape, rightErr := tensorShapeExpressions(right)
		if leftErr != nil || rightErr != nil || left.Type != node.Type || right.Type != node.Type ||
			len(shape) != 2 || len(leftShape) != 2 || len(rightShape) != 2 ||
			!leftShape[1].Equal(rightShape[0]) || !shape[0].Equal(leftShape[0]) || !shape[1].Equal(rightShape[1]) {
			return fmt.Errorf("node %q (matmul) has incompatible rank, type, contraction, or output shape", node.Name)
		}
	}
	return nil
}

func tensorNodeKey(node *metalir.Node) string { return fmt.Sprintf("node-%d", node.ID) }
func tensorLeafIdentity(node *metalir.Node) string {
	return fmt.Sprintf("node-%d:%s", node.ID, node.Name)
}

func cloneBucketIntervals(intervals map[string]egraph.DimensionInterval) map[string]egraph.DimensionInterval {
	copy := make(map[string]egraph.DimensionInterval, len(intervals))
	for name, interval := range intervals {
		copy[name] = interval
	}
	return copy
}

func cloneBucketIndices(indices map[string]int) map[string]int {
	copy := make(map[string]int, len(indices))
	for name, index := range indices {
		copy[name] = index
	}
	return copy
}
