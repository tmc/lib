package egraph

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalshape"
)

// DimensionShape is an immutable shape payload for an ENode. Use
// NewDimensionShape to copy the expression slice before storing it in a graph.
type DimensionShape struct {
	dimensions []metalshape.Expr
	identity   string
}

// DimensionNodeData pairs an analysis-visible shape with operator payload.
// The payload remains part of ENode identity, so scalar operator parameters
// survive hash-consing and extraction while DimensionAnalysis reads Shape.
type DimensionNodeData struct {
	shape   DimensionShape
	payload any
}

// NewDimensionNodeData creates a node payload carrying both shape and op data.
func NewDimensionNodeData(shape DimensionShape, payload any) DimensionNodeData {
	return DimensionNodeData{shape: NewDimensionShape(shape.dimensions, shape.identity), payload: payload}
}

// Shape returns the immutable shape payload.
func (d DimensionNodeData) Shape() DimensionShape {
	return NewDimensionShape(d.shape.dimensions, d.shape.identity)
}

// Payload returns the operation-specific payload.
func (d DimensionNodeData) Payload() any { return d.payload }

// NewDimensionShape copies dimensions into an immutable graph payload. An
// optional identity distinguishes otherwise equal leaf payloads, such as
// separate tensor inputs with the same shape.
func NewDimensionShape(dimensions []metalshape.Expr, identity ...string) DimensionShape {
	shape := DimensionShape{dimensions: append([]metalshape.Expr(nil), dimensions...)}
	if len(identity) > 0 {
		shape.identity = identity[0]
	}
	return shape
}

// Dimensions returns a defensive copy of the shape expressions.
func (s DimensionShape) Dimensions() []metalshape.Expr {
	return append([]metalshape.Expr(nil), s.dimensions...)
}

// Identity returns the optional identity attached to a shape payload.
func (s DimensionShape) Identity() string { return s.identity }

// DimensionFact is an immutable symbolic shape and interval fact for an
// e-class. Bounds returns a defensive copy. A conflict or invalid expression
// is reported by ConflictOrInvalid.
type DimensionFact struct {
	bounds   []metalshape.Interval
	shape    []metalshape.Expr
	identity string
	known    bool
	conflict bool
	invalid  bool
}

// Bounds returns the known per-dimension intervals and whether they are
// available. The returned slice is a copy.
func (f DimensionFact) Bounds() ([]metalshape.Interval, bool) {
	if !f.known || f.conflict || f.invalid {
		return nil, false
	}
	return append([]metalshape.Interval(nil), f.bounds...), true
}

// Shape returns the bucket-specialized symbolic dimensions for a known fact.
// The returned slice is a copy.
func (f DimensionFact) Shape() ([]metalshape.Expr, bool) {
	if !f.known || f.conflict || f.invalid {
		return nil, false
	}
	return append([]metalshape.Expr(nil), f.shape...), true
}

// Known reports whether this fact has a non-conflicting shape.
func (f DimensionFact) Known() bool { return f.known && !f.conflict && !f.invalid }

// ConflictOrInvalid reports incompatible shapes or invalid expression bounds.
func (f DimensionFact) ConflictOrInvalid() bool { return f.conflict || f.invalid }

// DimensionAnalysis propagates shape intervals under immutable bucket
// assumptions. Construct one analysis per bucket with NewDimensionAnalysis.
type DimensionAnalysis struct {
	intervals map[string]metalshape.Interval
}

// NewDimensionAnalysis creates an analysis with a defensive copy of intervals.
func NewDimensionAnalysis(intervals map[string]metalshape.Interval) (*DimensionAnalysis, error) {
	copyIntervals := make(map[string]metalshape.Interval, len(intervals))
	for symbol, interval := range intervals {
		if symbol == "" {
			return nil, fmt.Errorf("egraph: dimension interval symbol must be non-empty")
		}
		if interval.Min < 0 || interval.Min > interval.Max {
			return nil, fmt.Errorf("egraph: invalid interval for dimension symbol %q", symbol)
		}
		copyIntervals[symbol] = interval
	}
	return &DimensionAnalysis{intervals: copyIntervals}, nil
}

// Make derives bounds for a node carrying a DimensionShape payload. Nodes
// without that payload have unknown shape facts.
func (a *DimensionAnalysis) Make(_ *EGraph, node ENode) any {
	shape, ok := dimensionShapeFromData(node.Data)
	if !ok {
		return nil
	}
	fact := DimensionFact{
		known:  true,
		shape:  append([]metalshape.Expr(nil), shape.dimensions...),
		bounds: make([]metalshape.Interval, len(shape.dimensions)),
	}
	if payload, ok := node.Data.(DimensionNodeData); ok && payload.payload != nil {
		fact.identity = fmt.Sprintf("%T:%#v", payload.payload, payload.payload)
	}
	for i, expression := range shape.dimensions {
		bounds, err := expression.Bounds(a.intervals)
		if err != nil {
			return DimensionFact{known: true, conflict: true, invalid: true}
		}
		fact.bounds[i] = bounds
	}
	return fact
}

func dimensionShapeFromData(data any) (DimensionShape, bool) {
	switch value := data.(type) {
	case DimensionShape:
		return value, true
	case DimensionNodeData:
		return value.shape, true
	default:
		return DimensionShape{}, false
	}
}

// Merge combines facts for equivalent nodes. Unknown is the identity; a
// conflict is absorbing, and invalidity is retained when either input is
// invalid. Symbolic dimensions must match structurally after bucket
// simplification; overlapping intervals alone do not prove shape equality.
// Matching intervals are intersected because equivalent e-classes satisfy
// both bounds.
func (a *DimensionAnalysis) Merge(left, right any) (any, bool) {
	lf, lok := left.(DimensionFact)
	rf, rok := right.(DimensionFact)
	if left == nil {
		if right == nil {
			return nil, false
		}
		if !rok {
			return DimensionFact{known: true, conflict: true, invalid: true}, true
		}
		return cloneDimensionFact(rf), true
	}
	if right == nil {
		if !lok {
			return DimensionFact{known: true, conflict: true, invalid: true}, true
		}
		return cloneDimensionFact(lf), false
	}
	if !lok || !rok {
		merged := DimensionFact{known: true, conflict: true, invalid: true}
		return merged, !lok || !sameDimensionFact(lf, merged)
	}
	merged := mergeDimensionFacts(lf, rf)
	return merged, !sameDimensionFact(lf, merged)
}

func mergeDimensionFacts(a, b DimensionFact) DimensionFact {
	if a.conflict || b.conflict || a.invalid || b.invalid {
		return DimensionFact{known: true, conflict: true, invalid: a.invalid || b.invalid}
	}
	if !a.known {
		return cloneDimensionFact(b)
	}
	if !b.known {
		return cloneDimensionFact(a)
	}
	if len(a.bounds) != len(b.bounds) || len(a.shape) != len(b.shape) {
		return DimensionFact{known: true, conflict: true}
	}
	if a.identity != "" && b.identity != "" && a.identity != b.identity {
		return DimensionFact{known: true, conflict: true}
	}
	bounds := make([]metalshape.Interval, len(a.bounds))
	for i := range a.bounds {
		if !a.shape[i].Equal(b.shape[i]) {
			return DimensionFact{known: true, conflict: true}
		}
		bounds[i] = metalshape.Interval{
			Min: maxInt64(a.bounds[i].Min, b.bounds[i].Min),
			Max: minInt64(a.bounds[i].Max, b.bounds[i].Max),
		}
		if bounds[i].Min > bounds[i].Max {
			return DimensionFact{known: true, conflict: true}
		}
	}
	return DimensionFact{
		known:    true,
		shape:    append([]metalshape.Expr(nil), a.shape...),
		bounds:   bounds,
		identity: firstNonempty(a.identity, b.identity),
	}
}

func firstNonempty(left, right string) string {
	if left != "" {
		return left
	}
	return right
}

func cloneDimensionFact(f DimensionFact) DimensionFact {
	f.bounds = append([]metalshape.Interval(nil), f.bounds...)
	f.shape = append([]metalshape.Expr(nil), f.shape...)
	return f
}

func sameDimensionFact(a, b DimensionFact) bool {
	if a.known != b.known || a.conflict != b.conflict || a.invalid != b.invalid || a.identity != b.identity || len(a.bounds) != len(b.bounds) || len(a.shape) != len(b.shape) {
		return false
	}
	for i := range a.bounds {
		if a.bounds[i] != b.bounds[i] || !a.shape[i].Equal(b.shape[i]) {
			return false
		}
	}
	return true
}

// DimensionShapeGraphBuilder populates a per-bucket graph with nodes carrying
// DimensionShape payloads and returns the roots to retain.
type DimensionShapeGraphBuilder func(BucketContext, *EGraph, map[string]DimensionShape) ([]Id, error)

// BuildDimensionSearchSpace creates a fresh e-graph with a bucket-specific
// DimensionAnalysis, supplies specialized named shapes and the graph to build,
// and saturates it. Builders must attach NewDimensionShape payloads to relevant
// nodes; arbitrary ENode.Data shapes are not inferred automatically. The build
// fails if any class has conflicting or invalid dimension facts after
// saturation.
func BuildDimensionSearchSpace(
	shapes map[string]DimensionShape,
	buckets DimensionBuckets,
	base map[string]DimensionInterval,
	opts SearchSpaceOptions,
	build DimensionShapeGraphBuilder,
	phases []SaturationPhase,
) (*SearchSpace, error) {
	return buildDimensionSearchSpace(shapes, buckets, base, opts, build, func(builder BucketGraphBuilder) (*SearchSpace, error) {
		return BuildSearchSpace(buckets, base, opts, builder, phases)
	})
}

// BuildDimensionSearchSpaceWithSchedule mirrors BuildDimensionSearchSpace
// while executing a composable schedule for every bucket graph.
func BuildDimensionSearchSpaceWithSchedule(
	shapes map[string]DimensionShape,
	buckets DimensionBuckets,
	base map[string]DimensionInterval,
	opts SearchSpaceOptions,
	build DimensionShapeGraphBuilder,
	schedule *RewriteSchedule,
) (*SearchSpace, error) {
	return buildDimensionSearchSpace(shapes, buckets, base, opts, build, func(builder BucketGraphBuilder) (*SearchSpace, error) {
		return BuildSearchSpaceWithSchedule(buckets, base, opts, builder, schedule)
	})
}

func buildDimensionSearchSpace(
	shapes map[string]DimensionShape,
	buckets DimensionBuckets,
	base map[string]DimensionInterval,
	opts SearchSpaceOptions,
	build DimensionShapeGraphBuilder,
	search func(BucketGraphBuilder) (*SearchSpace, error),
) (*SearchSpace, error) {
	if build == nil {
		return nil, fmt.Errorf("egraph: nil dimension graph builder")
	}
	if search == nil {
		return nil, fmt.Errorf("egraph: nil dimension search function")
	}
	shapeCopy := make(map[string]DimensionShape, len(shapes))
	for name, shape := range shapes {
		if name == "" {
			return nil, fmt.Errorf("egraph: dimension shape name must be non-empty")
		}
		shapeCopy[name] = NewDimensionShape(shape.dimensions)
	}
	space, err := search(func(context BucketContext) (*EGraph, []Id, error) {
		analysis, err := NewDimensionAnalysis(context.Intervals)
		if err != nil {
			return nil, nil, err
		}
		graph := NewWithAnalysis(analysis)
		specialized := make(map[string]DimensionShape, len(shapeCopy))
		for name, shape := range shapeCopy {
			dimensions := make([]metalshape.Expr, len(shape.dimensions))
			for i, expression := range shape.dimensions {
				value, err := expression.SimplifyWithIntervals(context.Intervals)
				if err != nil {
					return nil, nil, fmt.Errorf("shape %q dimension %d: %w", name, i, err)
				}
				dimensions[i] = value
			}
			specialized[name] = NewDimensionShape(dimensions)
		}
		roots, err := build(context, graph, specialized)
		if err != nil {
			return nil, nil, err
		}
		return graph, roots, nil
	})
	if err != nil {
		return nil, err
	}
	for _, bucket := range space.buckets {
		classes, err := bucket.Graph.ClassesChecked()
		if err != nil {
			return nil, fmt.Errorf("egraph: inspect bucket %d analysis: %w", bucket.Index, err)
		}
		for _, class := range classes {
			fact, ok := class.AnalysisData.(DimensionFact)
			if ok && fact.ConflictOrInvalid() {
				return nil, fmt.Errorf("egraph: bucket %d class %d has conflicting or invalid dimension facts", bucket.Index, class.ID)
			}
		}
	}
	return space, nil
}
