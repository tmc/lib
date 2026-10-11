package egraph

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/apple/x/metal/metalshape"
)

// DimensionBuckets assigns an ordered list of intervals to each dimension.
// The map's keys are sorted before Cartesian combinations are enumerated.
type DimensionBuckets map[string][]DimensionInterval

// BucketContext describes one Cartesian bucket combination. Its maps are
// private copies: callbacks may mutate them without affecting the search space.
type BucketContext struct {
	Index         int
	BucketIndices map[string]int
	Intervals     map[string]DimensionInterval
}

// BucketGraph is the independently built and saturated graph for one bucket.
// Roots are caller-selected e-class IDs in Graph.
type BucketGraph struct {
	Index         int
	BucketIndices map[string]int
	Intervals     map[string]DimensionInterval
	Graph         *EGraph
	Roots         []Id
	Saturation    SaturationResult
}

// SearchSpace retains the result for every successfully built bucket.
// Use Buckets to obtain defensive copies of its metadata and root slices.
type SearchSpace struct {
	buckets []BucketGraph
}

// Buckets returns results in deterministic Cartesian order. The returned
// maps and slices are copies; Graph pointers refer to the built mutable graphs.
func (s *SearchSpace) Buckets() []BucketGraph {
	if s == nil {
		return nil
	}
	out := make([]BucketGraph, len(s.buckets))
	for i, bucket := range s.buckets {
		out[i] = bucket
		out[i].BucketIndices = cloneBucketIndices(bucket.BucketIndices)
		out[i].Intervals = cloneDimensionIntervals(bucket.Intervals)
		out[i].Roots = append([]Id(nil), bucket.Roots...)
		out[i].Saturation.Phases = append([]SaturationPhaseResult(nil), bucket.Saturation.Phases...)
	}
	return out
}

// BucketGraphBuilder creates a fresh graph and the roots to retain for one
// bucket. The callback receives the bucket's intersected interval assumptions.
type BucketGraphBuilder func(BucketContext) (*EGraph, []Id, error)

// BucketShapeGraphBuilder receives expressions simplified under one bucket's
// interval assumptions, before its graph is saturated.
type BucketShapeGraphBuilder func(BucketContext, map[string][]metalshape.Expr) (*EGraph, []Id, error)

// SearchSpaceOptions bounds bucket expansion and saturation. Incomplete
// saturation is rejected unless explicitly allowed.
type SearchSpaceOptions struct {
	MaxCombinations int
	Saturation      SaturationOptions
	AllowIncomplete bool
}

// BuildSearchSpace builds one independent e-graph for each Cartesian bucket
// combination, then saturates each graph with the same schedule. Base
// intervals are intersected inclusively with selected buckets; a contradiction
// is an error. Bucket intervals for one dimension may not overlap. The positive
// MaxCombinations bound is checked without overflow before any callback is
// invoked. With no bucketed dimensions, one unbucketed graph is built. The
// builder must return a non-empty set of roots belonging to that graph. A
// non-fixed-point saturation returns an error unless AllowIncomplete is set.
// On any error, BuildSearchSpace returns no partial SearchSpace. Graphs retained
// in a successful result are mutable and are not safe for concurrent use.
func BuildSearchSpace(
	buckets DimensionBuckets,
	base map[string]DimensionInterval,
	opts SearchSpaceOptions,
	build BucketGraphBuilder,
	phases []SaturationPhase,
) (*SearchSpace, error) {
	phaseCopy := cloneSaturationPhases(phases)
	return buildSearchSpace(buckets, base, opts, build, func(graph *EGraph, opts SaturationOptions) (SaturationResult, error) {
		return SaturateSchedule(graph, phaseCopy, opts)
	})
}

// BuildSearchSpaceWithSchedule builds one independent e-graph for each bucket
// combination and executes schedule on each graph. The schedule is copied
// before graph construction; incomplete execution is rejected unless allowed.
func BuildSearchSpaceWithSchedule(
	buckets DimensionBuckets,
	base map[string]DimensionInterval,
	opts SearchSpaceOptions,
	build BucketGraphBuilder,
	schedule *RewriteSchedule,
) (*SearchSpace, error) {
	if err := validateSchedule(schedule); err != nil {
		return nil, fmt.Errorf("egraph: invalid bucket rewrite schedule: %w", err)
	}
	scheduleCopy := cloneRewriteSchedule(schedule)
	return buildSearchSpace(buckets, base, opts, build, func(graph *EGraph, opts SaturationOptions) (SaturationResult, error) {
		return ExecuteSchedule(graph, scheduleCopy, opts)
	})
}

func buildSearchSpace(
	buckets DimensionBuckets,
	base map[string]DimensionInterval,
	opts SearchSpaceOptions,
	build BucketGraphBuilder,
	saturate func(*EGraph, SaturationOptions) (SaturationResult, error),
) (*SearchSpace, error) {
	if build == nil {
		return nil, fmt.Errorf("egraph: nil bucket graph builder")
	}
	if saturate == nil {
		return nil, fmt.Errorf("egraph: nil bucket graph saturator")
	}
	if opts.MaxCombinations <= 0 {
		return nil, fmt.Errorf("egraph: max bucket combinations must be positive")
	}
	if err := validateDimensionIntervals(base); err != nil {
		return nil, fmt.Errorf("egraph: base intervals: %w", err)
	}
	dimensions := make([]string, 0, len(buckets))
	for dimension, intervals := range buckets {
		if dimension == "" {
			return nil, fmt.Errorf("egraph: bucket dimension must be non-empty")
		}
		if len(intervals) == 0 {
			return nil, fmt.Errorf("egraph: dimension %q has no buckets", dimension)
		}
		for i, interval := range intervals {
			if interval.Min < 0 {
				return nil, fmt.Errorf("egraph: dimension %q bucket %d has negative minimum", dimension, i)
			}
			if interval.Min > interval.Max {
				return nil, fmt.Errorf("egraph: dimension %q bucket %d has min greater than max", dimension, i)
			}
		}
		for i, interval := range intervals {
			for j := 0; j < i; j++ {
				other := intervals[j]
				if interval.Min <= other.Max && other.Min <= interval.Max {
					return nil, fmt.Errorf("egraph: dimension %q buckets %d and %d overlap inclusively", dimension, j, i)
				}
			}
		}
		dimensions = append(dimensions, dimension)
	}
	sort.Strings(dimensions)
	count := 1
	maxInt := int(^uint(0) >> 1)
	for _, dimension := range dimensions {
		n := len(buckets[dimension])
		if count > maxInt/n || count*n > opts.MaxCombinations {
			return nil, fmt.Errorf("egraph: bucket combination count exceeds limit %d", opts.MaxCombinations)
		}
		count *= n
	}
	space := &SearchSpace{buckets: make([]BucketGraph, 0, count)}
	indices := make(map[string]int, len(dimensions))
	contexts := make([]BucketContext, 0, count)
	var enumerate func(int) error
	enumerate = func(depth int) error {
		if depth < len(dimensions) {
			dimension := dimensions[depth]
			for i := range buckets[dimension] {
				indices[dimension] = i
				if err := enumerate(depth + 1); err != nil {
					return err
				}
			}
			delete(indices, dimension)
			return nil
		}
		intervals, err := intersectBucketIntervals(base, buckets, indices)
		if err != nil {
			return err
		}
		contexts = append(contexts, BucketContext{Index: len(contexts), BucketIndices: cloneBucketIndices(indices), Intervals: intervals})
		return nil
	}
	if err := enumerate(0); err != nil {
		return nil, err
	}
	usedGraphs := make(map[*EGraph]bool, count)
	for index, context := range contexts {
		context = BucketContext{Index: context.Index, BucketIndices: cloneBucketIndices(context.BucketIndices), Intervals: cloneDimensionIntervals(context.Intervals)}
		graph, roots, err := build(context)
		if err != nil {
			return nil, fmt.Errorf("egraph: build bucket %d (%v): %w", index, context.BucketIndices, err)
		}
		if graph == nil {
			return nil, fmt.Errorf("egraph: build bucket %d returned nil graph", index)
		}
		if usedGraphs[graph] {
			return nil, fmt.Errorf("egraph: build bucket %d reused a graph", index)
		}
		usedGraphs[graph] = true
		if len(roots) == 0 {
			return nil, fmt.Errorf("egraph: build bucket %d (%v) returned no roots", index, context.BucketIndices)
		}
		for i, root := range roots {
			if _, err := graph.FindChecked(root); err != nil {
				return nil, fmt.Errorf("egraph: bucket %d (%v) root %d: %w", index, context.BucketIndices, i, err)
			}
		}
		result, err := saturate(graph, opts.Saturation)
		if err != nil {
			return nil, fmt.Errorf("egraph: saturate bucket %d (%v): %w", index, context.BucketIndices, err)
		}
		if !opts.AllowIncomplete && result.StopReason != SaturationFixedPoint && result.StopReason != SaturationScheduleDone {
			return nil, fmt.Errorf("egraph: bucket %d (%v) saturation stopped at %q", index, context.BucketIndices, result.StopReason)
		}
		canonicalRoots := make([]Id, len(roots))
		for i, root := range roots {
			canonical, err := graph.FindChecked(root)
			if err != nil {
				return nil, fmt.Errorf("egraph: bucket %d (%v) root %d after saturation: %w", index, context.BucketIndices, i, err)
			}
			canonicalRoots[i] = canonical
		}
		space.buckets = append(space.buckets, BucketGraph{
			Index: index, BucketIndices: cloneBucketIndices(context.BucketIndices),
			Intervals: cloneDimensionIntervals(context.Intervals), Graph: graph,
			Roots: canonicalRoots, Saturation: result,
		})
	}
	return space, nil
}

func cloneRewriteSchedule(schedule *RewriteSchedule) *RewriteSchedule {
	if schedule == nil {
		return nil
	}
	copySchedule := &RewriteSchedule{
		Kind:        schedule.Kind,
		Name:        schedule.Name,
		Rules:       append([]*Rewrite(nil), schedule.Rules...),
		Repetitions: schedule.Repetitions,
		Children:    make([]*RewriteSchedule, len(schedule.Children)),
	}
	for i, child := range schedule.Children {
		copySchedule.Children[i] = cloneRewriteSchedule(child)
	}
	return copySchedule
}

// BuildShapeSearchSpace specializes supplied shape expressions for each bucket
// before calling build, then delegates graph construction and saturation to
// BuildSearchSpace. Every symbol referenced by a shape expression must have a
// valid interval in each bucket context. This helper supplies specialized
// shapes to the builder; it does not rewrite graph nodes or guarantee that the
// builder uses the shapes.
func BuildShapeSearchSpace(
	shapes map[string][]metalshape.Expr,
	buckets DimensionBuckets,
	base map[string]DimensionInterval,
	opts SearchSpaceOptions,
	build BucketShapeGraphBuilder,
	phases []SaturationPhase,
) (*SearchSpace, error) {
	if build == nil {
		return nil, fmt.Errorf("egraph: nil bucket shape graph builder")
	}
	shapeCopy := cloneShapeExpressions(shapes)
	for name := range shapeCopy {
		if name == "" {
			return nil, fmt.Errorf("egraph: shape name must be non-empty")
		}
	}
	return BuildSearchSpace(buckets, base, opts, func(context BucketContext) (*EGraph, []Id, error) {
		specialized := make(map[string][]metalshape.Expr, len(shapeCopy))
		for name, expressions := range shapeCopy {
			out := make([]metalshape.Expr, len(expressions))
			for i, expression := range expressions {
				value, err := expression.SimplifyWithIntervals(context.Intervals)
				if err != nil {
					return nil, nil, fmt.Errorf("shape %q dimension %d: %w", name, i, err)
				}
				out[i] = value
			}
			specialized[name] = out
		}
		return build(context, specialized)
	}, phases)
}

func cloneShapeExpressions(in map[string][]metalshape.Expr) map[string][]metalshape.Expr {
	out := make(map[string][]metalshape.Expr, len(in))
	for name, shape := range in {
		out[name] = append([]metalshape.Expr(nil), shape...)
	}
	return out
}

func validateDimensionIntervals(intervals map[string]DimensionInterval) error {
	for dimension, interval := range intervals {
		if dimension == "" {
			return fmt.Errorf("dimension name must be non-empty")
		}
		if interval.Min > interval.Max {
			return fmt.Errorf("dimension %q has min greater than max", dimension)
		}
		if interval.Min < 0 {
			return fmt.Errorf("dimension %q has negative minimum", dimension)
		}
	}
	return nil
}

func intersectBucketIntervals(base map[string]DimensionInterval, buckets DimensionBuckets, indices map[string]int) (map[string]DimensionInterval, error) {
	out := cloneDimensionIntervals(base)
	for dimension, bucketIndex := range indices {
		selected := buckets[dimension][bucketIndex]
		if existing, ok := out[dimension]; ok {
			selected.Min = maxInt64(selected.Min, existing.Min)
			selected.Max = minInt64(selected.Max, existing.Max)
			if selected.Min > selected.Max {
				return nil, fmt.Errorf("egraph: bucket interval for dimension %q does not overlap base interval", dimension)
			}
		}
		out[dimension] = selected
	}
	return out, nil
}

func cloneBucketIndices(in map[string]int) map[string]int {
	out := make(map[string]int, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func cloneDimensionIntervals(in map[string]DimensionInterval) map[string]DimensionInterval {
	out := make(map[string]DimensionInterval, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

func cloneSaturationPhases(in []SaturationPhase) []SaturationPhase {
	out := make([]SaturationPhase, len(in))
	for i, phase := range in {
		out[i] = phase
		out[i].Rules = append([]*Rewrite(nil), phase.Rules...)
	}
	return out
}

func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
