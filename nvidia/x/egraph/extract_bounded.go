package egraph

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// ErrExtractionLimit reports exhausted extraction work, output size or depth.
var ErrExtractionLimit = errors.New("egraph: extraction limit exceeded")

// ErrPendingGraphChanges reports that bounded extraction needs a clean graph.
var ErrPendingGraphChanges = errors.New("egraph: graph has pending changes")

// ExtractionLimits bounds one extraction. Every limit must be positive.
// Work charges a node plus its child edges during cost computation, and one
// unit per materialized expression node. CostModel calls must return promptly;
// a synchronous callback cannot be forcibly interrupted.
type ExtractionLimits struct{ MaxWork, MaxNodes, MaxDepth int }

// DefaultExtractionLimits bounds cost work and repeated DAG expansion into trees.
func DefaultExtractionLimits() ExtractionLimits {
	return ExtractionLimits{MaxWork: 1000000, MaxNodes: 65536, MaxDepth: 256}
}

type extractionBudget struct {
	ctx           context.Context
	limits        ExtractionLimits
	work, nodes   int
	workExhausted bool
}

// ExtractionStats records completed work, including work in failed attempts.
type ExtractionStats struct {
	Work, Nodes   int
	WorkExhausted bool
}

func (b *extractionBudget) take(n int) error {
	if b == nil {
		return nil
	}
	if err := b.ctx.Err(); err != nil {
		return err
	}
	if n < 0 || n > b.limits.MaxWork-b.work {
		b.workExhausted = true
		return ErrExtractionLimit
	}
	b.work += n
	return nil
}

func (b *extractionBudget) node(depth int) error {
	if b == nil {
		return nil
	}
	if err := b.take(1); err != nil {
		return err
	}
	if depth > b.limits.MaxDepth || b.nodes >= b.limits.MaxNodes {
		return ErrExtractionLimit
	}
	b.nodes++
	return nil
}

// ExtractBounded computes a concrete expression with bounded cost work and tree
// allocation. The graph must be clean; call RebuildChecked before extraction.
// Cancellation or exhaustion returns no partial expression. It uses the same
// finite fixed-point pass limit as Extractor, not a global optimum proof.
func ExtractBounded(ctx context.Context, g *EGraph, root Id, model CostModel, limits ExtractionLimits) (*Expr, float64, error) {
	expr, cost, _, err := ExtractBoundedWithStats(ctx, g, root, model, limits)
	return expr, cost, err
}

// ExtractBoundedWithStats performs bounded extraction and reports its usage.
// The graph must be clean; call RebuildChecked first. WorkExhausted
// distinguishes work exhaustion from output size/depth limits.
func ExtractBoundedWithStats(ctx context.Context, g *EGraph, root Id, model CostModel, limits ExtractionLimits) (expr *Expr, cost float64, stats ExtractionStats, err error) {
	if ctx == nil || g == nil || limits.MaxWork <= 0 || limits.MaxNodes <= 0 || limits.MaxDepth <= 0 {
		return nil, 0, stats, fmt.Errorf("egraph: invalid bounded extraction request")
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, stats, err
	}
	if g.store == nil || !g.store.Initialized() {
		return nil, 0, stats, fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if len(g.worklist) != 0 {
		return nil, 0, stats, fmt.Errorf("%w; call RebuildChecked before bounded extraction", ErrPendingGraphChanges)
	}
	if err := g.RebuildChecked(); err != nil {
		return nil, 0, stats, fmt.Errorf("egraph: rebuild before extraction: %w", err)
	}
	b := &extractionBudget{ctx: ctx, limits: limits}
	defer func() { stats = ExtractionStats{Work: b.work, Nodes: b.nodes, WorkExhausted: b.workExhausted} }()
	if err := b.take(0); err != nil {
		return nil, 0, stats, err
	}
	if model == nil {
		model = AstSizeCost{}
	}
	ext := &Extractor{g: g, costModel: model, costs: make(map[Id]float64), bestNodes: make(map[Id]ENode), revision: g.revision}
	if err := ext.computeCostsBounded(b); err != nil {
		return nil, 0, stats, err
	}
	id := g.Find(root)
	cost, ok := ext.costs[id]
	if !ok || math.IsNaN(cost) || math.IsInf(cost, 0) {
		return nil, 0, stats, fmt.Errorf("egraph: no finite expression for e-class %d", root)
	}
	expr, err = ext.buildExprBounded(id, make(map[Id]bool), b, 1)
	if err != nil {
		return nil, 0, stats, err
	}
	return expr, cost, stats, nil
}
