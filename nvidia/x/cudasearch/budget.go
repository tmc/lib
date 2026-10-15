package cudasearch

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type workBudgetKey struct{}

func validateSearchBudget(b SearchBudget) error {
	for _, limit := range []struct {
		name  string
		value int64
	}{
		{"states", int64(b.MaxStates)}, {"depth", int64(b.MaxDepth)},
		{"nodes per state", int64(b.MaxNodesPerState)}, {"classes per state", int64(b.MaxClassesPerState)},
		{"actions", int64(b.MaxActions)}, {"total actions", int64(b.MaxTotalActions)},
		{"memory", b.MaxMemoryBytes}, {"callback memory", b.CallbackMemoryBytes}, {"evaluations", int64(b.MaxEvaluations)},
		{"extractions", int64(b.MaxExtractions)}, {"compilations", int64(b.MaxCompilations)},
		{"extraction work", int64(b.MaxExtractionWork)},
		{"timeout", int64(b.Timeout)},
	} {
		if limit.value < 0 {
			return fmt.Errorf("cudasearch: negative %s budget", limit.name)
		}
	}
	return nil
}

func admitGraphBudget(s *egraph.SearchState, b SearchBudget) error {
	classes, nodes := s.Graph.Size()
	if b.MaxNodesPerState > 0 && nodes > b.MaxNodesPerState {
		return fmt.Errorf("%w: state has %d nodes, limit %d", ErrBudgetExhausted, nodes, b.MaxNodesPerState)
	}
	if b.MaxClassesPerState > 0 && classes > b.MaxClassesPerState {
		return fmt.Errorf("%w: state has %d classes, limit %d", ErrBudgetExhausted, classes, b.MaxClassesPerState)
	}
	return nil
}

// workBudget belongs to one sequential Search call, including its rollouts.
type workBudget struct {
	memory                            *memoryBudget
	callbackMemory                    int64
	extractionWork, maxExtractionWork int
	extractions, compilations         int
	maxExtractions, maxCompilations   int
	known                             bool
	exhausted                         bool
	scoring                           *scoringSnapshot
}

func searchWorkBudget(ctx context.Context, b SearchBudget, eval Evaluator) (context.Context, *workBudget, error) {
	w := &workBudget{memory: &memoryBudget{limit: b.MaxMemoryBytes}, callbackMemory: b.CallbackMemoryBytes, maxExtractions: b.MaxExtractions, maxCompilations: b.MaxCompilations}
	w.maxExtractionWork = b.MaxExtractionWork
	switch e := eval.(type) {
	case *ExecutableEvaluator:
		w.known = true
	case *MockEvaluator:
		w.known = true
		w.scoring = snapshotScoring(e)
	}
	if (b.MaxExtractions > 0 || b.MaxCompilations > 0 || b.MaxExtractionWork > 0) && !w.known {
		return ctx, nil, fmt.Errorf("cudasearch: evaluator does not support extraction and compilation budgets")
	}
	ctx = context.WithValue(ctx, memoryBudgetKey{}, w.memory)
	return context.WithValue(ctx, workBudgetKey{}, w), w, nil
}

func extractSearch(ctx context.Context, s *egraph.SearchState, root egraph.Id, model egraph.CostModel) (*egraph.Expr, float64, error) {
	limits := egraph.DefaultExtractionLimits()
	w, _ := ctx.Value(workBudgetKey{}).(*workBudget)
	globalLimit := false
	if w != nil && w.maxExtractionWork > 0 {
		remaining := w.maxExtractionWork - w.extractionWork
		if remaining <= 0 {
			w.exhausted = true
			return nil, 0, ErrBudgetExhausted
		}
		if remaining <= limits.MaxWork {
			limits.MaxWork = remaining
			globalLimit = true
		}
	}
	expr, cost, stats, err := egraph.ExtractBoundedWithStats(ctx, s.Graph, root, model, limits)
	if w != nil {
		w.extractionWork += stats.Work
		if globalLimit && stats.WorkExhausted {
			w.exhausted = true
			return nil, 0, fmt.Errorf("%w: %w", ErrBudgetExhausted, err)
		}
	}
	return expr, cost, err
}

func chargeWork(ctx context.Context, compile bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	w, _ := ctx.Value(workBudgetKey{}).(*workBudget)
	if w == nil {
		return nil
	}
	count, limit := &w.extractions, w.maxExtractions
	if compile {
		count, limit = &w.compilations, w.maxCompilations
	}
	if limit > 0 && *count >= limit {
		w.exhausted = true
		return ErrBudgetExhausted
	}
	*count++
	return nil
}
