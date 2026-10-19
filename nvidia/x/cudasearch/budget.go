package cudasearch

import (
	"context"
	"fmt"
	"reflect"

	"github.com/tmc/lib/egraphcore"
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
	setup                             *egraphcore.MemoryScope
	bootstrap                         int64
}

func searchWorkBudget(ctx context.Context, b SearchBudget, eval Evaluator) (context.Context, *workBudget, error) {
	if err := ctx.Err(); err != nil {
		return ctx, nil, fmt.Errorf("%w: %w", ErrSearchCancelled, err)
	}
	bootstrap := int64(reflect.TypeFor[workBudget]().Size() + reflect.TypeFor[memoryBudget]().Size())
	if b.MaxMemoryBytes > 0 && bootstrap > b.MaxMemoryBytes {
		return ctx, nil, ErrMemoryBudgetExceeded
	}
	w := &workBudget{memory: &memoryBudget{limit: b.MaxMemoryBytes}, callbackMemory: b.CallbackMemoryBytes, maxExtractions: b.MaxExtractions, maxCompilations: b.MaxCompilations}
	if b.MaxMemoryBytes > 0 {
		if err := w.memory.Reserve(bootstrap); err != nil {
			return ctx, nil, err
		}
		w.bootstrap = bootstrap
		var err error
		w.setup, err = egraphcore.NewMemoryScope(w.memory)
		if err != nil {
			w.closeSearch()
			return ctx, nil, err
		}
	}
	ready := false
	defer func() {
		if !ready {
			w.closeSearch()
		}
	}()
	w.maxExtractionWork = b.MaxExtractionWork
	switch e := eval.(type) {
	case *ExecutableEvaluator:
		w.known = true
	case *MockEvaluator:
		w.known = true
		var err error
		w.scoring, err = snapshotScoringWithMemory(e, w.setup)
		if err != nil {
			return ctx, nil, fmt.Errorf("cudasearch: scoring setup: %w", err)
		}
	}
	if (b.MaxExtractions > 0 || b.MaxCompilations > 0 || b.MaxExtractionWork > 0) && !w.known {
		return ctx, nil, fmt.Errorf("cudasearch: evaluator does not support extraction and compilation budgets")
	}
	ctx = context.WithValue(ctx, memoryBudgetKey{}, w.memory)
	ready = true
	return context.WithValue(ctx, workBudgetKey{}, w), w, nil
}

// closeSearch releases search-lifetime storage. Returned result scopes and
// caller reservations keep their own owners; neither is released here.
func (w *workBudget) closeSearch() {
	if w == nil {
		return
	}
	closeStateClones(w.memory.takeClones())
	w.scoring = nil
	w.setup.Close()
	w.setup = nil
	w.memory.Release(w.bootstrap)
	w.bootstrap = 0
}

func extractSearch(ctx context.Context, s *egraph.SearchState, root egraph.Id, model egraph.CostModel) (*egraph.Extraction, error) {
	limits := egraph.DefaultExtractionLimits()
	w, _ := ctx.Value(workBudgetKey{}).(*workBudget)
	globalLimit := false
	if w != nil && w.maxExtractionWork > 0 {
		remaining := w.maxExtractionWork - w.extractionWork
		if remaining <= 0 {
			w.exhausted = true
			return nil, ErrBudgetExhausted
		}
		if remaining <= limits.MaxWork {
			limits.MaxWork = remaining
			globalLimit = true
		}
	}
	var memory *memoryBudget
	if w != nil && w.memory.limit > 0 {
		memory = w.memory
	}
	if memory != nil {
		switch model.(type) {
		case nil, egraph.AstSizeCost, egraph.FusedMegakernelCost, egraph.MemoryCostModel:
		default:
			envelope, _ := ctx.Value(callbackReservationKey{}).(*memoryReservation)
			if envelope == nil || envelope.budget != memory || envelope.bytes <= 0 {
				return nil, ErrMemoryAccountingUnavailable
			}
			adapter, err := egraphcore.NewMemoryScope(memory)
			if err != nil {
				return nil, err
			}
			if err := adapter.Reserve(int64(reflect.TypeFor[declaredCostModel]().Size())); err != nil {
				adapter.Close()
				return nil, err
			}
			defer adapter.Close()
			model = &declaredCostModel{CostModel: model, envelope: envelope}
		}
	}
	var owner *egraph.Extraction
	var err error
	if memory == nil {
		owner, err = egraph.ExtractBoundedWithMemory(ctx, s.Graph, root, model, limits, nil)
	} else {
		owner, err = egraph.ExtractBoundedWithMemory(ctx, s.Graph, root, model, limits, memory)
	}
	var stats egraph.ExtractionStats
	if owner != nil {
		stats = owner.Stats
	}
	if w != nil {
		w.extractionWork += stats.Work
		if globalLimit && stats.WorkExhausted {
			w.exhausted = true
			owner.Close()
			return nil, fmt.Errorf("%w: %w", ErrBudgetExhausted, err)
		}
	}
	if err != nil {
		owner.Close()
		return nil, err
	}
	return owner, nil
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
