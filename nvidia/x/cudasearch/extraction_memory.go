package cudasearch

import (
	"context"
	"reflect"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type callbackReservationKey struct{}

func reservePendingExecutable(ctx context.Context) ([]Executable, *egraphcore.MemoryScope, error) {
	budget := searchMemoryBudget(ctx)
	if budget == nil || budget.limit == 0 {
		return nil, nil, nil
	}
	scope, err := egraphcore.NewMemoryScope(budget)
	if err != nil {
		return nil, nil, err
	}
	if err := scope.Reserve(int64(reflect.TypeFor[Executable]().Size())); err != nil {
		scope.Close()
		return nil, nil, err
	}
	return make([]Executable, 1), scope, nil
}

// declaredCostModel delegates under the opaque evaluator envelope already
// reserved by evaluateSearch. The envelope stays live through all synchronous
// callbacks and compilation in that evaluation; it is charged once. This is a
// caller declaration, not measurement or enforcement of callback allocations.
type declaredCostModel struct {
	egraph.CostModel
	envelope *memoryReservation
}

func (m *declaredCostModel) NodeCostWithMemory(node egraph.ENode, costs []float64, _ *egraphcore.MemoryScope) (float64, error) {
	if m.envelope == nil || m.envelope.bytes <= 0 {
		return 0, ErrMemoryAccountingUnavailable
	}
	return m.NodeCost(node, costs), nil
}

// Close closes every candidate owned by r. Aliased candidates share an
// idempotent extraction owner and release its reservations once. All owned
// expressions become invalid, including aliases held outside r. Borrowed
// candidate expressions and external resources are left unchanged.
// Close must not race with use or another Close.
func (r *SearchResult) Close() {
	if r == nil {
		return
	}
	r.Winner.Close()
	for _, candidate := range r.AllCandidates {
		if candidate != r.Winner {
			candidate.Close()
		}
	}
}
