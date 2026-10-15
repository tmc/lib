package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// memoryBudget records storage reservations for one search. Guarded operations
// reserve before allocation and keep a result owner. Some legacy bookkeeping
// still measures retained values after allocation. Its peak includes the
// simultaneous retained and temporary reservations, not all process storage.
// This is separate from the search-memory catalog of exploration priors.
type memoryBudget struct {
	mu        sync.Mutex
	limit     int64
	live      int64
	peak      int64
	exhausted bool
	clones    *stateClone
}

type memoryBudgetKey struct{}

type memoryReservation struct {
	budget *memoryBudget
	bytes  int64
	once   sync.Once
}

func (b *memoryBudget) reserve(bytes int64) (*memoryReservation, error) {
	if err := b.Reserve(bytes); err != nil {
		return nil, err
	}
	return &memoryReservation{budget: b, bytes: bytes}, nil
}

// Reserve and Release implement the shared matching allocation capability
// without allocating a separate ledger entry for every collection growth.
func (b *memoryBudget) Reserve(bytes int64) error {
	if bytes < 0 {
		return fmt.Errorf("cudasearch: negative memory reservation")
	}
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if bytes > math.MaxInt64-b.live || b.limit > 0 && bytes > b.limit-b.live {
		b.exhausted = true
		return ErrMemoryBudgetExceeded
	}
	b.live += bytes
	if b.live > b.peak {
		b.peak = b.live
	}
	return nil
}

func (b *memoryBudget) Release(bytes int64) {
	if b == nil {
		return
	}
	b.mu.Lock()
	b.live -= bytes
	b.mu.Unlock()
}

func (r *memoryReservation) release() {
	if r == nil || r.budget == nil {
		return
	}
	r.once.Do(func() {
		r.budget.Release(r.bytes)
	})
}

func reserveSearchMemory(ctx context.Context, bytes int64) (*memoryReservation, error) {
	b, _ := ctx.Value(memoryBudgetKey{}).(*memoryBudget)
	return b.reserve(bytes)
}

func (b *memoryBudget) accountedBytes() int64 {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.peak
}

// ReserveSearchMemory reserves owned storage before a callback allocates it.
// release must be called only after that storage is no longer retained; it is
// safe to call more than once. Outside Search there is no limit. Reservations
// account requested storage, not allocator overhead, process RSS, driver code,
// or compiler internals. Callbacks remain responsible for their declarations.
func ReserveSearchMemory(ctx context.Context, bytes int64) (release func(), err error) {
	r, err := reserveSearchMemory(ctx, bytes)
	if err != nil {
		return nil, err
	}
	return r.release, nil
}

func searchMemoryBudget(ctx context.Context) *memoryBudget {
	b, _ := ctx.Value(memoryBudgetKey{}).(*memoryBudget)
	return b
}

// ErrMemoryAccountingUnavailable reports storage or a callback whose capacity
// has not been declared for a search with a nonzero memory limit.
var ErrMemoryAccountingUnavailable = errors.New("cudasearch: memory accounting unavailable")

type storageVisit struct {
	typ      reflect.Type
	pointer  uintptr
	capacity int
}

// storageBytes counts value storage, slice capacity, strings, live map entries
// and reachable payloads. Map bucket capacity and allocator overhead are not
// observable through reflection and are deliberately outside these units.
// Aliases are counted independently, except back edges of cyclic payloads.
// This conservatively covers cloning payloads independently at separate nodes.
func storageBytes(value any) (int64, error) {
	var total int64
	visits := 0
	active := make(map[storageVisit]bool)
	add := func(n uint64) error {
		if n > uint64(math.MaxInt64-total) {
			return ErrMemoryBudgetExceeded
		}
		total += int64(n)
		return nil
	}
	var walk func(reflect.Value, bool, int) error
	walk = func(v reflect.Value, inline bool, depth int) error {
		if !v.IsValid() {
			return nil
		}
		visits++
		if visits > 1000000 || depth > 256 {
			return ErrMemoryAccountingUnavailable
		}
		if inline {
			if err := add(uint64(v.Type().Size())); err != nil {
				return err
			}
		}
		if v.Kind() == reflect.Pointer || v.Kind() == reflect.Map || v.Kind() == reflect.Slice {
			if v.IsNil() {
				return nil
			}
			key := storageVisit{typ: v.Type(), pointer: v.Pointer()}
			if v.Kind() == reflect.Slice {
				key.capacity = v.Cap()
			}
			if active[key] {
				return nil
			}
			active[key] = true
			defer delete(active, key)
		}
		switch v.Kind() {
		case reflect.Interface:
			if !v.IsNil() {
				return walk(v.Elem(), true, depth+1)
			}
		case reflect.Pointer:
			return walk(v.Elem(), true, depth+1)
		case reflect.String:
			return add(uint64(v.Len()))
		case reflect.Slice:
			if uint64(v.Cap()) > math.MaxInt64/uint64(max(v.Type().Elem().Size(), 1)) {
				return ErrMemoryBudgetExceeded
			}
			if err := add(uint64(v.Cap()) * uint64(v.Type().Elem().Size())); err != nil {
				return err
			}
			if !storageHasReferences(v.Type().Elem()) {
				return nil
			}
			v = v.Slice(0, v.Cap())
			for i := 0; i < v.Len(); i++ {
				if err := walk(v.Index(i), false, depth+1); err != nil {
					return err
				}
			}
		case reflect.Array:
			for i := 0; i < v.Len(); i++ {
				if err := walk(v.Index(i), false, depth+1); err != nil {
					return err
				}
			}
		case reflect.Map:
			iter := v.MapRange()
			for iter.Next() {
				if err := walk(iter.Key(), true, depth+1); err != nil {
					return err
				}
				if err := walk(iter.Value(), true, depth+1); err != nil {
					return err
				}
			}
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				if err := walk(v.Field(i), false, depth+1); err != nil {
					return err
				}
			}
		case reflect.Func, reflect.Chan, reflect.UnsafePointer:
			if !v.IsNil() {
				return ErrMemoryAccountingUnavailable
			}
		}
		return nil
	}
	err := walk(reflect.ValueOf(value), true, 0)
	return total, err
}

func retainSearchValue(ctx context.Context, value any) error {
	b := searchMemoryBudget(ctx)
	if b == nil {
		return nil
	}
	bytes, err := storageBytes(value)
	if err != nil {
		if b.limit == 0 {
			return nil
		}
		return err
	}
	_, err = b.reserve(bytes)
	return err
}

// stateClone owns an admitted branch and all its copy scratch on the existing
// ledger. The intrusive link needs no separately growing slice or map. Pruned
// states remain owned because candidate/rejection derivations can alias them.
type stateClone struct {
	state  *egraph.SearchState
	memory *egraphcore.MemoryScope
	work   *egraphcore.WorkBudget
	next   *stateClone
	closed bool
}

func (c *stateClone) close() {
	if c == nil || c.closed {
		return
	}
	c.closed = true
	c.state = nil
	c.memory.Close()
}

func closeStateClones(head *stateClone) {
	for c := head; c != nil; c = c.next {
		c.close()
	}
}

func (b *memoryBudget) takeClones() *stateClone {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	head := b.clones
	b.clones = nil
	return head
}

func (b *memoryBudget) retainClone(c *stateClone) {
	b.mu.Lock()
	c.next = b.clones
	b.clones = c
	b.mu.Unlock()
}

func newStateClone(ctx context.Context) (*stateClone, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b := searchMemoryBudget(ctx)
	scope, err := egraphcore.NewMemoryScope(b)
	if err != nil {
		return nil, err
	}
	if err := scope.Reserve(int64(reflect.TypeFor[stateClone]().Size())); err != nil {
		scope.Close()
		return nil, err
	}
	c := &stateClone{memory: scope}
	c.work, err = egraphcore.NewWorkBudgetWithMemory(0, scope)
	if err != nil {
		c.close()
		return nil, err
	}
	return c, nil
}

func cloneSearchState(ctx context.Context, state *egraph.SearchState) (*egraph.SearchState, error) {
	b := searchMemoryBudget(ctx)
	if b == nil {
		return state.Clone(), nil
	}
	c, err := newStateClone(ctx)
	if err != nil {
		return nil, err
	}
	if b.limit == 0 {
		// Unlimited APIs keep their ordinary payload/cycle semantics. This
		// snapshot is bookkeeping only and is not allocation-site admission.
		c.state = state.Clone()
		bytes, _ := storageBytes(c.state)
		err = c.memory.Reserve(bytes)
	} else {
		c.state, err = state.CloneWithBudget(ctx, c.work)
	}
	if err != nil {
		c.close()
		return nil, cloneMemoryError(err)
	}
	b.retainClone(c)
	return c.state, nil
}

func cloneMemoryError(err error) error {
	if errors.Is(err, egraph.ErrMemoryOperation) {
		return fmt.Errorf("%w: %w", ErrMemoryAccountingUnavailable, err)
	}
	return err
}

// A no-op owner is returned for the caller to close after inspecting the
// result. Productive owners transfer to the search/result list. Errors close
// immediately. This avoids returning a usable clone after releasing its charge.
func applySearchAction(ctx context.Context, state *egraph.SearchState, action egraph.ActionCandidate) (*egraph.TransitionResult, *stateClone, error) {
	b := searchMemoryBudget(ctx)
	if b == nil {
		result, err := state.ApplyAction(action)
		return result, nil, err
	}
	c, err := newStateClone(ctx)
	if err != nil {
		return nil, nil, err
	}
	var result *egraph.TransitionResult
	if b.limit == 0 {
		result, err = state.ApplyAction(action)
	} else {
		result, err = state.ApplyActionWithCloneBudget(ctx, action, c.work)
	}
	if err != nil {
		c.close()
		return nil, nil, cloneMemoryError(err)
	}
	c.state = result.State
	// Transition matching/staging/growth still has no preallocation contract.
	// Keep its existing reachable-storage measurement as a conservative extra
	// charge, now joined to the actual owner. This never qualifies its peak.
	bytes, measureErr := storageBytes(result.State)
	if measureErr != nil && b.limit > 0 {
		c.close()
		return nil, nil, measureErr
	}
	if err := c.memory.Reserve(bytes); err != nil {
		c.close()
		return nil, nil, err
	}
	if !result.NoOp {
		b.retainClone(c)
	}
	return result, c, nil
}

func evaluateSearch(ctx context.Context, eval Evaluator, state *egraph.SearchState, root egraph.Id) (*CandidatePlan, error) {
	w, _ := ctx.Value(workBudgetKey{}).(*workBudget)
	if w == nil {
		return eval.Evaluate(ctx, state, root)
	}
	known := false
	if m, ok := eval.(*MockEvaluator); ok {
		builtin := m.CostModel == nil
		switch m.CostModel.(type) {
		case egraph.AstSizeCost, egraph.FusedMegakernelCost, egraph.MemoryCostModel:
			builtin = true
		}
		known = builtin && m.FailCandidate == nil && m.Workload.ResetState == nil
	}
	if w.memory.limit > 0 && !known && w.callbackMemory == 0 {
		return nil, ErrMemoryAccountingUnavailable
	}
	r, err := w.memory.reserve(w.callbackMemory)
	if err != nil {
		return nil, err
	}
	defer func() {
		if e, ok := eval.(*ExecutableEvaluator); ok && len(e.pending) > 0 {
			e.memoryReservations = append(e.memoryReservations, r)
		} else {
			r.release()
		}
	}()
	ctx = context.WithValue(ctx, callbackReservationKey{}, r)
	candidate, err := eval.Evaluate(ctx, state, root)
	if err != nil {
		candidate.Close()
		return nil, err
	}
	value := candidate
	if candidate != nil && candidate.extraction != nil {
		copy := *candidate
		copy.extraction, copy.extractionLedger = nil, nil
		if w.memory.limit > 0 && candidate.extractionLedger == w.memory {
			copy.Expr = nil
		}
		value = &copy
	}
	if err := retainSearchValue(ctx, value); err != nil {
		candidate.Close()
		return nil, err
	}
	return candidate, nil
}

func storageHasReferences(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.String, reflect.Slice, reflect.Map, reflect.Pointer, reflect.Interface, reflect.Func, reflect.Chan, reflect.UnsafePointer:
		return true
	case reflect.Array:
		return storageHasReferences(t.Elem())
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if storageHasReferences(t.Field(i).Type) {
				return true
			}
		}
	}
	return false
}

func enumerateSearchActions(ctx context.Context, state *egraph.SearchState, rules []*egraph.Rewrite, limit int) (*egraph.ActionSet, error) {
	b := searchMemoryBudget(ctx)
	if b == nil || b.limit == 0 {
		return &egraph.ActionSet{Actions: egraph.EnumerateActionsBounded(state.Graph, rules, limit)}, nil
	}
	return egraph.EnumerateActionsWithMemory(state.Graph, rules, limit, b)
}

func retainSearchRejection(ctx context.Context, state *egraph.SearchState, eval Evaluator, failure error) (CandidateRejection, error) {
	r := rejection(state, eval, failure)
	return r, retainSearchValue(ctx, r)
}

func reserveBorrowedExecutable(ctx context.Context, x Executable) (*memoryReservation, error) {
	b := searchMemoryBudget(ctx)
	var bytes int64
	switch x := x.(type) {
	case *sequenceExecutable:
		if x.memory == b {
			return b.reserve(0)
		}
		for name, buffer := range x.buffers {
			n := int64(buffer.Len())
			if r := x.memoryReservations[name]; r != nil {
				n = r.bytes
			}
			if n > math.MaxInt64-bytes {
				return nil, ErrMemoryBudgetExceeded
			}
			bytes += n
		}
	case *prefillExecutable:
		if x.memory == b {
			return b.reserve(0)
		}
		for _, allocation := range []struct {
			count int
			size  int64
		}{{x.a.Len(), 2}, {x.b.Len(), 2}, {x.out.Len(), 4}} {
			n, err := memoryProduct(allocation.count, allocation.size)
			if err != nil || n > math.MaxInt64-bytes {
				return nil, ErrMemoryBudgetExceeded
			}
			bytes += n
		}
		if x.deviceBytes > bytes {
			bytes = x.deviceBytes
		}
	}
	return b.reserve(bytes)
}

func memoryProduct(count int, size int64) (int64, error) {
	if count < 0 || size < 0 || size != 0 && int64(count) > math.MaxInt64/size {
		return 0, ErrMemoryBudgetExceeded
	}
	return int64(count) * size, nil
}
