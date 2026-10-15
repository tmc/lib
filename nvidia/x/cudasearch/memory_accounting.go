package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// memoryBudget records storage reservations for one search. A reservation is
// made before allocation and remains live until the corresponding owner is
// released. Its peak includes simultaneous retained and temporary storage.
// This is separate from the search-memory catalog of exploration priors.
type memoryBudget struct {
	mu        sync.Mutex
	limit     int64
	live      int64
	peak      int64
	exhausted bool
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

// A graph clone can add relation lineage records. Two source-storage units
// cover that bounded clone overhead; payload aliases are already charged per
// occurrence by storageBytes. Trim the reservation after measuring the clone.
func reserveStateClone(ctx context.Context, state *egraph.SearchState) (*memoryReservation, error) {
	b := searchMemoryBudget(ctx)
	bytes, err := storageBytes(state)
	if err != nil && b != nil && b.limit > 0 {
		return nil, err
	}
	if bytes > math.MaxInt64/2 {
		return nil, ErrMemoryBudgetExceeded
	}
	return b.reserve(2 * bytes)
}

func (r *memoryReservation) trim(bytes int64) {
	if r == nil || r.budget == nil {
		return
	}
	r.budget.mu.Lock()
	defer r.budget.mu.Unlock()
	if bytes >= 0 && bytes < r.bytes {
		r.budget.live -= r.bytes - bytes
		r.bytes = bytes
	}
}

func finishStateClone(ctx context.Context, r *memoryReservation, state *egraph.SearchState) error {
	b := searchMemoryBudget(ctx)
	actual, err := storageBytes(state)
	if err != nil && b != nil && b.limit > 0 {
		r.release()
		return err
	}
	if actual > r.bytes {
		if _, err := b.reserve(actual - r.bytes); err != nil {
			r.release()
			return err
		}
	} else {
		r.trim(actual)
	}
	return nil
}

func cloneSearchState(ctx context.Context, state *egraph.SearchState) (*egraph.SearchState, error) {
	r, err := reserveStateClone(ctx, state)
	if err != nil {
		return nil, err
	}
	cloned := state.Clone()
	if err := finishStateClone(ctx, r, cloned); err != nil {
		return nil, err
	}
	return cloned, nil
}

func applySearchAction(ctx context.Context, state *egraph.SearchState, action egraph.ActionCandidate) (*egraph.TransitionResult, error) {
	r, err := reserveStateClone(ctx, state)
	if err != nil {
		return nil, err
	}
	result, err := state.ApplyAction(action)
	if err != nil || result.NoOp {
		r.release()
		return result, err
	}
	if err := finishStateClone(ctx, r, result.State); err != nil {
		return nil, err
	}
	return result, nil
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
