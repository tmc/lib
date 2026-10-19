package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
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
	book      *searchBookkeeping
}

// Policy bookkeeping uses the same declared units as graph memory: value
// sizes, requested slice capacities, string bytes and a header plus one key/
// value slot per map insertion. Map buckets, allocator rounding and runtime
// implementation storage are not observable and are outside these units.
// Old slice backing remains charged until its scope closes, conservatively
// covering growth copies. Scratch closes on Search return; result storage
// transfers to SearchResult.Close. Neither scope owns caller data.
type searchBookkeeping struct {
	scratch, result *egraphcore.MemoryScope
	ledger          *memoryBudget
	transferred     bool
}

func newSearchBookkeeping(ctx context.Context) (*searchBookkeeping, error) {
	b := searchMemoryBudget(ctx)
	if b == nil || b.limit == 0 {
		return nil, nil
	}
	scratch, err := egraphcore.NewMemoryScope(b)
	if err != nil {
		return nil, err
	}
	if err := scratch.Reserve(int64(reflect.TypeFor[searchBookkeeping]().Size())); err != nil {
		scratch.Close()
		return nil, err
	}
	result, err := egraphcore.NewMemoryScope(b)
	if err != nil {
		scratch.Close()
		return nil, err
	}
	// The result scopes retain the ledger after the per-search work owner
	// closes. Bootstrap accounting covers it before these scopes exist.
	if err := result.Reserve(int64(reflect.TypeFor[SearchResult]().Size() + reflect.TypeFor[memoryBudget]().Size())); err != nil {
		scratch.Close()
		result.Close()
		return nil, err
	}
	book := &searchBookkeeping{scratch: scratch, result: result, ledger: b}
	b.book = book
	return book, nil
}

func (b *searchBookkeeping) close() {
	if b == nil {
		return
	}
	b.scratch.Close()
	b.ledger.book = nil
	if !b.transferred {
		b.result.Close()
	}
}

func (b *searchBookkeeping) takeResult() *egraphcore.MemoryScope {
	if b == nil {
		return nil
	}
	b.transferred = true
	return b.result
}

func (b *searchBookkeeping) temporary() *egraphcore.MemoryScope {
	if b == nil {
		return nil
	}
	return b.scratch
}

func (b *searchBookkeeping) retained() *egraphcore.MemoryScope {
	if b == nil {
		return nil
	}
	return b.result
}

func bookkeepingSlots[T any](scope *egraphcore.MemoryScope, count int) error {
	bytes, err := memoryProduct(count, int64(reflect.TypeFor[T]().Size()))
	if err != nil {
		return err
	}
	return scope.Reserve(bytes)
}

func bookkeepingAppend[T any](scope *egraphcore.MemoryScope, values []T, value T) ([]T, error) {
	if len(values) == cap(values) {
		if cap(values) > math.MaxInt/2 {
			return values, ErrMemoryBudgetExceeded
		}
		capacity := max(1, 2*cap(values))
		if err := bookkeepingSlots[T](scope, capacity); err != nil {
			return values, err
		}
		grown := make([]T, len(values), capacity)
		copy(grown, values)
		values = grown
	}
	return append(values, value), nil
}

func bookkeepingMap[K comparable, V any](scope *egraphcore.MemoryScope) (map[K]V, error) {
	if err := bookkeepingSlots[map[K]V](scope, 1); err != nil {
		return nil, err
	}
	return make(map[K]V), nil
}

func bookkeepingPut[K comparable, V any](scope *egraphcore.MemoryScope, values map[K]V, key K, value V) error {
	if _, exists := values[key]; !exists {
		if err := bookkeepingSlots[struct {
			key   K
			value V
		}](scope, 1); err != nil {
			return err
		}
	}
	values[key] = value
	return nil
}

// Count ties before drawing once, preserving choose's seeded order without
// allocating a temporary choices slice for every expansion and rollout.
func chooseSearchAction(profile profileActions, actions []egraph.ActionCandidate, rng *rand.Rand) int {
	if len(profile.ranks) == 0 && len(profile.weights) == 0 {
		return rng.Intn(len(actions))
	}
	best, weight, count := len(profile.ranks), math.Inf(-1), 0
	for _, action := range actions {
		r, w := profile.rank(action), profile.memoryWeight(action.Rule.Name)
		if w > weight || w == weight && r < best {
			best, weight, count = r, w, 0
		}
		if w == weight && r == best {
			count++
		}
	}
	selected := rng.Intn(count)
	for i, action := range actions {
		if profile.rank(action) == best && profile.memoryWeight(action.Rule.Name) == weight {
			if selected == 0 {
				return i
			}
			selected--
		}
	}
	return 0
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
	return storageBytesWithMemory(value, nil)
}

// The guarded rejection path uses the existing walker with its visited-map
// storage admitted too. Keeping that scratch in the result scope is
// conservative. Reflection never invokes Error, String or arbitrary callbacks;
// unsupported opaque fields still refuse instead of becoming free storage.
func storageBytesWithMemory(value any, scope *egraphcore.MemoryScope) (int64, error) {
	var total int64
	visits := 0
	active, err := bookkeepingMap[storageVisit, bool](scope)
	if err != nil {
		return 0, err
	}
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
			if err := bookkeepingPut(scope, active, key, true); err != nil {
				return err
			}
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
	err = walk(reflect.ValueOf(value), true, 0)
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

// A failed transition or no-op owner is returned unlinked so the caller can
// finish inspecting the error or result: terminal stop/error owners transfer
// into the search/result list, while ordinary rejected or no-op owners close
// after inspection. Productive owners transfer to the search/result list.
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
		// The caller inspects the error before releasing or retaining its owner.
		return nil, c, cloneMemoryError(err)
	}
	// Built-in transition matching, staging and growth reserve into the clone
	// scope before allocation. Keep reachable-storage measurement as a
	// conservative extra charge joined to the actual owner.
	bytes, measureErr := storageBytes(result.State)
	if measureErr != nil && b.limit > 0 {
		return nil, c, measureErr
	}
	if err := c.memory.Reserve(bytes); err != nil {
		return nil, c, err
	}
	c.state = result.State
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
	b := searchMemoryBudget(ctx)
	var scope *egraphcore.MemoryScope
	if b != nil {
		scope = b.book.retained()
	}
	return rejectionWithMemory(scope, state, eval, failure)
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
