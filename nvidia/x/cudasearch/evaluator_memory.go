package cudasearch

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"time"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func mockScoringMemory(ctx context.Context, m *MockEvaluator) (*scoringSnapshot, *egraphcore.MemoryScope, error) {
	if w, _ := ctx.Value(workBudgetKey{}).(*workBudget); w != nil && w.scoring != nil && w.scoring.owner == m {
		return w.scoring, nil, nil
	}
	scope, err := mockMemoryScope(ctx)
	if err != nil {
		return nil, nil, err
	}
	snapshot, err := snapshotScoringWithMemory(m, scope)
	if err != nil {
		scope.Close()
		return nil, nil, err
	}
	return snapshot, scope, nil
}

func mockMemoryScope(ctx context.Context) (*egraphcore.MemoryScope, error) {
	b := searchMemoryBudget(ctx)
	if b == nil || b.limit == 0 {
		return nil, nil
	}
	return egraphcore.NewMemoryScope(b)
}

// mockFormatBound bounds default fmt formatting without invoking caller methods.
// Opaque formatting methods have no allocation contract and refuse under a cap.
func mockFormatBound(v reflect.Value, scope *egraphcore.MemoryScope, depth int, visits *int) (int64, error) {
	*visits = *visits + 1
	if depth > 256 || *visits > 1000000 {
		return 0, ErrMemoryAccountingUnavailable
	}
	if !v.IsValid() {
		return 8, nil
	}
	for _, iface := range []reflect.Type{reflect.TypeFor[fmt.Formatter](), reflect.TypeFor[fmt.Stringer](), reflect.TypeFor[error]()} {
		if v.Type().Implements(iface) {
			return 0, ErrMemoryAccountingUnavailable
		}
	}
	var total int64 = 32 + int64(len(v.Type().String()))
	add := func(child reflect.Value) error {
		n, err := mockFormatBound(child, scope, depth+1, visits)
		if err != nil {
			return err
		}
		if err := setupAdd(&total, n); err != nil {
			return err
		}
		return setupAdd(&total, 2)
	}
	switch v.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr, reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		return total + 128, nil
	case reflect.String:
		if err := setupAdd(&total, int64(v.Len())); err != nil {
			return 0, err
		}
	case reflect.Interface, reflect.Pointer:
		if !v.IsNil() {
			if err := add(v.Elem()); err != nil {
				return 0, err
			}
		}
	case reflect.Array, reflect.Slice:
		for i := 0; i < v.Len(); i++ {
			if err := add(v.Index(i)); err != nil {
				return 0, err
			}
		}
	case reflect.Map:
		if err := bookkeepingSlots[struct{ key, value reflect.Value }](scope, v.Len()); err != nil {
			return 0, err
		}
		it := v.MapRange()
		for it.Next() {
			if err := add(it.Key()); err != nil {
				return 0, err
			}
			if err := add(it.Value()); err != nil {
				return 0, err
			}
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if err := add(v.Field(i)); err != nil {
				return 0, err
			}
		}
	default:
		return 0, ErrMemoryAccountingUnavailable
	}
	return total, nil
}

func reserveMockFingerprint(scope *egraphcore.MemoryScope, expr *egraph.Expr) error {
	if scope == nil {
		return nil
	}
	active, err := bookkeepingMap[*egraph.Expr, bool](scope)
	if err != nil {
		return err
	}
	var bytes int64
	visits := 0
	var walk func(*egraph.Expr, int) error
	walk = func(e *egraph.Expr, depth int) error {
		visits++
		if depth > 256 || visits > 1000000 || active[e] {
			return ErrMemoryAccountingUnavailable
		}
		if e == nil {
			return setupAdd(&bytes, 8)
		}
		if err := bookkeepingPut(scope, active, e, true); err != nil {
			return err
		}
		defer delete(active, e)
		if err := setupAdd(&bytes, int64(len(e.Op))); err != nil {
			return err
		}
		if err := setupAdd(&bytes, 4); err != nil {
			return err
		}
		if e.Data != nil {
			n, err := mockFormatBound(reflect.ValueOf(e.Data), scope, 0, &visits)
			if err != nil {
				return err
			}
			if err := setupAdd(&bytes, n); err != nil {
				return err
			}
		}
		// formatExpr has up to two events per edge plus a leave and closing
		// event per node. Charge conservative backing including growth copies.
		if len(e.Children) > (math.MaxInt-8)/4 {
			return ErrMemoryBudgetExceeded
		}
		if err := bookkeepingSlots[struct {
			expr  *egraph.Expr
			text  string
			leave bool
		}](scope, 4*(len(e.Children)+2)); err != nil {
			return err
		}
		for _, child := range e.Children {
			if err := walk(child, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk(expr, 0); err != nil {
		return err
	}
	if bytes > (math.MaxInt64-512)/8 {
		return ErrMemoryBudgetExceeded
	}
	// Builder growth, hash byte conversion, and fmt's temporary output.
	return scope.Reserve(8*bytes + 512)
}

func reserveMockMetadata(scope *egraphcore.MemoryScope, expr *egraph.Expr, derivation []egraph.Action) error {
	if scope == nil {
		return nil
	}
	for _, bytes := range []int64{int64(reflect.TypeFor[CandidatePlan]().Size()), int64(reflect.TypeFor[cudatrial.Receipt]().Size()), int64(reflect.TypeFor[ProfileEvidence]().Size()), int64(reflect.TypeFor[map[cudatrial.TimingKind]time.Duration]().Size()), 64, int64(len(expr.Op)) + 32} {
		if err := scope.Reserve(bytes); err != nil {
			return err
		}
	}
	if err := bookkeepingSlots[struct {
		key   cudatrial.TimingKind
		value time.Duration
	}](scope, 2); err != nil {
		return err
	}
	return bookkeepingSlots[egraph.Action](scope, len(derivation))
}

func newMockMetadata(ctx context.Context, expr *egraph.Expr, derivation []egraph.Action) (owned, scratch *egraphcore.MemoryScope, err error) {
	owned, err = mockMemoryScope(ctx)
	if err != nil {
		return nil, nil, err
	}
	scratch, err = mockMemoryScope(ctx)
	if err == nil {
		err = reserveMockMetadata(owned, expr, derivation)
	}
	if err == nil {
		err = reserveMockFingerprint(scratch, expr)
	}
	if err != nil {
		owned.Close()
		scratch.Close()
		return nil, nil, err
	}
	return owned, scratch, nil
}
