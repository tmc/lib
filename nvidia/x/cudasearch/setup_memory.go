package cudasearch

import (
	"encoding"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"time"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type searchPolicyOptions struct {
	beamWidth, iterations, rolloutDepth int
	exploration                         float64
}

func policyScopes(work *workBudget) (scratch, result *egraphcore.MemoryScope, err error) {
	if work == nil || work.memory.limit == 0 {
		return nil, nil, nil
	}
	if work.memory.book == nil {
		return nil, nil, ErrMemoryAccountingUnavailable
	}
	scratch, err = egraphcore.NewMemoryScope(work.memory)
	return scratch, work.memory.book.retained(), err
}

func setupAdd(total *int64, n int64) error {
	if n < 0 || n > math.MaxInt64-*total {
		return ErrMemoryBudgetExceeded
	}
	*total += n
	return nil
}

func setupStringBound(s string) (int64, error) {
	// JSON and Go quoted strings use at most six bytes per input byte.
	bytes, err := memoryProduct(len(s), 6)
	if err != nil || bytes > math.MaxInt64-2 {
		return 0, ErrMemoryBudgetExceeded
	}
	return bytes + 2, nil
}

// setupJSONBound sizes the supported policy values without calling marshalers.
// It counts every field, including omitted fields, and aliases independently.
// Private codec caches and reflection implementation storage are not units of
// this application ledger; produced byte storage and map sorting slots are.
func setupJSONBound(value any, scratch *egraphcore.MemoryScope) (int64, error) {
	visits := 0
	var size func(reflect.Value, int) (int64, error)
	size = func(v reflect.Value, depth int) (int64, error) {
		visits++
		if depth > 256 || visits > 1000000 {
			return 0, ErrMemoryAccountingUnavailable
		}
		if !v.IsValid() {
			return 4, nil
		}
		if v.Type() == reflect.TypeFor[time.Time]() {
			return 64, nil
		}
		if v.Type().Implements(reflect.TypeFor[json.Marshaler]()) || v.Type().Implements(reflect.TypeFor[encoding.TextMarshaler]()) {
			return 0, ErrMemoryAccountingUnavailable
		}
		switch v.Kind() {
		case reflect.Interface, reflect.Pointer:
			if v.IsNil() {
				return 4, nil
			}
			return size(v.Elem(), depth+1)
		case reflect.String:
			return setupStringBound(v.String())
		case reflect.Bool:
			return 5, nil
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64:
			return 32, nil
		case reflect.Map:
			if v.IsNil() {
				return 4, nil
			}
			if v.Type().Key().Kind() != reflect.String {
				return 0, ErrMemoryAccountingUnavailable
			}
			if err := bookkeepingSlots[struct {
				key, value reflect.Value
				name       string
			}](scratch, v.Len()); err != nil {
				return 0, err
			}
			var total int64 = 2
			it := v.MapRange()
			for it.Next() {
				for _, item := range []reflect.Value{it.Key(), it.Value()} {
					n, err := size(item, depth+1)
					if err != nil {
						return 0, err
					}
					if err := setupAdd(&total, n); err != nil {
						return 0, err
					}
				}
				if err := setupAdd(&total, 2); err != nil {
					return 0, err
				}
			}
			return total, nil
		case reflect.Slice, reflect.Array:
			var total int64 = 4
			for i := 0; i < v.Len(); i++ {
				n, err := size(v.Index(i), depth+1)
				if err != nil {
					return 0, err
				}
				if err := setupAdd(&total, n+1); err != nil {
					return 0, err
				}
			}
			return total, nil
		case reflect.Struct:
			var total int64 = 2
			for i := 0; i < v.NumField(); i++ {
				field := v.Type().Field(i)
				if field.PkgPath != "" || field.Tag.Get("json") == "-" {
					continue
				}
				name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
				if name == "" {
					name = field.Name
				}
				key, err := setupStringBound(name)
				if err != nil {
					return 0, err
				}
				n, err := size(v.Field(i), depth+1)
				if err != nil {
					return 0, err
				}
				for _, bytes := range []int64{key, n, 2} {
					if err := setupAdd(&total, bytes); err != nil {
						return 0, err
					}
				}
			}
			return total, nil
		}
		return 0, ErrMemoryAccountingUnavailable
	}
	return size(reflect.ValueOf(value), 0)
}

func setupRuleBytes(rule *egraph.Rewrite, scratch *egraphcore.MemoryScope) (int64, error) {
	active, err := bookkeepingMap[*egraph.Pattern, bool](scratch)
	if err != nil {
		return 0, err
	}
	var total int64
	visits := 0
	var pattern func(*egraph.Pattern, int) error
	pattern = func(p *egraph.Pattern, depth int) error {
		visits++
		if p == nil || active[p] || depth > 256 || visits > 1000000 {
			return ErrMemoryAccountingUnavailable
		}
		if err := bookkeepingPut(scratch, active, p, true); err != nil {
			return err
		}
		defer delete(active, p)
		// String recursively copies each subtree into its ancestors. Data is
		// deliberately absent from the existing Pattern.String identity.
		n, err := memoryProduct(len(p.Op), int64(depth+1))
		if err != nil {
			return err
		}
		if err := setupAdd(&total, n+int64(4*(depth+1))); err != nil {
			return err
		}
		if !p.IsVar {
			for _, child := range p.Children {
				if err := pattern(child, depth+1); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := setupAdd(&total, int64(len(rule.Name))+32); err != nil {
		return 0, err
	}
	for _, p := range []*egraph.Pattern{rule.LHS, rule.RHS} {
		if err := pattern(p, 0); err != nil {
			return 0, err
		}
	}
	for _, p := range rule.Conditions {
		if p == nil {
			if err := setupAdd(&total, 8); err != nil {
				return 0, err
			}
		} else if err := pattern(p, 0); err != nil {
			return 0, err
		}
	}
	for _, condition := range rule.RelationConditions {
		if err := setupAdd(&total, int64(len(condition.Name))+32); err != nil {
			return 0, err
		}
		for _, term := range condition.Terms {
			for _, s := range []string{term.Name, term.Value.String} {
				n, err := setupStringBound(s)
				if err != nil {
					return 0, err
				}
				if err := setupAdd(&total, n+1024); err != nil {
					return 0, err
				}
			}
		}
	}
	for _, action := range rule.RelationActions {
		if err := setupAdd(&total, int64(len(action.Relation))+32); err != nil {
			return 0, err
		}
		for _, term := range action.Terms {
			for _, s := range []string{term.Name, term.Value.String} {
				n, err := setupStringBound(s)
				if err != nil {
					return 0, err
				}
				if err := setupAdd(&total, n+1024); err != nil {
					return 0, err
				}
			}
		}
	}
	for _, count := range []int{len(rule.Conditions), len(rule.RelationConditions), len(rule.RelationActions)} {
		if err := bookkeepingSlots[string](scratch, count); err != nil {
			return 0, err
		}
	}
	if total > math.MaxInt64/8 {
		return 0, ErrMemoryBudgetExceeded
	}
	return 8 * total, nil
}

func reserveWorkloadKey(scratch *egraphcore.MemoryScope, b WorkloadBindings) error {
	if scratch == nil {
		return nil
	}
	for _, count := range []int{len(b.InputIdentities), len(b.WeightIdentities)} {
		if err := bookkeepingSlots[string](scratch, count); err != nil {
			return err
		}
	}
	if err := bookkeepingSlots[int](scratch, b.Shape.Rank()); err != nil {
		return err
	}
	var bytes int64 = 256
	for _, s := range []string{b.Name, b.Layout, string(b.NumericalPolicy), string(b.Objective)} {
		if err := setupAdd(&bytes, int64(len(s))); err != nil {
			return err
		}
	}
	for _, name := range b.InputNames {
		if err := setupAdd(&bytes, int64(len(name))); err != nil {
			return err
		}
	}
	for _, identities := range []map[string]string{b.InputIdentities, b.WeightIdentities} {
		for k, v := range identities {
			if err := setupAdd(&bytes, int64(len(k))); err != nil {
				return err
			}
			if err := setupAdd(&bytes, int64(len(v))); err != nil {
				return err
			}
		}
	}
	dims, err := memoryProduct(b.Shape.Rank(), 24)
	if err != nil || setupAdd(&bytes, dims) != nil || bytes > math.MaxInt64/2 {
		return ErrMemoryBudgetExceeded
	}
	if err := scratch.Reserve(2 * bytes); err != nil {
		return err
	}
	return nil
}

func reserveSearchPolicy(scratch, retained *egraphcore.MemoryScope, budget SearchBudget, rules []*egraph.Rewrite, eval Evaluator, work *workBudget) error {
	if scratch == nil {
		return nil
	}
	// The largest current branch has fifteen options. Reserve one map value
	// and sixteen aligned entry slots, all possible digests/number strings and
	// the known JSON error diagnostic before building the policy. These are
	// upper bounds on this fixed schema, not a limit on caller string lengths.
	if err := bookkeepingSlots[map[string]string](retained, 1); err != nil {
		return err
	}
	if err := bookkeepingSlots[struct{ key, value string }](retained, 16); err != nil {
		return err
	}
	if err := retained.Reserve(3*64 + 5*32 + 1024); err != nil {
		return err
	}
	encode := func(value any, keep bool) error {
		n, err := setupJSONBound(value, scratch)
		if err != nil {
			return err
		}
		if n > math.MaxInt64/2 {
			return ErrMemoryBudgetExceeded
		}
		if err := scratch.Reserve(2 * n); err != nil {
			return err
		}
		if keep {
			return retained.Reserve(n)
		}
		return nil
	}
	if err := bookkeepingSlots[SearchBudget](scratch, 1); err != nil {
		return err
	}
	if err := encode(budget, true); err != nil {
		return err
	}
	if err := bookkeepingSlots[string](scratch, len(rules)); err != nil {
		return err
	}
	var identityBytes int64 = 4
	for i, rule := range rules {
		if rule == nil || rule.LHS == nil || rule.RHS == nil {
			return fmt.Errorf("cudasearch: invalid rewrite at index %d", i)
		}
		n, err := setupRuleBytes(rule, scratch)
		if err != nil {
			return err
		}
		if err := scratch.Reserve(n); err != nil {
			return err
		}
		if n > (math.MaxInt64-4)/6 || setupAdd(&identityBytes, 6*n+4) != nil {
			return ErrMemoryBudgetExceeded
		}
	}
	if identityBytes > math.MaxInt64/2 {
		return ErrMemoryBudgetExceeded
	}
	if err := scratch.Reserve(2 * identityBytes); err != nil {
		return err
	}
	switch e := eval.(type) {
	case *MockEvaluator:
		if err := reserveWorkloadKey(scratch, e.Workload); err != nil {
			return err
		}
		if err := bookkeepingSlots[ScoringModel](scratch, 1); err != nil {
			return err
		}
		if err := encode(work.scoring.model, true); err != nil {
			return err
		}
		return encode(work.scoring.evidence, false)
	case *ExecutableEvaluator:
		if err := reserveWorkloadKey(scratch, e.Workload); err != nil {
			return err
		}
		type evaluation struct {
			Rounds         int
			Timing         string
			Policy         OutputPolicy
			CompileTimeout time.Duration
		}
		if err := bookkeepingSlots[evaluation](scratch, 1); err != nil {
			return err
		}
		return encode(evaluation{e.Rounds, string(e.TimingKind), e.Policy, e.CompileTimeout}, true)
	}
	return nil
}
