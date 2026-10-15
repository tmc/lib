package egraph

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"sort"
	"strconv"

	"github.com/tmc/lib/egraphcore"
)

// ErrMemoryOperation reports a bounded operation whose storage cannot yet be
// reserved before allocation. Unlimited operations retain their usual behavior.
var ErrMemoryOperation = errors.New("egraph: operation lacks memory accounting")

func graphSlots[T any](budget *egraphcore.WorkBudget, count int) error {
	size := uint64(reflect.TypeFor[T]().Size())
	if count < 0 || size != 0 && uint64(count) > math.MaxInt64/size {
		return egraphcore.ErrMemorySize
	}
	if !budget.ReserveMemory(int64(uint64(count) * size)) {
		return budget.MemoryError()
	}
	return nil
}

func graphSlice[T any](budget *egraphcore.WorkBudget, length, capacity int) ([]T, error) {
	if length < 0 || capacity < length {
		return nil, egraphcore.ErrMemorySize
	}
	if err := graphSlots[T](budget, capacity); err != nil {
		return nil, err
	}
	return make([]T, length, capacity), nil
}

func graphAppend[T any](budget *egraphcore.WorkBudget, values []T, added ...T) ([]T, error) {
	if len(added) > math.MaxInt-len(values) {
		return nil, egraphcore.ErrMemorySize
	}
	length := len(values) + len(added)
	if length > cap(values) {
		capacity := cap(values)
		if capacity <= math.MaxInt/2 {
			capacity *= 2
		}
		if capacity < length {
			capacity = length
		}
		next, err := graphSlice[T](budget, len(values), capacity)
		if err != nil {
			return nil, err
		}
		copy(next, values)
		values = next
	}
	return append(values, added...), nil
}

func graphMap[K comparable, V any](budget *egraphcore.WorkBudget, hint int) (map[K]V, error) {
	if err := graphSlots[map[K]V](budget, 1); err != nil {
		return nil, err
	}
	if err := graphSlots[struct {
		key   K
		value V
	}](budget, hint); err != nil {
		return nil, err
	}
	return make(map[K]V, hint), nil
}

func graphPut[K comparable, V any](budget *egraphcore.WorkBudget, values map[K]V, key K, value V) error {
	if _, exists := values[key]; !exists {
		if err := graphSlots[struct {
			key   K
			value V
		}](budget, 1); err != nil {
			return err
		}
	}
	values[key] = value
	return nil
}

// matchingView borrows graph structure and owns a canonical fact copy. Bounded
// readers cannot rebuild a caller-owned graph: rebuilt storage would outlive
// the operation's reservations. Repeated searches share this view and scope.
func matchingView(g *EGraph, budget *egraphcore.WorkBudget) (*EGraph, error) {
	if budget.MemoryScope() == nil {
		return g, g.RebuildChecked()
	}
	if g == nil || g.store == nil || !g.store.Initialized() {
		return nil, fmt.Errorf("egraph: uninitialized graph")
	}
	if len(g.worklist) != 0 {
		return nil, ErrPendingGraphChanges
	}
	if g.memoryWork == budget {
		return g, nil
	}
	if err := graphSlots[EGraph](budget, 1); err != nil {
		return nil, err
	}
	view := new(EGraph)
	*view = *g
	facts, err := g.facts.CanonicalCopyWithBudget(g.Find, budget)
	if err != nil {
		if errors.Is(err, egraphcore.ErrMatchWorkLimit) {
			return nil, matchingError(budget)
		}
		return nil, err
	}
	view.facts, view.memoryWork = facts, budget
	return view, nil
}

func matchingRoots(g *EGraph, budget *egraphcore.WorkBudget) ([]Id, error) {
	roots, err := graphSlice[Id](budget, 0, g.storage().ClassCount())
	if err != nil {
		return nil, err
	}
	g.storage().RangeClasses(func(id Id, _ *EClass) bool {
		roots = append(roots, id)
		return true
	})
	slices.Sort(roots)
	return roots, nil
}

func matchingError(budget *egraphcore.WorkBudget) error {
	if err := budget.MemoryError(); err != nil {
		return err
	}
	return ErrMatchWorkLimit
}

func lookupNodeKey(node ENode, budget *egraphcore.WorkBudget) (string, error) {
	if budget.MemoryScope() == nil {
		return node.Key(), nil
	}
	if node.Data == nil && len(node.Children) == 0 {
		return node.Op, nil
	}
	bound := int64(len(node.Op)) + 2
	if len(node.Children) > math.MaxInt64/11 {
		return "", egraphcore.ErrMemorySize
	}
	if int64(len(node.Children))*11 > math.MaxInt64-bound {
		return "", egraphcore.ErrMemorySize
	}
	bound += int64(len(node.Children)) * 11
	if node.Data != nil {
		value, ok := node.Data.(string)
		if !ok {
			return "", ErrMemoryOperation
		}
		if int64(len(value)) > (math.MaxInt64-bound-11)/6 {
			return "", egraphcore.ErrMemorySize
		}
		bound += 11 + 6*int64(len(value))
	}
	if bound > int64(math.MaxInt) {
		return "", egraphcore.ErrMemorySize
	}
	data, err := graphSlice[byte](budget, 0, int(bound))
	if err != nil {
		return "", err
	}
	data = append(data, node.Op...)
	if node.Data != nil {
		data = append(data, "{string:"...)
		data = strconv.AppendQuote(data, node.Data.(string))
		data = append(data, '}')
	}
	if len(node.Children) != 0 {
		data = append(data, '(')
		for i, child := range node.Children {
			if i > 0 {
				data = append(data, ',')
			}
			data = strconv.AppendUint(data, uint64(child), 10)
		}
		data = append(data, ')')
	}
	if err := graphSlots[byte](budget, len(data)); err != nil {
		return "", err
	}
	return string(data), nil
}

func dataComparisonAllocates(t reflect.Type) bool {
	if t == nil {
		return false
	}
	switch t.Kind() {
	case reflect.Array:
		return dataComparisonAllocates(t.Elem())
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if dataComparisonAllocates(t.Field(i).Type) {
				return true
			}
		}
	case reflect.Interface, reflect.Pointer, reflect.Map, reflect.Slice:
		return true
	}
	return false
}

// actionKeyWithBudget uses the same representation as ActionKey, without
// repeatedly formatting values in sort comparisons. Value has only primitive
// fields, so its complete Go representation has a checked finite bound.
func actionKeyWithBudget(c ActionCandidate, budget *egraphcore.WorkBudget) (string, error) {
	if budget.MemoryScope() == nil {
		return c.ActionKey(), nil
	}
	if c.Rule == nil {
		return "", nil
	}
	typeName := reflect.TypeFor[egraphcore.Value[Id]]().String()
	bound := int64(len(c.Rule.Name)) + 32
	add := func(n int64) bool {
		if n < 0 || n > math.MaxInt64-bound {
			return false
		}
		bound += n
		return true
	}
	for name := range c.Match.Subst {
		if !add(int64(len(name)) + 12) {
			return "", egraphcore.ErrMemorySize
		}
	}
	binding := c.Match.Binding
	if binding == nil {
		var err error
		binding, err = graphMap[string, egraphcore.Value[Id]](budget, len(c.Match.Subst))
		if err != nil {
			return "", err
		}
		for name, id := range c.Match.Subst {
			if err := graphPut(budget, binding, name, egraphcore.Class(id)); err != nil {
				return "", err
			}
		}
	}
	for name, value := range binding {
		if len(value.String) > math.MaxInt64/6 || !add(int64(len(name))+int64(len(typeName))+256+6*int64(len(value.String))) {
			return "", egraphcore.ErrMemorySize
		}
	}
	if bound > int64(math.MaxInt) {
		return "", egraphcore.ErrMemorySize
	}
	data, err := graphSlice[byte](budget, 0, int(bound))
	if err != nil {
		return "", err
	}
	count := max(len(c.Match.Subst), len(binding))
	keys, err := graphSlice[string](budget, 0, count)
	if err != nil {
		return "", err
	}
	data = append(data, c.Rule.Name...)
	data = append(data, '@')
	data = strconv.AppendUint(data, uint64(c.Match.ClassID), 10)
	data = append(data, '[')
	for key := range c.Match.Subst {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for i, key := range keys {
		if i > 0 {
			data = append(data, ',')
		}
		data = append(data, key...)
		data = append(data, ':')
		data = strconv.AppendUint(data, uint64(c.Match.Subst[key]), 10)
	}
	keys = keys[:0]
	for key := range binding {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		v := binding[key]
		data = append(data, '|')
		data = append(data, key...)
		data = append(data, ':')
		data = append(data, typeName...)
		data = append(data, "{Kind:0x"...)
		data = strconv.AppendUint(data, uint64(v.Kind), 16)
		data = append(data, ", Class:0x"...)
		data = strconv.AppendUint(data, uint64(v.Class), 16)
		data = append(data, ", Int64:"...)
		data = strconv.AppendInt(data, v.Int64, 10)
		data = append(data, ", Bool:"...)
		data = strconv.AppendBool(data, v.Bool)
		data = append(data, ", Float:0x"...)
		data = strconv.AppendUint(data, v.Float, 16)
		data = append(data, ", String:"...)
		data = strconv.AppendQuote(data, v.String)
		data = append(data, '}')
	}
	data = append(data, ']')
	if err := graphSlots[byte](budget, len(data)); err != nil {
		return "", err
	}
	return string(data), nil
}
