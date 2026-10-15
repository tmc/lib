package egraphcore

import (
	"errors"
	"reflect"
	"testing"
)

var errCloneReservation = errors.New("clone reservation refused")

type cloneMemory struct {
	live        int64
	calls, fail int
}

func (m *cloneMemory) Reserve(n int64) error {
	m.calls++
	if m.calls == m.fail {
		return errCloneReservation
	}
	m.live += n
	return nil
}
func (m *cloneMemory) Release(n int64) { m.live -= n }

func TestCloneMemoryGuardCharges(t *testing.T) {
	m := new(cloneMemory)
	scope, _ := NewMemoryScope(m)
	defer scope.Close()
	budget, _ := NewWorkBudgetWithMemory(0, scope)
	before := m.live
	source := NewStore[uint32, int]()
	source.Insert("a", func(uint32) int { return 7 })
	copy, err := source.CloneWithBudget(nil, budget)
	if err != nil || copy.Class(0) != 7 {
		t.Fatal(copy, err)
	}
	want := int64(reflect.TypeFor[Store[uint32, int]]().Size() + reflect.TypeFor[uint32]().Size()*2 + reflect.TypeFor[uint64]().Size() + reflect.TypeFor[map[uint32]int]().Size() + reflect.TypeFor[map[string][]uint32]().Size() + 2*reflect.TypeFor[struct {
		Key   uint32
		Value int
	}]().Size() + 2*reflect.TypeFor[struct {
		Key   string
		Value []uint32
	}]().Size())
	if m.live-before != want {
		t.Fatalf("store charge=%d want=%d", m.live-before, want)
	}
	index := NewValueIndex[uint32](1)
	index.Add([]Value[uint32]{Int64[uint32](3)}, "one")
	index.Add([]Value[uint32]{Int64[uint32](3)}, "two")
	before = m.live
	cloned, err := index.cloneWithBudget(budget)
	if err != nil || !reflect.DeepEqual(cloned, index) {
		t.Fatal(cloned, err)
	}
	want = int64(reflect.TypeFor[ValueIndex[uint32]]().Size() + 2*reflect.TypeFor[map[Value[uint32]]map[string]struct{}]().Size() + reflect.TypeFor[map[string]struct{}]().Size() + 2*reflect.TypeFor[struct {
		Key   Value[uint32]
		Value map[string]struct{}
	}]().Size() + 4*reflect.TypeFor[struct {
		Key   string
		Value struct{}
	}]().Size())
	if m.live-before != want {
		t.Fatalf("index charge=%d want=%d", m.live-before, want)
	}
}

func TestCloneMemoryCallbackOrder(t *testing.T) {
	source := NewStore[uint32, int]()
	source.Insert("a", func(uint32) int { return 7 })
	// Scope, WorkBudget, two arrays, two map headers plus hinted slots,
	// then the Store object all precede the backend callback.
	for fail := 1; fail <= 9; fail++ {
		m := &cloneMemory{fail: fail}
		scope, err := NewMemoryScope(m)
		if err != nil {
			continue
		}
		budget, err := NewWorkBudgetWithMemory(0, scope)
		calls := 0
		if err == nil {
			_, err = source.CloneWithBudget(func(n int, _ *WorkBudget) (int, error) { calls++; return n, nil }, budget)
		}
		scope.Close()
		if !errors.Is(err, errCloneReservation) || calls != 0 || m.live != 0 {
			t.Fatalf("guard %d callback=%d live=%d err=%v", fail, calls, m.live, err)
		}
	}
}

func TestCloneMemoryEveryReservation(t *testing.T) {
	store := NewStore[uint32, int]()
	store.Insert("a", func(uint32) int { return 7 })
	store.Insert("a", func(uint32) int { return 9 })
	facts := NewFactStore[uint32]()
	relation, _ := facts.Register(Schema{Name: "r", Columns: []Column{{Kind: Int64Value}, {Kind: StringValue}}})
	relation.Add([]Value[uint32]{Int64[uint32](1), String[uint32]("one")}, nil)
	relation.MarkSubsumed([]Value[uint32]{Int64[uint32](2), String[uint32]("two")}, nil)
	cursor := relation.Snapshot()
	relation.Add([]Value[uint32]{Int64[uint32](3), String[uint32]("three")}, nil)
	for _, kind := range []string{"store", "facts"} {
		t.Run(kind, func(t *testing.T) {
			run := func(m *cloneMemory) error {
				scope, err := NewMemoryScope(m)
				if err != nil {
					return err
				}
				defer scope.Close()
				budget, err := NewWorkBudgetWithMemory(0, scope)
				if err != nil {
					return err
				}
				if kind == "store" {
					copy, err := store.CloneWithBudget(func(n int, _ *WorkBudget) (int, error) { return n, nil }, budget)
					if err != nil {
						if copy != nil {
							t.Fatal("partial store")
						}
						return err
					}
					if !reflect.DeepEqual(copy, store.Clone(nil)) {
						t.Fatal("store differs")
					}
					copy.parent[1] = 0
					copy.memo["a"][0] = 1
					copy.classes[0] = 99
					if store.parent[1] != 1 || store.memo["a"][0] != 0 || store.Class(0) != 7 {
						t.Fatal("store aliases source")
					}
				} else {
					copy, err := facts.CloneWithBudget(budget)
					if err != nil {
						if copy != nil {
							t.Fatal("partial facts")
						}
						return err
					}
					got := copy.Get("r")
					want := relation.Clone()
					if !reflect.DeepEqual(got.Rows(), want.Rows()) || !reflect.DeepEqual(got.index, want.index) || got.version != want.version || got.floor != want.floor || !reflect.DeepEqual(got.added, want.added) {
						t.Fatal("relation differs")
					}
					rows, err := got.AddedSince(cursor)
					if err != nil || len(rows) != 1 || rows[0][0].Int64 != 3 {
						t.Fatal(rows, err)
					}
					rows, complete, err := got.Query([]Filter[uint32]{{Column: 0, Value: Int64[uint32](2)}}, nil, nil)
					if err != nil || !complete || len(rows) != 0 {
						t.Fatal("subsumed fact visible", rows, err)
					}
					got.Add([]Value[uint32]{Int64[uint32](4), String[uint32]("four")}, nil)
					if relation.Len() != 3 {
						t.Fatal("facts alias source")
					}
				}
				return nil
			}
			baseline := new(cloneMemory)
			if err := run(baseline); err != nil {
				t.Fatal(err)
			}
			for fail := 1; fail <= baseline.calls; fail++ {
				m := &cloneMemory{fail: fail}
				if err := run(m); !errors.Is(err, errCloneReservation) || m.live != 0 {
					t.Fatalf("guard %d/%d: live=%d err=%v", fail, baseline.calls, m.live, err)
				}
			}
			if baseline.live != 0 {
				t.Fatal("leak", baseline.live)
			}
		})
	}
}

func TestCloneMemoryLineage(t *testing.T) {
	facts := NewFactStore[uint32]()
	r, _ := facts.Register(Schema{Name: "r", Columns: []Column{{Kind: Int64Value}}})
	r.Add([]Value[uint32]{Int64[uint32](1)}, nil)
	ancestor := r.Snapshot()
	sibling := facts.Clone().Get("r")
	sibling.Add([]Value[uint32]{Int64[uint32](2)}, nil)
	siblingCursor := sibling.Snapshot()
	for i := 0; i < 18; i++ {
		scope, _ := NewMemoryScope(new(cloneMemory))
		defer scope.Close()
		budget, _ := NewWorkBudgetWithMemory(0, scope)
		copy, err := facts.CloneWithBudget(budget)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := copy.Get("r").AddedSince(siblingCursor); !errors.Is(err, ErrRelationReset) {
			t.Fatal("sibling cursor accepted", err)
		}
		_, err = copy.Get("r").AddedSince(ancestor)
		if (i < 16 && err != nil) || (i >= 16 && !errors.Is(err, ErrRelationReset)) {
			t.Fatalf("clone %d: %v", i, err)
		}
		facts = copy
	}
}
