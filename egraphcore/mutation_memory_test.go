package egraphcore

import (
	"errors"
	"reflect"
	"testing"
)

func mutationSweep(t *testing.T, run func(*WorkBudget) error) {
	t.Helper()
	trial := func(fail int) (int, error) {
		m := &cloneMemory{fail: fail}
		scope, err := NewMemoryScope(m)
		if err == nil {
			var budget *WorkBudget
			budget, err = NewWorkBudgetWithMemory(0, scope)
			if err == nil {
				err = run(budget)
			}
			scope.Close()
			scope.Close()
		}
		if m.live != 0 {
			t.Fatalf("reservation %d leaked %d", fail, m.live)
		}
		return m.calls, err
	}
	calls, err := trial(0)
	if err != nil {
		t.Fatal(err)
	}
	for fail := 1; fail <= calls; fail++ {
		if _, err := trial(fail); !errors.Is(err, errCloneReservation) {
			t.Fatalf("reservation %d/%d: %v", fail, calls, err)
		}
	}
}

func TestMutationMemoryStore(t *testing.T) {
	mutationSweep(t, func(b *WorkBudget) error {
		s := NewStore[uint32, int]()
		s.Insert("same", func(uint32) int { return 7 })
		before := s.Clone(nil)
		id, err := s.InsertWithBudget("same", func(id uint32) (int, error) { return int(id) + 10, nil }, b)
		if err != nil {
			if !reflect.DeepEqual(s, before) {
				t.Fatal("partial store publication")
			}
			return err
		}
		if id != 1 || s.Class(id) != 11 || len(s.memo["same"]) != 2 {
			t.Fatal("insertion lost", s, id)
		}
		_, err = s.AddMemoCandidateIfWithBudget("other", id, func(a, c uint32) bool { return a == c }, b)
		return err
	})
}

func TestMutationMemoryMemoSpill(t *testing.T) {
	mutationSweep(t, func(b *WorkBudget) error {
		id, found, err := FindMemoCandidateIterWithBudget(func(visit func(uint32) bool) {
			for id := uint32(0); id < 10; id++ {
				if !visit(id) {
					return
				}
			}
		}, 99, func(id uint32) uint32 { return id }, func(id uint32) []int { return nil }, func(n int) (int, error) { return n, nil }, func(a, c int) bool { return a == c }, b)
		if err == nil && (found || id != 0) {
			t.Fatal(id, found)
		}
		return err
	})
	m := new(cloneMemory)
	scope, _ := NewMemoryScope(m)
	defer scope.Close()
	b, _ := NewWorkBudgetWithMemory(0, scope)
	before := m.calls
	_, _, err := FindMemoCandidateIterWithBudget(func(visit func(uint32) bool) {
		for id := uint32(0); id < 9; id++ {
			if !visit(id) {
				return
			}
		}
	}, 0, func(id uint32) uint32 { return id }, func(uint32) []int { return nil }, func(n int) (int, error) { return n, nil }, func(a, c int) bool { return a == c }, b)
	if err != nil || m.calls <= before {
		t.Fatal("ninth owner did not admit spill storage", err, m.calls, before)
	}
}

func mutationFacts(t *testing.T) (*FactStore[uint32], *Relation[uint32]) {
	t.Helper()
	s := NewFactStore[uint32]()
	r, err := s.Register(Schema{Name: "r", Columns: []Column{{Kind: Int64Value}, {Kind: StringValue}}})
	if err != nil {
		t.Fatal(err)
	}
	r.Add([]Value[uint32]{Int64[uint32](1), String[uint32]("one")}, nil)
	return s, r
}

func TestMutationMemoryFacts(t *testing.T) {
	for _, kind := range []FactActionKind{FactInsert, FactDelete, FactSubsume} {
		t.Run(map[FactActionKind]string{FactInsert: "insert", FactDelete: "delete", FactSubsume: "subsume"}[kind], func(t *testing.T) {
			mutationSweep(t, func(b *WorkBudget) error {
				s, r := mutationFacts(t)
				rows := r.Rows()
				cursor := r.Snapshot()
				ancestors := append([]relationAncestor(nil), r.ancestors...)
				values := []Value[uint32]{Int64[uint32](2), String[uint32]("new\x00\"value")}
				if kind != FactInsert {
					values = rows[0]
				}
				flags, changes, err := s.ApplyActionsDetailed([]ResolvedFactAction[uint32]{{Kind: kind, Relation: "r", Values: values}}, nil, b)
				if err != nil {
					if flags != nil || changes != 0 || !reflect.DeepEqual(r.Rows(), rows) || r.Snapshot() != cursor {
						t.Fatal("partial fact publication")
					}
					return err
				}
				if s.Get("r") != r || len(flags) != 1 || !flags[0] || changes != 1 {
					t.Fatal(flags, changes)
				}
				if kind == FactInsert {
					added, err := r.AddedSince(cursor)
					if err != nil || len(added) != 1 || !reflect.DeepEqual(r.ancestors, ancestors) {
						t.Fatal("singleton insert changed cursor lineage", added, err)
					}
				} else {
					if _, err := r.AddedSince(cursor); !errors.Is(err, ErrRelationReset) {
						t.Fatal("cursor not reset", err)
					}
				}
				return nil
			})
		})
	}
	mutationSweep(t, func(b *WorkBudget) error {
		s, r := mutationFacts(t)
		cursor := r.Snapshot()
		before := r.Rows()
		actions := []ResolvedFactAction[uint32]{{Kind: FactInsert, Relation: "r", Values: before[0]}}
		flags, changed, err := s.ApplyActionsDetailed(actions, nil, b)
		if err != nil {
			return err
		}
		if len(flags) != 1 || flags[0] || changed != 0 || r.Snapshot() != cursor {
			t.Fatal("no-op changed publication")
		}
		return nil
	})
}

func TestMutationMemoryCanonicalFacts(t *testing.T) {
	mutationSweep(t, func(b *WorkBudget) error {
		s := NewFactStore[uint32]()
		r, _ := s.Register(Schema{Name: "classes", Columns: []Column{{Kind: ClassValue}}})
		other, _ := s.Register(Schema{Name: "other", Columns: []Column{{Kind: Int64Value}}})
		r.Add([]Value[uint32]{Class[uint32](1)}, func(id uint32) uint32 { return id })
		r.MarkSubsumed([]Value[uint32]{Class[uint32](2)}, func(id uint32) uint32 { return id })
		other.Add([]Value[uint32]{Int64[uint32](5)}, nil)
		cursor, otherCursor := r.Snapshot(), other.Snapshot()
		rows := r.Rows()
		err := s.RecanonicalizeWithBudget(func(uint32) uint32 { return 0 }, b)
		if err != nil {
			if !reflect.DeepEqual(rows, r.Rows()) || r.Snapshot() != cursor || other.Snapshot() != otherCursor {
				t.Fatal("partial recanonicalization")
			}
			return err
		}
		if s.Get("classes") != r || r.Len() != 1 || len(r.subsumed) != 1 || r.version != 0 || len(r.added) != 0 {
			t.Fatal("canonical facts differ", r)
		}
		if _, err := r.AddedSince(cursor); !errors.Is(err, ErrRelationReset) {
			t.Fatal(err)
		}
		return nil
	})
}

func TestMutationMemoryHistoryAndBatch(t *testing.T) {
	for _, overflow := range []bool{false, true} {
		mutationSweep(t, func(b *WorkBudget) error {
			s, r := mutationFacts(t)
			if overflow {
				r.version = ^uint64(0)
			}
			cursor := r.Snapshot()
			rows := r.Rows()
			actions := []ResolvedFactAction[uint32]{{Kind: FactInsert, Relation: "r", Values: []Value[uint32]{Int64[uint32](2), String[uint32]("two")}}}
			if !overflow {
				actions = append(actions, ResolvedFactAction[uint32]{Kind: FactSubsume, Relation: "r", Values: rows[0]})
			}
			_, _, err := s.ApplyActionsDetailed(actions, nil, b)
			if err != nil {
				if r.Snapshot() != cursor || !reflect.DeepEqual(r.Rows(), rows) {
					t.Fatal("batch changed source")
				}
				return err
			}
			if r.Len() != 2 {
				t.Fatal(r.Len())
			}
			return nil
		})
	}
}

type mutationGraph struct{ parent [3]uint32 }

func (g *mutationGraph) Find(id uint32) uint32 {
	for g.parent[id] != id {
		id = g.parent[id]
	}
	return id
}
func (g *mutationGraph) AddChecked(int) (uint32, error) { return 0, nil }
func (g *mutationGraph) UnionChecked(a, c uint32) (uint32, bool, error) {
	a, c = g.Find(a), g.Find(c)
	if a == c {
		return a, false, nil
	}
	g.parent[c] = a
	return a, true, nil
}
func (g *mutationGraph) RebuildChecked() error { return nil }

func TestMutationMemoryCongruence(t *testing.T) {
	mutationSweep(t, func(b *WorkBudget) error {
		g := &mutationGraph{parent: [3]uint32{0, 1, 2}}
		owned := [3]OwnedNode[uint32, int]{{0, 7}, {1, 7}, {2, 8}}
		report, err := DriveCongruenceWithBudget(g, 3, func(pass, index int) (int, []OwnedNode[uint32, int], bool, error) {
			if index > 0 {
				return 0, nil, false, nil
			}
			for i := range owned {
				owned[i].Owner = g.Find(uint32(i))
			}
			return 0, owned[:], true, nil
		}, func(n int) (int, error) { return n, nil }, func(a, c int) bool { return a == c }, nil, func(_ int, groups map[uint32][]int) (bool, error) {
			if len(groups) != 2 {
				t.Fatal(groups)
			}
			return false, nil
		}, b)
		if err == nil && (report.Merges != 1 || report.Passes != 2 || g.Find(0) != g.Find(1)) {
			t.Fatal(report, g)
		}
		return err
	})
}
