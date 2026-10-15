package egraph

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/egraphcore"
)

var errAllocationFixture = errors.New("allocation fixture refused")

type allocationFixture struct {
	live, peak    int64
	calls, failAt int
}

func (m *allocationFixture) Reserve(bytes int64) error {
	m.calls++
	if m.calls == m.failAt {
		return errAllocationFixture
	}
	m.live += bytes
	m.peak = max(m.peak, m.live)
	return nil
}
func (m *allocationFixture) Release(bytes int64) { m.live -= bytes }

func allocationGraph(t *testing.T) (*EGraph, []*Rewrite) {
	t.Helper()
	g := New()
	a, b := g.Add(NewNode("a")), g.Add(NewNode("b"))
	g.Add(NewNode("pair", a, b))
	g.Add(NewNode("pair", b, a))
	if err := g.RebuildChecked(); err != nil {
		t.Fatal(err)
	}
	rules := []*Rewrite{
		{Name: "swap", LHS: Pat("pair", Var("x"), Var("y")), RHS: Pat("pair", Var("y"), Var("x"))},
		{Name: "first", LHS: Pat("pair", Var("x"), Var("y")), RHS: Var("x")},
	}
	return g, rules
}

func TestAllocationActionKeyIdentity(t *testing.T) {
	values := []egraphcore.Value[Id]{egraphcore.Class(Id(math.MaxUint32)), egraphcore.Int64[Id](math.MinInt64), egraphcore.Bool[Id](true), egraphcore.Float64[Id](math.NaN()), egraphcore.String[Id]("\x00\n\"世界"), egraphcore.Unit[Id](), egraphcore.BigInt[Id]("12345678901234567890"), egraphcore.BigRat[Id]("2/3")}
	for _, value := range values {
		memory := new(allocationFixture)
		scope, err := egraphcore.NewMemoryScope(memory)
		if err != nil {
			t.Fatal(err)
		}
		budget, err := egraphcore.NewWorkBudgetWithMemory(100000, scope)
		if err != nil {
			t.Fatal(err)
		}
		candidate := ActionCandidate{Rule: &Rewrite{Name: "quoted \"rule"}, Match: MatchMatch{ClassID: math.MaxUint32, Subst: Subst{"?z": 9, "?a": 2}, Binding: egraphcore.Binding[Id]{"?fact": value}}}
		got, err := actionKeyWithBudget(candidate, budget)
		if err != nil || got != candidate.ActionKey() {
			t.Fatalf("%#v: key=%q want=%q err=%v", value, got, candidate.ActionKey(), err)
		}
		scope.Close()
		if memory.live != 0 {
			t.Fatal(memory.live)
		}
	}
}

func TestAllocationEnumerationEveryReservation(t *testing.T) {
	for _, conditioned := range []bool{false, true} {
		t.Run(fmt.Sprint(conditioned), func(t *testing.T) {
			g, rules := allocationGraph(t)
			if conditioned {
				relation, err := g.Facts().Register(egraphcore.Schema{Name: "selected", Columns: []egraphcore.Column{{Kind: egraphcore.ClassValue}, {Kind: egraphcore.StringValue}}})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := relation.Add([]egraphcore.Value[Id]{egraphcore.Class(Id(0)), egraphcore.String[Id]("yes")}, g.Find); err != nil {
					t.Fatal(err)
				}
				rules = rules[:1]
				rules[0].Conditions = []*Pattern{Pat("a")}
				rules[0].RelationConditions = []RelationCondition{{Name: "selected", Terms: []egraphcore.Term[Id]{egraphcore.Variable[Id]("?x"), egraphcore.Variable[Id]("?selected")}}}
			}
			want, _, err := EnumerateActionsBoundedChecked(g, rules, 20)
			if err != nil {
				t.Fatal(err)
			}
			if len(want) == 0 {
				t.Fatal("fixture has no actions")
			}
			facts, revision := g.facts, g.revision
			run := func(memory *allocationFixture) error {
				owner, err := EnumerateActionsWithMemory(g, rules, 20, memory)
				if err != nil {
					if owner != nil {
						t.Fatal("partial result on refusal")
					}
					return err
				}
				defer owner.Close()
				if owner.Truncated || len(owner.Actions) != len(want) || memory.live == 0 {
					t.Fatal("missing actions or owner", owner, memory.live)
				}
				for i := range want {
					if owner.Actions[i].ActionKey() != want[i].ActionKey() {
						t.Fatal(owner.Actions[i].ActionKey(), want[i].ActionKey())
					}
				}
				return nil
			}
			baseline := new(allocationFixture)
			if err := run(baseline); err != nil {
				t.Fatal(err)
			}
			for fail := 1; fail <= baseline.calls; fail++ {
				memory := &allocationFixture{failAt: fail}
				if err := run(memory); !errors.Is(err, errAllocationFixture) {
					t.Fatalf("reservation %d/%d: %v", fail, baseline.calls, err)
				}
				if memory.live != 0 {
					t.Fatalf("reservation %d leaked %d", fail, memory.live)
				}
				if g.facts != facts || g.revision != revision {
					t.Fatal("source changed on refusal")
				}
			}
			if baseline.live != 0 || g.facts != facts || g.revision != revision {
				t.Fatal("source or reservations changed")
			}
		})
	}
}

func TestAllocationPendingAndPayload(t *testing.T) {
	g, rules := allocationGraph(t)
	g.Union(0, 1)
	owner, err := EnumerateActionsWithMemory(g, rules, 5, new(allocationFixture))
	if owner != nil || !errors.Is(err, ErrPendingGraphChanges) {
		t.Fatal(owner, err)
	}
	if err := g.RebuildChecked(); err != nil {
		t.Fatal(err)
	}
	g.Add(ENode{Op: "payload", Data: []int{1}})
	if err := g.RebuildChecked(); err != nil {
		t.Fatal(err)
	}
	rules[0].LHS = &Pattern{Op: "payload", HasData: true, Data: []int{1}}
	owner, err = EnumerateActionsWithMemory(g, rules[:1], 5, new(allocationFixture))
	if owner != nil || !errors.Is(err, ErrMemoryOperation) {
		t.Fatal(owner, err)
	}
	// The existing unlimited data comparator still accepts reference payloads.
	matches, _, err := rules[0].LHS.SearchBoundedChecked(g, 5)
	if err != nil || len(matches) != 1 {
		t.Fatal("unlimited payload matching changed", matches, err)
	}
}

func TestAllocationCanonicalFactView(t *testing.T) {
	g := New()
	a, b := g.Add(NewNode("a")), g.Add(NewNode("b"))
	g.Union(a, b)
	if err := g.RebuildChecked(); err != nil {
		t.Fatal(err)
	}
	a = g.Find(a)
	if b == a {
		t.Fatal("fixture needs a noncanonical class")
	}
	relation, err := g.Facts().Register(egraphcore.Schema{Name: "selected", Columns: []egraphcore.Column{{Kind: egraphcore.ClassValue}}})
	if err != nil {
		t.Fatal(err)
	}
	// Public facts can be inserted after rebuild with a stale representative.
	// The readonly matching view must canonicalize its own index and rows.
	identity := func(id Id) Id { return id }
	if _, err := relation.Add([]egraphcore.Value[Id]{egraphcore.Class(b)}, identity); err != nil {
		t.Fatal(err)
	}
	cursor, facts := relation.Snapshot(), g.facts
	rule := &Rewrite{Name: "selected", LHS: Var("x"), RHS: Pat("replacement"), RelationConditions: []RelationCondition{{Name: "selected", Terms: []egraphcore.Term[Id]{egraphcore.Constant(egraphcore.Class(a))}}}}
	memory := new(allocationFixture)
	owner, err := EnumerateActionsWithMemory(g, []*Rewrite{rule}, 5, memory)
	if err != nil {
		t.Fatal(err)
	}
	defer owner.Close()
	if len(owner.Actions) != 1 || owner.Actions[0].Match.ClassID != a || owner.Actions[0].Match.Binding["?x"].Class != a {
		t.Fatal(owner.Actions)
	}
	rows, complete, err := relation.Query(nil, identity, nil)
	if err != nil || !complete || len(rows) != 1 || rows[0][0].Class != b || g.facts != facts || relation.Snapshot() != cursor {
		t.Fatal("source facts changed", rows, complete, err)
	}
}

func ExampleEnumerateActionsWithMemory() {
	g := New()
	g.Add(NewNode("a"))
	_ = g.RebuildChecked()
	rule := &Rewrite{Name: "replace", LHS: Pat("a"), RHS: Pat("b")}
	memory := new(allocationFixture)
	owner, err := EnumerateActionsWithMemory(g, []*Rewrite{rule}, 10, memory)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(owner.Actions), owner.Truncated, memory.live > 0)
	owner.Close()
	fmt.Println(memory.live)
	// Output:
	// 1 false true
	// 0
}
