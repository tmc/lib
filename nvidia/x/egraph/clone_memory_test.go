package egraph

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func cloneFixture() *SearchState {
	g := New()
	p := &[]int{1, 2}
	id := g.Add(ENode{Op: "payload", Data: map[string]any{"a": p, "b": p}})
	g.Add(ENode{Op: "parent", Children: []Id{id}, Data: map[string]any{"a": p, "b": p}})
	s := NewSearchState(g, map[string]Id{"root": id})
	s.Derivation = []Action{{RuleName: "before", Subst: Subst{"?x": id}, Binding: egraphcore.Binding[Id]{"?n": egraphcore.Int64[Id](3)}}}
	shape, _ := cudashape.New(2, 3)
	s.Metadata[id] = TypedMetadata{Shape: shape, Alignment: 16, NumericalPolicy: "fixture"}
	relation, _ := s.Graph.Facts().Register(egraphcore.Schema{Name: "kept", Columns: []egraphcore.Column{{Kind: egraphcore.Int64Value}}})
	relation.Add([]egraphcore.Value[Id]{egraphcore.Int64[Id](3)}, nil)
	return s
}

func TestCloneMemoryEveryReservation(t *testing.T) {
	source := cloneFixture()
	before := source.Clone()
	cursor := source.Graph.Facts().Get("kept").Snapshot()
	run := func(m *allocationFixture) error {
		scope, err := egraphcore.NewMemoryScope(m)
		if err != nil {
			return err
		}
		defer scope.Close()
		budget, err := egraphcore.NewWorkBudgetWithMemory(0, scope)
		if err != nil {
			return err
		}
		copy, err := source.CloneWithBudget(context.Background(), budget)
		if err != nil {
			if copy != nil {
				t.Fatal("partial state")
			}
			return err
		}
		if !reflect.DeepEqual(copy, before) {
			t.Fatal("clone differs from ordinary copy")
		}
		if rows, err := copy.Graph.Facts().Get("kept").AddedSince(cursor); err != nil || len(rows) != 0 {
			t.Fatal("clone lost source cursor", rows, err)
		}
		data := copy.Graph.class(copy.Roots["root"]).Nodes[0].Data.(map[string]any)
		if data["a"] != data["b"] {
			t.Fatal("payload aliases lost")
		}
		(*data["a"].(*[]int))[0] = 99
		parentOccurrences := 0
		for _, class := range copy.Graph.classes() {
			for i, parent := range class.parents {
				if other, ok := parent.node.Data.(map[string]any); ok && (*other["a"].(*[]int))[0] != 1 {
					t.Fatal("node and parent payload occurrences alias")
				}
				if parent.node.Op == "parent" {
					parentOccurrences++
					class.parents[i].node.Children[0] = 43
				}
			}
			for i, node := range class.Nodes {
				if node.Op == "parent" {
					if node.Children[0] == 43 {
						t.Fatal("node and parent child arrays alias")
					}
					class.Nodes[i].Children[0] = 42
				}
			}
		}
		if parentOccurrences != 1 {
			t.Fatal("missing parent copy fixture", parentOccurrences)
		}
		copy.Roots["other"] = 10
		copy.Derivation[0].Subst["?x"] = 42
		copy.Derivation[0].Binding["?n"] = egraphcore.Int64[Id](4)
		metadata := copy.Metadata[copy.Roots["root"]]
		dims := metadata.Shape.Dims()
		dims[0] = 99
		if source.Metadata[source.Roots["root"]].Shape.Dim(0) != 2 {
			t.Fatal("shape access mutated metadata")
		}
		if !reflect.DeepEqual(source.Clone(), before) {
			t.Fatal("mutable storage aliases source")
		}
		return nil
	}
	baseline := new(allocationFixture)
	if err := run(baseline); err != nil {
		t.Fatal(err)
	}
	for fail := 1; fail <= baseline.calls; fail++ {
		m := &allocationFixture{failAt: fail}
		if err := run(m); !errors.Is(err, errAllocationFixture) || m.live != 0 {
			t.Fatalf("guard %d/%d: live=%d err=%v", fail, baseline.calls, m.live, err)
		}
		if !reflect.DeepEqual(source.Clone(), before) {
			t.Fatal("source changed on refusal")
		}
	}
}

func TestCloneMemoryPayloadCharge(t *testing.T) {
	charges := func(size int) int64 {
		g := New()
		g.Add(ENode{Op: "payload", Data: make([]byte, size)})
		memory := new(allocationFixture)
		scope, _ := egraphcore.NewMemoryScope(memory)
		defer scope.Close()
		budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
		if _, err := g.CloneWithBudget(context.Background(), budget); err != nil {
			t.Fatal(err)
		}
		return memory.live
	}
	if delta := charges(73) - charges(0); delta != 73 {
		t.Fatalf("payload backing charge=%d want=73", delta)
	}
}

func TestCloneMemoryRefusalDoesNotCanonicalizeSource(t *testing.T) {
	g := New()
	for _, op := range []string{"a", "b", "c", "d"} {
		g.Add(NewNode(op))
	}
	g.Union(0, 1)
	g.Union(2, 3)
	g.Union(0, 2)
	_, visits, ok := g.store.FindWithoutCompression(3, 10)
	if !ok || visits != 3 {
		t.Fatal("fixture is not an uncompressed chain", visits)
	}
	s := &SearchState{Graph: g, Roots: map[string]Id{"root": 3}}
	worklist := append([]Id(nil), g.worklist...)
	facts, revision := g.facts, g.revision
	action := ActionCandidate{Rule: MustRewrite("same", "a", "a"), Match: MatchMatch{ClassID: 3}}
	run := func(memory *allocationFixture) error {
		scope, err := egraphcore.NewMemoryScope(memory)
		if err != nil {
			return err
		}
		defer scope.Close()
		budget, err := egraphcore.NewWorkBudgetWithMemory(0, scope)
		if err != nil {
			return err
		}
		_, err = s.ApplyActionWithCloneBudget(context.Background(), action, budget)
		return err
	}
	baseline := new(allocationFixture)
	if err := run(baseline); err != nil {
		t.Fatal(err)
	}
	for fail := 1; fail <= baseline.calls; fail++ {
		memory := &allocationFixture{failAt: fail}
		if err := run(memory); !errors.Is(err, errAllocationFixture) || memory.live != 0 {
			t.Fatalf("guard %d: live=%d err=%v", fail, memory.live, err)
		}
		_, after, ok := g.store.FindWithoutCompression(3, 10)
		if !ok || after != visits || g.facts != facts || g.revision != revision || !reflect.DeepEqual(g.worklist, worklist) {
			t.Fatal("source canonicalized before refusal")
		}
	}
}

func TestCloneMemoryLegacyAndZero(t *testing.T) {
	ctx := context.Background()
	g := new(EGraph)
	scope, _ := egraphcore.NewMemoryScope(new(allocationFixture))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	copy, err := g.CloneWithBudget(ctx, budget)
	if err != nil || copy == nil || g.store != nil || g.facts != nil {
		t.Fatal("zero graph mutated", err)
	}
	payload := map[string]any{}
	payload["self"] = payload
	g = New()
	id := g.Add(ENode{Op: "cyclic", Data: payload})
	s := &SearchState{Graph: g, Roots: map[string]Id{"root": id}}
	if s.Clone() == nil {
		t.Fatal("ordinary cyclic clone lost")
	}
	if got, err := s.CloneWithBudget(ctx, budget); got != nil || !errors.Is(err, ErrMemoryOperation) {
		t.Fatal(got, err)
	}
	if got, err := s.CloneWithBudget(ctx, nil); got == nil || err != nil {
		t.Fatal(got, err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := s.CloneWithBudget(canceled, budget); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestCloneMemoryTransition(t *testing.T) {
	for _, rhs := range []string{"alternative", "base"} {
		t.Run(rhs, func(t *testing.T) {
			g := New()
			id := g.Add(NewNode("base"))
			s := NewSearchState(g, map[string]Id{"root": id})
			relation, _ := s.Graph.Facts().Register(egraphcore.Schema{Name: "source", Columns: []egraphcore.Column{{Kind: egraphcore.StringValue}}})
			relation.Add([]egraphcore.Value[Id]{egraphcore.String[Id]("kept")}, nil)
			rule := MustRewrite("change", "base", rhs)
			actions := EnumerateActionsBounded(s.Graph, []*Rewrite{rule}, 1)
			cursor := relation.Snapshot()
			before := s.Clone()
			want, err := before.Clone().ApplyAction(actions[0])
			if err != nil {
				t.Fatal(err)
			}
			run := func(m *allocationFixture) error {
				scope, err := egraphcore.NewMemoryScope(m)
				if err != nil {
					return err
				}
				defer scope.Close()
				budget, err := egraphcore.NewWorkBudgetWithMemory(0, scope)
				if err != nil {
					return err
				}
				got, err := s.ApplyActionWithCloneBudget(context.Background(), actions[0], budget)
				if err != nil {
					if got != nil {
						t.Fatal("partial transition")
					}
					return err
				}
				// Rebuild and ordinary Clone can represent empty work/child
				// slices differently. Compare the same ordinary snapshot of each
				// result, and require no pending work in either live result.
				if len(got.State.Graph.worklist) != 0 || len(want.State.Graph.worklist) != 0 {
					t.Fatal("pending work")
				}
				gotResult, wantResult := *got, *want
				gotResult.State, wantResult.State = got.State.Clone(), want.State.Clone()
				for _, name := range []string{"source", "nvidia.typed_metadata"} {
					a, b := got.State.Graph.Facts().Get(name), want.State.Graph.Facts().Get(name)
					if a == nil || b == nil || !reflect.DeepEqual(a.Rows(), b.Rows()) || !reflect.DeepEqual(a.Schema(), b.Schema()) {
						t.Fatal("fact semantics differ", name)
					}
				}
				if got.State.Graph.Facts().TupleCount() != want.State.Graph.Facts().TupleCount() {
					t.Fatal("fact counts differ")
				}
				_, gotCursorErr := got.State.Graph.Facts().Get("source").AddedSince(cursor)
				_, wantCursorErr := want.State.Graph.Facts().Get("source").AddedSince(cursor)
				if !errors.Is(gotCursorErr, wantCursorErr) {
					t.Fatal("transition cursor reset differs", gotCursorErr, wantCursorErr)
				}
				// Newly materialized metadata relations acquire their epoch at
				// validation, before ordinary Clone and after the guarded clone.
				// Existing relation lineage is checked separately above and by
				// the core clone-history controls.
				gotResult.State.Graph.facts, wantResult.State.Graph.facts = nil, nil
				if !reflect.DeepEqual(gotResult, wantResult) {
					t.Fatalf("transition differs: got=%+v want=%+v", got, want)
				}
				return nil
			}
			baseline := new(allocationFixture)
			if err := run(baseline); err != nil {
				t.Fatal(err)
			}
			for fail := 1; fail <= baseline.calls; fail++ {
				m := &allocationFixture{failAt: fail}
				if err := run(m); !errors.Is(err, errAllocationFixture) || m.live != 0 {
					t.Fatalf("guard %d: %v live=%d", fail, err, m.live)
				}
				if !reflect.DeepEqual(s.Clone(), before) {
					t.Fatal("source mutated on refusal")
				}
				if rows, err := relation.AddedSince(cursor); err != nil || len(rows) != 0 {
					t.Fatal("source cursor mutated", rows, err)
				}
			}
		})
	}
}
