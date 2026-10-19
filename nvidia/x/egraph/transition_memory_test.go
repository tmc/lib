package egraph

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/egraphcore"
)

type transitionProbe struct {
	allocationFixture
	failedAt string
}

type opaqueTransitionError struct{ calls int }

func (e *opaqueTransitionError) Error() string { e.calls++; return "opaque" }

type opaqueTransitionSlice []int

func (opaqueTransitionSlice) Error() string { panic("opaque Error invoked") }

func TestTransitionMemoryOpaqueErrors(t *testing.T) {
	m := new(allocationFixture)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	opaque := new(opaqueTransitionError)
	if got := graphErrorf(b, "prefix: %w", opaque); got != opaque || opaque.calls != 0 {
		t.Fatal("opaque error inspected or replaced")
	}
	var nonComparable error = opaqueTransitionSlice{1}
	if got := graphErrorf(b, "prefix: %w", nonComparable); reflect.TypeOf(got) != reflect.TypeOf(nonComparable) {
		t.Fatal("opaque error replaced")
	}
}

func (m *transitionProbe) Reserve(n int64) error {
	err := m.allocationFixture.Reserve(n)
	if err != nil {
		var pcs [32]uintptr
		n := runtime.Callers(1, pcs[:])
		frames := runtime.CallersFrames(pcs[:n])
		for {
			f, more := frames.Next()
			m.failedAt += f.Function + "\n"
			if !more {
				break
			}
		}
	}
	return err
}

func TestTransitionMemoryBudgetBoundary(t *testing.T) {
	s, action := transitionFixture(t, "growth")
	m := new(allocationFixture)
	scope, _ := egraphcore.NewMemoryScope(m)
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	if _, err := s.CloneWithBudget(context.Background(), b); err != nil {
		t.Fatal(err)
	}
	calls := m.calls
	scope.Close()
	probe := &transitionProbe{allocationFixture: allocationFixture{failAt: calls + 1}}
	scope, _ = egraphcore.NewMemoryScope(probe)
	defer scope.Close()
	b, _ = egraphcore.NewWorkBudgetWithMemory(0, scope)
	got, err := s.ApplyActionWithCloneBudget(context.Background(), action, b)
	if got != nil || !errors.Is(err, errAllocationFixture) || !strings.Contains(probe.failedAt, "egraphcore.NewWorkBudgetWithMemory") {
		t.Fatalf("transition budget was not admitted after clone: out=%v err=%v failure=%s", got, err, probe.failedAt)
	}
}

func TestTransitionMemoryDiagnostics(t *testing.T) {
	for _, format := range []string{"plain error", "name %q value %d", "%w: operand %s"} {
		var args []any
		switch format {
		case "name %q value %d":
			args = []any{"x\x00\"z", Id(9)}
		case "%w: operand %s":
			args = []any{ErrInvalidRewriteProof, "?x"}
		}
		want := fmt.Errorf(format, args...)
		run := func(fail int) (int, error) {
			m := &allocationFixture{failAt: fail}
			scope, err := egraphcore.NewMemoryScope(m)
			if err != nil {
				return m.calls, err
			}
			b, err := egraphcore.NewWorkBudgetWithMemory(0, scope)
			if err == nil {
				err = graphErrorf(b, format, args...)
			}
			scope.Close()
			if m.live != 0 {
				t.Fatal("diagnostic leaked")
			}
			return m.calls, err
		}
		calls, got := run(0)
		if got == nil || got.Error() != want.Error() || errors.Is(got, ErrInvalidRewriteProof) != errors.Is(want, ErrInvalidRewriteProof) {
			t.Fatal(got, want)
		}
		for fail := 1; fail <= calls; fail++ {
			if _, err := run(fail); !errors.Is(err, errAllocationFixture) {
				t.Fatal(fail, err)
			}
		}
	}
}

func TestTransitionMemorySiteAdmission(t *testing.T) {
	for _, site := range []string{"instantiate", "fingerprint", "identity", "action", "rebuild"} {
		t.Run(site, func(t *testing.T) {
			s, action := transitionFixture(t, "growth")
			m := new(allocationFixture)
			scope, _ := egraphcore.NewMemoryScope(m)
			defer scope.Close()
			b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
			m.failAt = m.calls + 1
			var err error
			switch site {
			case "instantiate":
				_, err = action.Rule.RHS.instantiateWithBudget(s.Graph, action.Match.Subst, b)
			case "fingerprint":
				_, err = classFingerprintWithBudget(s.Graph, action.Match.ClassID, b)
			case "identity":
				_, err = ruleIdentityWithBudget(action.Rule, b)
			case "action":
				_, _, err = copyActionBindings(action.Match.Subst, action.Match.Binding, b)
			case "rebuild":
				_, err = s.Graph.rebuildWithBudget(b)
			}
			if !errors.Is(err, errAllocationFixture) {
				t.Fatalf("%s bypassed admission: %v", site, err)
			}
		})
	}
}

func transitionFixture(t *testing.T, kind string) (*SearchState, ActionCandidate) {
	t.Helper()
	g := New()
	a := g.Add(NewNode("a"))
	b := g.Add(NewNode("b"))
	left := g.Add(NewNode("f", a))
	right := g.Add(NewNode("f", b))
	root := g.Add(NewNode("pair", left, right))
	s := NewSearchState(g, map[string]Id{"root": root})
	s.Derivation = []Action{{RuleName: "previous", Subst: Subst{"?old": a}, Binding: egraphcore.Binding[Id]{"old": egraphcore.String[Id]("retained")}}}
	r, _ := s.Graph.Facts().Register(egraphcore.Schema{Name: "facts", Columns: []egraphcore.Column{{Kind: egraphcore.ClassValue}, {Kind: egraphcore.StringValue}}})
	r.Add([]egraphcore.Value[Id]{egraphcore.Class(a), egraphcore.String[Id]("same")}, s.Graph.Find)
	r.MarkSubsumed([]egraphcore.Value[Id]{egraphcore.Class(b), egraphcore.String[Id]("same")}, s.Graph.Find)
	rule := MustRewrite("change", "a", "b")
	switch kind {
	case "growth":
		rule = MustRewrite("grow", "a", "(outer (inner a) leaf)")
	case "noop":
		rule = MustRewrite("same", "a", "a")
	case "facts":
		rule = &Rewrite{Name: "fact", LHS: Pat("a"), RelationActions: []egraphcore.FactAction[Id]{{Kind: egraphcore.FactInsert, Relation: "facts", Terms: []egraphcore.Term[Id]{egraphcore.Constant(egraphcore.Class(a)), egraphcore.Constant(egraphcore.String[Id]("new\x00\"value"))}}, {Kind: egraphcore.FactSubsume, Relation: "facts", Terms: []egraphcore.Term[Id]{egraphcore.Constant(egraphcore.Class(a)), egraphcore.Constant(egraphcore.String[Id]("same"))}}}}
	case "subsumption":
		rule, _ = NewSubsumingRewrite("hide", "a", "b")
	case "congruence":
	default:
		t.Fatal(kind)
	}
	actions, _, err := EnumerateActionsBoundedChecked(s.Graph, []*Rewrite{rule}, 20)
	if err != nil || len(actions) == 0 {
		t.Fatal(actions, err)
	}
	return s, actions[0]
}

func equalTransition(t *testing.T, got, want *TransitionResult) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatal(got, want)
	}
	a, b := *got, *want
	a.State, b.State = got.State.Clone(), want.State.Clone()
	for _, name := range []string{"facts", "nvidia.typed_metadata"} {
		x, y := a.State.Graph.Facts().Get(name), b.State.Graph.Facts().Get(name)
		if x == nil && y == nil {
			continue
		}
		if x == nil || y == nil || !reflect.DeepEqual(x.Schema(), y.Schema()) || !reflect.DeepEqual(x.Rows(), y.Rows()) {
			t.Fatal("fact semantics differ", name)
		}
		if x != nil {
			rows, complete, err := x.Query(nil, a.State.Graph.Find, egraphcore.NewWorkBudget(1000))
			other, done, e := y.Query(nil, b.State.Graph.Find, egraphcore.NewWorkBudget(1000))
			if err != nil || e != nil || !complete || !done || !reflect.DeepEqual(rows, other) {
				t.Fatal("hidden facts differ", name)
			}
		}
	}
	a.State.Graph.facts, b.State.Graph.facts = nil, nil
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("transition differs\ngot=%+v\nwant=%+v", a, b)
	}
}

func TestTransitionMemoryEveryReservation(t *testing.T) {
	for _, kind := range []string{"growth", "congruence", "facts", "subsumption", "noop"} {
		t.Run(kind, func(t *testing.T) {
			s, action := transitionFixture(t, kind)
			before := s.Clone()
			want, err := s.Clone().ApplyAction(action)
			if err != nil {
				t.Fatal(err)
			}
			run := func(fail int) (int, error) {
				m := &allocationFixture{failAt: fail}
				scope, err := egraphcore.NewMemoryScope(m)
				if err == nil {
					var budget *egraphcore.WorkBudget
					budget, err = egraphcore.NewWorkBudgetWithMemory(0, scope)
					if err == nil {
						var got *TransitionResult
						got, err = s.ApplyActionWithCloneBudget(context.Background(), action, budget)
						if err != nil {
							if got != nil {
								t.Fatal("partial transition escaped")
							}
						} else {
							equalTransition(t, got, want)
							if m.live != scope.Bytes() || m.live <= 0 {
								t.Fatal("result not owned")
							}
						}
					}
					scope.Close()
					scope.Close()
				}
				if m.live != 0 {
					t.Fatal("leaked", fail, m.live)
				}
				if !reflect.DeepEqual(s.Clone(), before) {
					t.Fatal("original changed", fail)
				}
				return m.calls, err
			}
			calls, err := run(0)
			if err != nil {
				t.Fatal(err)
			}
			for fail := 1; fail <= calls; fail++ {
				if _, err := run(fail); !errors.Is(err, errAllocationFixture) {
					t.Fatalf("guard %d/%d: %v", fail, calls, err)
				}
			}
			t.Logf("checked %d reservation refusals", calls)
		})
	}
}

func TestTransitionMemoryRuleEncoding(t *testing.T) {
	// Retain the predecessor's formatting independently of the new encoder.
	value := egraphcore.Value[Id]{Kind: egraphcore.StringValue, String: "a\x00\"z"}
	r := &Rewrite{Name: "identity", LHS: Pat("f", Var("?x")), RHS: Var("?x"), Conditions: []*Pattern{Pat("g", Var("?x"))}, RelationConditions: []RelationCondition{{Name: "r", Terms: []egraphcore.Term[Id]{egraphcore.Constant(value), egraphcore.Variable[Id]("n"), egraphcore.Wildcard[Id]()}}}, RelationActions: []egraphcore.FactAction[Id]{{Kind: egraphcore.FactInsert, Relation: "r", Terms: []egraphcore.Term[Id]{egraphcore.Constant(value)}}}}
	want := fmt.Sprintf("identity|(f ?x)=>?x where (g ?x) facts r(%d:%d:%d:%t:%d:%q,?n,_) do 1:r(=%#v)", value.Kind, uint32(value.Class), value.Int64, value.Bool, value.Float, value.String, value)
	m := new(allocationFixture)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	got, err := ruleIdentityWithBudget(r, b)
	if err != nil || got != want || r.Identity() != want {
		t.Fatalf("encoding differs: %q want %q err %v", got, want, err)
	}
}

func TestTransitionMemoryMandatoryDefinitions(t *testing.T) {
	// SHA256 of each predecessor name|lhs=>rhs literal, in public order.
	goldens := []string{
		"def4d24f7ea3e5a8677d4cacc4df0b3d850665a455e70a52c797bbb6d8a2c689",
		"d693794e2f77f81408aef6e7dadb374a1cd0eca908bb60c1982cc4e0fe90fc37",
		"1c9ed4a4c8ef9d0481fc93310071197fde1b5e39d69089dc2774010550935d24",
		"e6ecb4edd93a917b413f7f2c4e194a55bb4b1f3684fe82d853266ce3fe86f41b",
		"cafceef572f35e586fec57da8c89ab9bb731e56bcab1bdd0c43acb2a57661930",
		"11189df0c95a1a2f957bf3a7c44dcd5da9683bebad7d9ea073c02d84fd961d48",
		"fde7ed55a6c3323099dc2ee4f420b2dd746a071c2f9bb3fad017e3e6b7dda74f",
		"fdb3da304d6a844064276800fa71895c7c82b17e12552f8e3a2f868fe6bdf75a",
	}
	a, b := MandatoryRewriteRules(), MandatoryRewriteRules()
	if len(a) != 8 || len(b) != 8 {
		t.Fatal("definition population")
	}
	for i, d := range mandatoryRewriteDefinitions {
		if fmt.Sprintf("%x", sha256.Sum256([]byte(a[i].Identity()))) != goldens[i] {
			t.Fatal("predecessor definition changed", i)
		}
		if a[i].Name != d.name || a[i].LHS.String() != d.lhs || a[i].RHS.String() != d.rhs || a[i] == b[i] || a[i].LHS == b[i].LHS || a[i].RHS == b[i].RHS {
			t.Fatal("definition or owner changed", i)
		}
	}
	for _, r := range a {
		m := new(allocationFixture)
		scope, _ := egraphcore.NewMemoryScope(m)
		budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
		yes, err := isMandatoryRewriteWithBudget(r, budget)
		if err != nil || !yes {
			t.Fatal(r.Name, yes, err)
		}
		calls := m.calls
		scope.Close()
		for fail := 3; fail <= calls; fail++ {
			m := &allocationFixture{failAt: fail}
			scope, _ := egraphcore.NewMemoryScope(m)
			budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
			yes, err := isMandatoryRewriteWithBudget(r, budget)
			scope.Close()
			if yes || !errors.Is(err, errAllocationFixture) || m.live != 0 {
				t.Fatal("mandatory admission bypass", fail, yes, err)
			}
		}
	}
	a[0].LHS.Op = "mutated"
	if b[0].LHS.Op == "mutated" || MandatoryRewriteRules()[0].LHS.Op == "mutated" {
		t.Fatal("fresh definitions alias")
	}
}

func TestTransitionMemoryReplayAndCancellation(t *testing.T) {
	s, action := transitionFixture(t, "growth")
	s.Derivation = nil
	m := new(allocationFixture)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	got, err := s.ApplyActionWithCloneBudget(context.Background(), action, b)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := ReplaySearchState(s, []*Rewrite{action.Rule}, got.State.Derivation)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(replay.Derivation, got.State.Derivation) {
		t.Fatal("replay identity differs")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if out, err := s.ApplyActionWithCloneBudget(ctx, action, b); out != nil || !errors.Is(err, context.Canceled) {
		t.Fatal(out, err)
	}
}
