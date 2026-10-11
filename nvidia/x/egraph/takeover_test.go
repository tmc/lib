package egraph

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func TestTakeoverCloneMutablePayloadIsolation(t *testing.T) {
	g := New()
	payload := map[string][]int{"values": {1, 2, 3}}
	id := g.Add(ENode{Op: "payload", Data: payload})
	clone := g.Clone()
	payload["values"][0] = 99
	got := clone.Class(id).Nodes[0].Data.(map[string][]int)
	if got["values"][0] != 1 {
		t.Fatal("graph clone retained mutable payload alias")
	}

	state := NewSearchState(g, map[string]Id{"root": id})
	state.Derivation = []Action{{RuleName: "r", Subst: Subst{"?x": id}}}
	branch := state.Clone()
	branch.Derivation[0].Subst["?x"] = 999
	if state.Derivation[0].Subst["?x"] == 999 {
		t.Fatal("search-state clone retained substitution alias")
	}
}

func TestTakeoverCloneCyclicPayload(t *testing.T) {
	payload := map[string]any{}
	payload["self"] = payload
	g := New()
	g.Add(ENode{Op: "cyclic", Data: payload})
	clone := g.Clone()
	if clone == nil || len(clone.Classes()) != len(g.Classes()) {
		t.Fatal("clone lost cyclic payload graph")
	}
	var got map[string]any
	for _, class := range clone.Classes() {
		for _, node := range class.Nodes {
			if node.Op == "cyclic" {
				got, _ = node.Data.(map[string]any)
			}
		}
	}
	if got == nil {
		t.Fatal("cyclic payload missing from clone")
	}
	inner, ok := got["self"].(map[string]any)
	if !ok || reflect.ValueOf(inner).Pointer() != reflect.ValueOf(got).Pointer() {
		t.Fatal("clone did not preserve cycle within cloned payload")
	}
}

func TestTakeoverMandatoryRenameCannotBypassGuard(t *testing.T) {
	rules := MandatoryRewriteRules()
	rules[0].Name = "renamed_q2k_rule"
	g := New()
	w := g.Add(NewLeaf("w"))
	scales := g.Add(NewLeaf("scales"))
	x := g.Add(NewLeaf("x"))
	load := g.Add(NewNode("load_unhoisted_q2k", w, scales))
	root := g.Add(NewNode("q2_k_matvec", load, x))
	if root == 0 {
		t.Fatal("unexpected root id")
	}
	if got := rules[0].Apply(g); got != 0 {
		t.Fatalf("renamed mandatory rule applied %d merges", got)
	}
}

func TestTakeoverCyclicPayloadRejectedByCheckedAdd(t *testing.T) {
	payload := map[string]any{}
	payload["self"] = payload
	g := New()
	if _, err := g.AddChecked(ENode{Op: "cyclic", Data: payload}); err == nil {
		t.Fatal("accepted cyclic mutable payload")
	}
}

func TestTakeoverMatchWorkBudget(t *testing.T) {
	g := New()
	leaf := g.Add(NewLeaf("x"))
	root := leaf
	for i := 0; i < 100; i++ {
		root = g.Add(NewNode("wrap", root))
	}
	p := Pat("wrap", Var("x"))
	if _, err := p.MatchEClassBounded(g, root, make(Subst), 1); !errors.Is(err, ErrMatchWorkLimit) {
		t.Fatalf("match budget error = %v", err)
	}
}

func TestTakeoverOperandProofIsRequired(t *testing.T) {
	g := New()
	tokens := g.Add(NewLeaf("tokens"))
	indices := g.Add(NewLeaf("indices"))
	width := g.Add(NewLeaf(8))
	root := g.Add(NewNode("moe_gather", tokens, indices, width))
	state := NewSearchState(g, map[string]Id{"root": root})
	shape, _ := cudashape.New(8, 16)
	meta := TypedMetadata{Shape: shape, DType: cudair.Float32, Layout: cudashape.LayoutRowMajor, Effect: cudair.EffectReadOnly, Alignment: 16}
	state.SetMetadata(root, meta)
	state.SetMetadata(tokens, meta)
	state.SetMetadata(width, meta)
	mut := meta
	mut.Effect = cudair.EffectMutating
	state.SetMetadata(indices, mut)
	candidates := EnumerateActions(state.Graph, MandatoryRewriteRules())
	for _, candidate := range candidates {
		if candidate.Rule.Name == "vectorize_moe_gather_w16" {
			if _, err := state.ApplyAction(candidate); !errors.Is(err, ErrEffectBoundaryViolation) {
				t.Fatalf("operand effect error = %v", err)
			}
			return
		}
	}
	t.Fatal("vectorization candidate missing")
}
