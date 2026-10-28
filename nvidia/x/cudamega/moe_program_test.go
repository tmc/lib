package cudamega

import (
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
	"testing"
)

func mixtureProgram(t *testing.T, workers uint32) *Program {
	t.Helper()
	p, err := NewMoEMixtureProgram(MoEWeightConfig{3, 2, 8, 17}, Grid{CTAs: workers, ThreadsPerCTA: 1, MaxElements: 8, Policy: CountedPolicy}, SM80)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestMoEMixtureContract(t *testing.T) {
	for _, workers := range []uint32{1, 2, 4, 8} {
		p := mixtureProgram(t, workers)
		inv, err := p.Invocation(8)
		if err != nil {
			t.Fatal(err)
		}
		if inv.ScratchWords != 24 {
			t.Fatalf("scratch words=%d", inv.ScratchWords)
		}
		for _, n := range []uint64{0, 7, 9, 17} {
			if _, err := p.Invocation(n); err == nil {
				t.Fatalf("accepted extent %d", n)
			}
		}
		layout, err := p.Storage(8)
		if err != nil {
			t.Fatal(err)
		}
		for i, s := range layout {
			if s.Offset != 8*i || s.Elements != 8 {
				t.Fatalf("layout=%+v", layout)
			}
		}
		steps := p.Steps()
		if len(steps) != 3 || len(p.inputSources(0)) != 0 || len(p.inputSources(2)) != 2 {
			t.Fatal("incorrect data bindings")
		}
		d, ok := steps[0].Operation.MoE()
		if !ok || d.Expert != 0 {
			t.Fatal("missing descriptor")
		}
		d.Config.Hidden = 999
		again, _ := steps[0].Operation.MoE()
		if again.Config.Hidden != 17 {
			t.Fatal("descriptor aliases program")
		}
	}
}

func TestMoETypedAdmission(t *testing.T) {
	cfg := MoEWeightConfig{3, 2, 8, 17}
	a, _ := MoEGateUpOperation(cfg, 0)
	c, _ := MoECombineOperation(cfg)
	for _, expert := range []int{-1, 2} {
		if _, err := MoEGateUpOperation(cfg, expert); err == nil {
			t.Fatal("accepted invalid expert")
		}
	}
	for _, tt := range []struct {
		name  string
		tasks []cudataskgraph.Task
		ops   map[string]Operation
	}{
		{"forged", []cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice, Extent: 8}}, map[string]Operation{"a": {Kind: MoEGateUp}}},
		{"mixed", []cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice, Extent: 8}, {ID: "b", Scope: cudataskgraph.ScopeDevice, Extent: 8}}, map[string]Operation{"a": a, "b": {Kind: Add}}},
		{"combine without input", []cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice, Extent: 8}}, map[string]Operation{"a": c}},
		{"wrong extent", []cudataskgraph.Task{{ID: "a", Scope: cudataskgraph.ScopeDevice, Extent: 7}}, map[string]Operation{"a": a}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g, err := cudataskgraph.New(tt.tasks)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = NewProgram(g, tt.ops, Grid{CTAs: 1, ThreadsPerCTA: 1, MaxElements: 8, Policy: CountedPolicy}, SM80); err == nil {
				t.Fatal("accepted invalid typed program")
			}
		})
	}
}
