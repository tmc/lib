package egraph

import (
	"context"
	"errors"
	"math"
	"math/big"
	"reflect"
	"testing"

	"github.com/tmc/lib/egraphcore"
)

type extractionData struct {
	Values []float64
	Named  map[string][]int
	Alias  []float64
}

func extractionMemoryGraph(t *testing.T) (*EGraph, Id) {
	t.Helper()
	values := []float64{2, 3}
	g := New()
	a := g.Add(ENode{Op: "payload", Data: &extractionData{Values: values, Alias: values, Named: map[string][]int{"a": {7, 9}}}})
	root := g.Add(NewNode("pair", a, a))
	if err := g.RebuildChecked(); err != nil {
		t.Fatal(err)
	}
	return g, root
}

func TestExtractionMemoryEveryReservation(t *testing.T) {
	g, root := extractionMemoryGraph(t)
	want, cost, stats, err := ExtractBoundedWithStats(context.Background(), g, root, nil, DefaultExtractionLimits())
	if err != nil || cost != 3 || stats.Nodes != 3 {
		t.Fatal(want, cost, stats, err)
	}
	source := g.class(g.Find(want.Children[0].Class)).Nodes[0].Data.(*extractionData)
	run := func(memory *allocationFixture) error {
		owner, err := ExtractBoundedWithMemory(context.Background(), g, root, nil, DefaultExtractionLimits(), memory)
		if err != nil {
			if owner != nil && owner.Expr != nil {
				t.Fatal("partial expression on refusal")
			}
			owner.Close()
			return err
		}
		defer owner.Close()
		if owner.Cost != cost || owner.Stats != stats || !owner.Expr.Equal(want) {
			t.Fatal(owner.Expr, want, owner.Cost, cost, owner.Stats, stats)
		}
		if memory.live <= 0 || memory.live >= memory.peak {
			t.Fatal("scratch not released or result uncharged", memory.live, memory.peak)
		}
		data := owner.Expr.Children[0].Data.(*extractionData)
		if data == source || &data.Values[0] == &source.Values[0] || &data.Named["a"][0] == &source.Named["a"][0] {
			t.Fatal("mutable source alias")
		}
		if &data.Values[0] != &data.Alias[0] {
			t.Fatal("payload alias structure changed")
		}
		data.Values[0], data.Named["a"][0] = 100, 200
		if source.Values[0] != 2 || source.Named["a"][0] != 7 || want.Children[0].Data.(*extractionData).Values[0] != 2 {
			t.Fatal("source or oracle mutated")
		}
		if owner.Expr.Children[1].Data.(*extractionData).Values[0] != 2 {
			t.Fatal("separate tree occurrences share mutable payload")
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
			t.Fatal("reservation leaked", fail, memory.live)
		}
	}
}

type memoryCostFunc func(ENode, []float64, *egraphcore.MemoryScope) (float64, error)

func (f memoryCostFunc) NodeCost(node ENode, costs []float64) float64 {
	cost, _ := f(node, costs, nil)
	return cost
}
func (f memoryCostFunc) NodeCostWithMemory(node ENode, costs []float64, scope *egraphcore.MemoryScope) (float64, error) {
	return f(node, costs, scope)
}

func TestExtractionMemoryCancellationAndModelCapability(t *testing.T) {
	g := New()
	root := g.Add(NewNode("leaf"))
	_ = g.RebuildChecked()
	unknownCalls := 0
	unknown := boundedCostFunc(func(ENode, []float64) float64 { unknownCalls++; return 1 })
	memory := new(allocationFixture)
	owner, err := ExtractBoundedWithMemory(context.Background(), g, root, unknown, DefaultExtractionLimits(), memory)
	if owner != nil || !errors.Is(err, ErrMemoryOperation) || unknownCalls != 0 || memory.live != 0 {
		t.Fatal(owner, err, unknownCalls, memory.live)
	}
	legacy, err := ExtractBoundedWithMemory(context.Background(), g, root, unknown, DefaultExtractionLimits(), nil)
	if err != nil || legacy.Expr == nil || unknownCalls == 0 {
		t.Fatal(legacy, err, unknownCalls)
	}
	legacy.Close()
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	model := memoryCostFunc(func(ENode, []float64, *egraphcore.MemoryScope) (float64, error) { calls++; cancel(); return 1, nil })
	owner, err = ExtractBoundedWithMemory(ctx, g, root, model, DefaultExtractionLimits(), memory)
	if !errors.Is(err, context.Canceled) || calls != 1 || owner == nil || owner.Expr != nil || owner.Stats.Work == 0 || memory.live != 0 {
		t.Fatal(owner, err, calls, memory.live)
	}
	owner.Close()
	owner.Close()
	if memory.live != 0 {
		t.Fatal(memory.live)
	}
}

type extractionFormatter string

func (extractionFormatter) String() string { return "formatter must not run" }

func TestExtractionMemoryUnsupportedPayloads(t *testing.T) {
	cyclic := map[string]any{}
	cyclic["self"] = cyclic
	private := struct{ private []byte }{private: []byte{1}}
	values := []any{func() {}, make(chan int), extractionFormatter("value"), private, map[*int]int{new(int): 1}, cyclic}
	for _, value := range values {
		g := New()
		root := g.Add(ENode{Op: "value", Data: value})
		_ = g.RebuildChecked()
		memory := new(allocationFixture)
		owner, err := ExtractBoundedWithMemory(context.Background(), g, root, nil, DefaultExtractionLimits(), memory)
		if !errors.Is(err, ErrMemoryOperation) || owner != nil && owner.Expr != nil || memory.live != 0 {
			t.Fatalf("%T owner=%v err=%v live=%d", value, owner, err, memory.live)
		}
		owner.Close()
	}
}

func TestExtractionMemoryPrimitiveKeysAndProvenance(t *testing.T) {
	values := []any{nil, "\x00\n世界", false, int64(math.MinInt64), uint64(math.MaxUint64), float32(1.234567), math.NaN(), math.Inf(1), math.Copysign(0, -1)}
	for _, value := range values {
		memory := new(allocationFixture)
		scope, _ := egraphcore.NewMemoryScope(memory)
		work, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
		node := ENode{Op: "value", Data: value, Children: []Id{0, math.MaxUint32}}
		key, err := primitiveNodeKey(node, work)
		if err != nil || key != node.Key() {
			t.Fatalf("%T: key=%q want=%q err=%v", value, key, node.Key(), err)
		}
		scope.Close()
	}
	g := New()
	leaf := g.Add(NewLeaf(int64(7)))
	root := g.Add(NewNode("original", leaf))
	_ = g.RebuildChecked()
	rule := MustRewrite("replace", "(original ?x)", "(replacement ?x)")
	if _, err := rule.ApplyWithBudget(g, egraphcore.NewWorkBudget(100000)); err != nil {
		t.Fatal(err)
	}
	if err := g.RebuildChecked(); err != nil {
		t.Fatal(err)
	}
	model := memoryCostFunc(func(node ENode, children []float64, _ *egraphcore.MemoryScope) (float64, error) {
		cost := float64(1)
		for _, child := range children {
			cost += child
		}
		if node.Op == "original" {
			cost += 10
		}
		return cost, nil
	})
	want, cost, err := ExtractBounded(context.Background(), g, root, model, DefaultExtractionLimits())
	if err != nil {
		t.Fatal(err)
	}
	if want.Op != "replacement" || want.Origin != "replace" {
		t.Fatal("provenance fixture not selected", want.Op, want.Origin)
	}
	memory := new(allocationFixture)
	owner, err := ExtractBoundedWithMemory(context.Background(), g, root, model, DefaultExtractionLimits(), memory)
	if err != nil || owner.Cost != cost || !owner.Expr.Equal(want) || owner.Expr.Origin != want.Origin {
		t.Fatal(owner, want, err)
	}
	owner.Close()
}

func TestExtractionMemoryExactLargeRank(t *testing.T) {
	children := []egraphcore.RankedBestCostNode[ENode, *big.Int]{{Rank: new(big.Int).Lsh(big.NewInt(1), 130)}, {Rank: new(big.Int).Lsh(big.NewInt(1), 130)}}
	memory := new(allocationFixture)
	scope, _ := egraphcore.NewMemoryScope(memory)
	defer scope.Close()
	work, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	got, err := memoryTreeRank(children, work)
	want := new(big.Int).Add(children[0].Rank, children[1].Rank)
	want.Add(want, big.NewInt(1))
	if err != nil || got.Cmp(want) != 0 || got.BitLen() != 132 {
		t.Fatal(got, want, err)
	}
	if reflect.ValueOf(got.Bits()).Pointer() == reflect.ValueOf(children[0].Rank.Bits()).Pointer() {
		t.Fatal("rank aliases child storage")
	}
}
