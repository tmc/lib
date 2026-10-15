package egraphcore

import (
	"errors"
	"reflect"
	"testing"
)

type extractionFixtureNode struct{ children []uint32 }

func extractionFixture(work *WorkBudget, nodes func(uint32) []extractionFixtureNode) (map[uint32]RankedBestCostNode[extractionFixtureNode, uint64], int, error) {
	return RelaxBestCostsWorklistWithBudget([]uint32{0, 1, 2}, nodes,
		func(id uint32) uint32 { return id }, func(n extractionFixtureNode) []uint32 { return n.children }, func(id uint32) uint32 { return id }, nil, nil,
		func(_ extractionFixtureNode, children []RankedBestCostNode[extractionFixtureNode, uint64], known bool) (float64, uint64, bool, error) {
			if !known {
				return 0, 0, false, nil
			}
			cost, rank := float64(1), uint64(1)
			for _, child := range children {
				cost += child.Cost
				rank += child.Rank
			}
			return cost, rank, true, nil
		}, func(a, b RankedBestCostNode[extractionFixtureNode, uint64]) bool { return a.Cost < b.Cost }, nil, 100, work)
}

func TestExtractionMemoryCoreBeforeCallbacks(t *testing.T) {
	for _, cycle := range []bool{false, true} {
		memory := &fixtureMemory{failAt: 3}
		budget, scope, err := fixtureWork(t, memory)
		if err != nil {
			t.Fatal(err)
		}
		calls := 0
		if cycle {
			_, err = WouldCreateCycleWithBudget(uint32(4), []uint32{1}, func(id uint32) uint32 { return id }, func(uint32) (extractionFixtureNode, bool) { calls++; return extractionFixtureNode{}, true }, func(n extractionFixtureNode) []uint32 { return n.children }, nil, budget)
		} else {
			_, _, err = extractionFixture(budget, func(uint32) []extractionFixtureNode { calls++; return nil })
		}
		if !errors.Is(err, errMemoryFixture) || calls != 0 {
			t.Fatal(cycle, calls, err)
		}
		scope.Close()
		if memory.used != 0 {
			t.Fatal(memory.used)
		}
	}
}

func TestExtractionMemoryCoreEveryReservation(t *testing.T) {
	graph := map[uint32][]extractionFixtureNode{0: {{}}, 1: {{children: []uint32{0, 0}}}, 2: {{children: []uint32{1, 0}}}}
	nodes := func(id uint32) []extractionFixtureNode { return graph[id] }
	want, _, err := extractionFixture(nil, nodes)
	if err != nil || want[2].Cost != 5 || want[2].Rank != 5 {
		t.Fatal(want, err)
	}
	run := func(memory *fixtureMemory) error {
		budget, scope, err := fixtureWork(t, memory)
		if err != nil {
			return err
		}
		defer scope.Close()
		got, _, err := extractionFixture(budget, nodes)
		if err != nil {
			if got != nil {
				t.Fatal("partial table on refusal")
			}
			return err
		}
		if !reflect.DeepEqual(got, want) || scope.Bytes() == 0 {
			t.Fatal(got, want, scope.Bytes())
		}
		return nil
	}
	baseline := new(fixtureMemory)
	if err := run(baseline); err != nil {
		t.Fatal(err)
	}
	for fail := 1; fail <= baseline.calls; fail++ {
		memory := &fixtureMemory{failAt: fail}
		if err := run(memory); !errors.Is(err, errMemoryFixture) {
			t.Fatalf("reservation %d/%d: %v", fail, baseline.calls, err)
		}
		if memory.used != 0 {
			t.Fatal("leaked reservation", fail, memory.used)
		}
	}
}

func TestExtractionMemoryCycleEveryReservation(t *testing.T) {
	children := [][]uint32{nil, {0}, {1}}
	run := func(memory *fixtureMemory) error {
		budget, scope, err := fixtureWork(t, memory)
		if err != nil {
			return err
		}
		defer scope.Close()
		cycle, err := WouldCreateCycleWithBudget(uint32(3), []uint32{2, 1}, func(id uint32) uint32 { return id }, func(id uint32) (uint32, bool) { return id, true }, func(id uint32) []uint32 { return children[id] }, nil, budget)
		if err != nil {
			return err
		}
		if cycle || scope.Bytes() == 0 {
			t.Fatal(cycle, scope.Bytes())
		}
		return nil
	}
	baseline := new(fixtureMemory)
	if err := run(baseline); err != nil {
		t.Fatal(err)
	}
	for fail := 1; fail <= baseline.calls; fail++ {
		memory := &fixtureMemory{failAt: fail}
		if err := run(memory); !errors.Is(err, errMemoryFixture) {
			t.Fatalf("reservation %d: %v", fail, err)
		}
		if memory.used != 0 {
			t.Fatal(fail, memory.used)
		}
	}
}
