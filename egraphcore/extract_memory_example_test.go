package egraphcore_test

import (
	"fmt"

	"github.com/tmc/lib/egraphcore"
)

func ExampleRelaxBestCostsWorklistWithBudget() {
	nodes := []string{"leaf"}
	scope, _ := egraphcore.NewMemoryScope(new(exampleMemory))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	best, _, err := egraphcore.RelaxBestCostsWorklistWithBudget([]uint32{0}, func(uint32) []string { return nodes },
		func(id uint32) uint32 { return id }, func(string) []uint32 { return nil }, func(id uint32) uint32 { return id }, nil, nil,
		func(string, []egraphcore.RankedBestCostNode[string, uint64], bool) (float64, uint64, bool, error) {
			return 1, 1, true, nil
		},
		func(a, b egraphcore.RankedBestCostNode[string, uint64]) bool { return a.Cost < b.Cost }, nil, 10, budget)
	fmt.Println(best[0].Cost, err)
	// Output: 1 <nil>
}

func ExampleWouldCreateCycleWithBudget() {
	scope, _ := egraphcore.NewMemoryScope(new(exampleMemory))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	cycle, err := egraphcore.WouldCreateCycleWithBudget(uint32(1), []uint32{0}, func(id uint32) uint32 { return id }, func(id uint32) (uint32, bool) { return id, true }, func(uint32) []uint32 { return nil }, nil, budget)
	fmt.Println(cycle, err)
	// Output: false <nil>
}
