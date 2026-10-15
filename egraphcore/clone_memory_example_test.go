package egraphcore_test

import (
	"fmt"
	"github.com/tmc/lib/egraphcore"
)

func ExampleStore_CloneWithBudget() {
	source := egraphcore.NewStore[uint32, int]()
	source.Insert("leaf", func(uint32) int { return 7 })
	scope, _ := egraphcore.NewMemoryScope(new(exampleMemory))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	copy, err := source.CloneWithBudget(nil, budget)
	fmt.Println(copy.Class(0), err)
	// Output: 7 <nil>
}

func ExampleFactStore_CloneWithBudget() {
	source := egraphcore.NewFactStore[uint32]()
	relation, _ := source.Register(egraphcore.Schema{Name: "count", Columns: []egraphcore.Column{{Kind: egraphcore.Int64Value}}})
	cursor := relation.Snapshot()
	relation.Add([]egraphcore.Value[uint32]{egraphcore.Int64[uint32](3)}, nil)
	scope, _ := egraphcore.NewMemoryScope(new(exampleMemory))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	copy, err := source.CloneWithBudget(budget)
	rows, historyErr := copy.Get("count").AddedSince(cursor)
	fmt.Println(len(rows), err, historyErr)
	// Output: 1 <nil> <nil>
}
