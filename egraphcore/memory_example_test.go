package egraphcore_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/egraphcore"
)

type exampleMemory struct{ used int64 }

func (m *exampleMemory) Reserve(bytes int64) error {
	if bytes > 1<<20-m.used {
		return errors.New("memory limit")
	}
	m.used += bytes
	return nil
}
func (m *exampleMemory) Release(bytes int64) { m.used -= bytes }

func ExampleMemoryScope() {
	memory := new(exampleMemory)
	scope, err := egraphcore.NewMemoryScope(memory)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(scope.Reserve(16), scope.Bytes() > 0, scope.Err())
	scope.Close()
	fmt.Println(memory.used)
	// Output:
	// <nil> true <nil>
	// 0
}

func ExampleNewWorkBudgetWithMemory() {
	memory := new(exampleMemory)
	scope, err := egraphcore.NewMemoryScope(memory)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer scope.Close() // Matches remain owned until their last use below.
	budget, err := egraphcore.NewWorkBudgetWithMemory(1000, scope)
	if err != nil {
		fmt.Println(err)
		return
	}
	pattern := &egraphcore.Pattern{Op: "?x", Variable: true}
	matches, limited, err := egraphcore.Search(pattern, []uint32{2, 5}, budget, 10,
		func(id uint32) uint32 { return id }, func(uint32) []string { return nil },
		func(node string) string { return node }, func(string) []uint32 { return nil })
	fmt.Println(len(matches), limited, err)
	// Output: 2 false <nil>
}

func ExampleFactStore_CanonicalCopyWithBudget() {
	source := egraphcore.NewFactStore[uint32]()
	relation, _ := source.Register(egraphcore.Schema{Name: "class", Columns: []egraphcore.Column{{Kind: egraphcore.ClassValue}}})
	_, _ = relation.Add([]egraphcore.Value[uint32]{egraphcore.Class(uint32(2))}, func(id uint32) uint32 { return id })
	scope, _ := egraphcore.NewMemoryScope(new(exampleMemory))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(1000, scope)
	copy, err := source.CanonicalCopyWithBudget(func(uint32) uint32 { return 1 }, budget)
	if err != nil {
		fmt.Println(err)
		return
	}
	rows, _, _ := copy.Get("class").Query(nil, func(id uint32) uint32 { return id }, budget)
	fmt.Println(rows[0][0].Class, relation.Len())
	// Output: 1 1
}

func ExampleRelation_SchemaWithBudget() {
	relation, _ := egraphcore.NewRelation[uint32](egraphcore.Schema{Name: "count", Columns: []egraphcore.Column{{Kind: egraphcore.Int64Value}}})
	scope, _ := egraphcore.NewMemoryScope(new(exampleMemory))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(100, scope)
	schema, err := relation.SchemaWithBudget(budget)
	fmt.Println(schema.Name, len(schema.Columns), err)
	// Output: count 1 <nil>
}
