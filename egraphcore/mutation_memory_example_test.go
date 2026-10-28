package egraphcore_test

import (
	"fmt"
	"github.com/tmc/lib/egraphcore"
)

func ExampleStore_InsertWithBudget() {
	m := new(exampleMemory)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	s := egraphcore.NewStore[uint32, int]()
	id, err := s.InsertWithBudget("a", func(uint32) (int, error) { return 7, nil }, b)
	fmt.Println(id, s.Class(id), err)
	// Output: 0 7 <nil>
}

func ExampleStore_AddMemoCandidateIfWithBudget() {
	m := new(exampleMemory)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	s := egraphcore.NewStore[uint32, int]()
	added, err := s.AddMemoCandidateIfWithBudget("a", 0, func(a, c uint32) bool { return a == c }, b)
	fmt.Println(added, err)
	// Output: true <nil>
}

func ExampleFindMemoCandidateIterWithBudget() {
	m := new(exampleMemory)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	nodes := []string{"a"}
	id, found, err := egraphcore.FindMemoCandidateIterWithBudget(func(visit func(uint32) bool) { visit(3) }, "a", func(id uint32) uint32 { return id }, func(uint32) []string { return nodes }, func(s string) (string, error) { return s, nil }, func(a, c string) bool { return a == c }, b)
	fmt.Println(id, found, err)
	// Output: 3 true <nil>
}

func ExampleFactStore_RecanonicalizeWithBudget() {
	m := new(exampleMemory)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	s := egraphcore.NewFactStore[uint32]()
	r, _ := s.Register(egraphcore.Schema{Name: "r", Columns: []egraphcore.Column{{Kind: egraphcore.ClassValue}}})
	r.Add([]egraphcore.Value[uint32]{egraphcore.Class[uint32](2)}, func(id uint32) uint32 { return id })
	fmt.Println(s.RecanonicalizeWithBudget(func(uint32) uint32 { return 0 }, b), r.Rows()[0][0].Class)
	// Output: <nil> 0
}

type mutationExampleGraph struct{}

func (mutationExampleGraph) Find(id uint32) uint32                          { return id }
func (mutationExampleGraph) AddChecked(string) (uint32, error)              { return 0, nil }
func (mutationExampleGraph) UnionChecked(a, c uint32) (uint32, bool, error) { return a, a != c, nil }
func (mutationExampleGraph) RebuildChecked() error                          { return nil }

func ExampleDriveCongruenceWithBudget() {
	m := new(exampleMemory)
	scope, _ := egraphcore.NewMemoryScope(m)
	defer scope.Close()
	b, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	// No backend scopes: the common driver observes stability in one pass.
	report, err := egraphcore.DriveCongruenceWithBudget(mutationExampleGraph{}, 1, func(int, int) (int, []egraphcore.OwnedNode[uint32, string], bool, error) { return 0, nil, false, nil }, func(n string) (string, error) { return n, nil }, func(a, c string) bool { return a == c }, nil, func(int, map[uint32][]string) (bool, error) { return false, nil }, b)
	fmt.Println(report.Passes, err)
	// Output: 1 <nil>
}
