package cudaheadroom_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaheadroom"
)

func Example() {
	_, err := cudaheadroom.Inspect(nil)
	fmt.Println(err)
	// Output: cudaheadroom: kernel has no device
}
func ExampleInspect() {
	_, err := cudaheadroom.Inspect(nil)
	fmt.Println(err)
	// Output: cudaheadroom: kernel has no device
}
func ExampleCheckPlan() {
	_, err := cudaheadroom.CheckPlan(nil, nil, 0)
	fmt.Println(err)
	// Output: cudaheadroom: kernel has no device
}
func ExamplePreparePlan() {
	_, err := cudaheadroom.PreparePlan(nil, nil, 0)
	fmt.Println(err)
	// Output: cudaheadroom: kernel has no device
}
func ExampleBudget() {
	var b cudaheadroom.Budget
	fmt.Println(b.Reserved == nil)
	// Output: true
}
