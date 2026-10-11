package cudabpf_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudabpf"
)

func Example() {
	r := cudabpf.Reorder{Destination: cudabpf.DestinationUnused, Position: cudabpf.PositionTail}
	fmt.Println(r.Valid())
	fmt.Println(r)
	// Output:
	// <nil>
	// unused/tail
}

func ExampleBudget() {
	b := cudabpf.NewBudget(2)
	fmt.Println(b.Step(2))
	fmt.Println(b.Used())
	// Output:
	// <nil>
	// 2
}
