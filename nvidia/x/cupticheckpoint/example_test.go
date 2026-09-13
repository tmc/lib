package cupticheckpoint_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cupticheckpoint"
)

func ExampleOptimizations_String() {
	fmt.Println(cupticheckpoint.OptNone)
	fmt.Println(cupticheckpoint.OptTransfer)
	// Output:
	// None
	// Transfer
}
