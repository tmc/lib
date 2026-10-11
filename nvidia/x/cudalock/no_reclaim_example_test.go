package cudalock_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

func ExampleWithNoReclaim() {
	// Pass this option to Acquire with a bounded context. A contended lock is
	// retained even if its owner is dead or its metadata is unavailable.
	option := cudalock.WithNoReclaim()
	fmt.Println(option != nil)
	// Output: true
}
