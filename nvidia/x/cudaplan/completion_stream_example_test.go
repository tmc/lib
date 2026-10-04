package cudaplan_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func ExampleNewStreamCompletion() {
	// An actual queued invocation passes its live stream and borrow releases.
	// A missing stream never transfers ownership to a completion.
	c, err := cudaplan.NewStreamCompletion(nil)
	fmt.Println(c == nil, err != nil)
	// Output: true true
}
