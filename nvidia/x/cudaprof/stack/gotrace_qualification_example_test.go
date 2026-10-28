package stack_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

// Decoded content that holds no event other than Sync, or no clock
// snapshot, is unproven: neither malformed input nor usable evidence.
func ExampleErrGoTraceUnproven() {
	err := fmt.Errorf("%w: no event other than Sync", stack.ErrGoTraceUnproven)
	fmt.Println(errors.Is(err, stack.ErrGoTraceUnproven), errors.Is(err, stack.ErrGoTraceMalformed))
	fmt.Println(err)
	// Output:
	// true false
	// go trace decoded content unproven: no event other than Sync
}
