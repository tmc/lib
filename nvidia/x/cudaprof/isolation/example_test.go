package isolation_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
)

func Example() {
	result := isolation.Evaluate(nil, "", isolation.Binding{})
	fmt.Println(result.State)
	// Output: unchecked
}
