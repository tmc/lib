package cudaready_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaready"
)

func Example() {
	for _, fn := range []string{"signal", "wait"} {
		fmt.Println(fn, strings.Contains(cudaready.Header, "void "+fn+"("))
	}
	// Output:
	// signal true
	// wait true
}
