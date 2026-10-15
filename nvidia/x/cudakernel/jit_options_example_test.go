package cudakernel_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func ExampleJITOptions_Validate() {
	jit := cudakernel.JITOptions{OptLevelSet: true} // Explicit zero.
	fmt.Println(jit.Validate())
	opts, values := jit.DriverOptions()
	fmt.Println(len(opts), values)
	// Output:
	// <nil>
	// 1 [0]
}
