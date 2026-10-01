package nvrtcx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func ExampleSpliceTemplate() {
	tmpl := `{{FORWARD_DEVICE_FUNCTION}}
__global__ void {{KERNEL_NAME}}() {}`

	subs := map[string]string{
		"FORWARD_DEVICE_FUNCTION": "__device__ void fwd() {}",
		"KERNEL_NAME":             "example_kernel",
	}

	result, err := nvrtcx.SpliceTemplate(tmpl, subs)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	fmt.Printf("%t\n", len(result) > 0)
	// Output:
	// true
}
