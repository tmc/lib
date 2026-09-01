package nvtx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtx"
)

func Example() {
	// Nested timeline range
	nvtx.RangePush("TransformerForward")
	// Execute forward pass...
	nvtx.RangePop()

	// Instantaneous event marker
	nvtx.Mark("CheckpointReached")

	fmt.Println("NVTX operations completed successfully")
	// Output:
	// NVTX operations completed successfully
}

func ExampleDomain() {
	// Isolate instrumentation to a custom domain
	domain := nvtx.CreateDomain("DeepSeekInference")
	defer domain.Destroy()

	domain.RangePush("MoEExpertDispatch")
	// Execute dispatch...
	domain.RangePop()

	fmt.Println("Domain operations completed successfully")
	// Output:
	// Domain operations completed successfully
}
