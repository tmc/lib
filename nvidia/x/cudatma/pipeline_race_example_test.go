package cudatma_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatma"
)

func ExampleAsyncCopyPipeline() {
	pipe, err := cudatma.NewAsyncCopyPipeline(2, 256)
	if err != nil {
		panic(err)
	}

	// Issue async copy into stage 0
	_ = pipe.IssueCopy(0, 256)
	_ = pipe.CommitGroup()
	_ = pipe.WaitGroup(0)

	// Consumer reads stage 0 after wait_group
	_ = pipe.ReadStage(0)

	hazards := pipe.CheckHazards()
	fmt.Printf("Hazards detected: %d\n", len(hazards))

	// Output:
	// Hazards detected: 0
}
