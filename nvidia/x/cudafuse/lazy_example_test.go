package cudafuse_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func ExampleLazyArray() {
	ctx := cudafuse.NewLazyContext()
	shape := []int{1024}

	// Build deferred lazy graph
	a := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	b := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)
	c := cudafuse.NewLazyArray(ctx, shape, cudair.Float32)

	ab := a.Add(b)
	abc := ab.Mul(c)

	// Chaining records no output allocation before the materialization boundary
	fmt.Printf("allocations before materialization: %d\n", ctx.Tracker.Snapshot().Allocations)

	// Materializing reduction compiles region source and records host allocation bookkeeping
	plan, err := abc.Sum()
	if err != nil {
		fmt.Printf("materialize error: %v\n", err)
		return
	}

	fmt.Printf("output buffers allocated: %d\n", plan.OutputBuffers)
	fmt.Printf("total allocations: %d\n", ctx.Tracker.Snapshot().Allocations)

	// Execution on known targets reports ErrHardwarePending without physical GPU
	execErr := plan.Execute("sm_121a")
	fmt.Printf("hardware pending: %t\n", errors.Is(execErr, cudaerrors.ErrHardwarePending))

	// Output:
	// allocations before materialization: 0
	// output buffers allocated: 1
	// total allocations: 1
	// hardware pending: true
}
