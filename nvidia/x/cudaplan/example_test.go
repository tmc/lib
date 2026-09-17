package cudaplan_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func ExampleCompile() {
	// 1. Build a computation graph with cudair
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2, 4}, cudair.Float32)
	w := g.AddInput("W", []int{4, 2}, cudair.Float32)
	xw := g.AddMatMul(x, w)
	if err := g.SetOutputs(xw); err != nil {
		log.Fatalf("SetOutputs: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		log.Fatalf("Snapshot: %v", err)
	}

	// 2. Compile into an executable plan
	plan, err := cudaplan.Compile(snap, cudaplan.WithPeakWorkspaceBudget(64*1024*1024))
	if err != nil {
		log.Fatalf("Compile: %v", err)
	}
	defer plan.Close()

	// 3. Prepare bindings
	bindings := cudaplan.NewBindings()
	_ = bindings.SetInput("X", driver.DevicePtr(0x1000), 2*4*4, nil)
	_ = bindings.SetInput("W", driver.DevicePtr(0x2000), 4*2*4, nil)
	_ = bindings.SetOutput("matmul_2", driver.DevicePtr(0x3000), 2*2*4, nil)

	// Execution requires an authorized CUDA device and stream.
	fmt.Println("Plan compiled; native execution requires a CUDA device and stream")
	// Output:
	// Plan compiled; native execution requires a CUDA device and stream
}
