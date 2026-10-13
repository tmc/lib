package cudacheck_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacheck"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func Example() {
	acc := cudacheck.NewAccumulator()
	acc.FoldFloat32s([]float32{1.0, 2.0, 3.0})
	fmt.Printf("Trace hash: 0x%016x\n", acc.Hash())
	// Output:
	// Trace hash: 0x1de40d89811fe258
}

func ExampleAccumulator() {
	acc := cudacheck.NewAccumulator(cudacheck.WithPolicy(cudacheck.PolicyRawBits))
	acc.FoldFloat32(42.0)
	fmt.Printf("Count: %d\n", acc.Count())
	// Output:
	// Count: 1
}

func ExampleTriage() {
	accA := cudacheck.NewAccumulator(cudacheck.WithRecording(true))
	accB := cudacheck.NewAccumulator(cudacheck.WithRecording(true))

	accA.FoldFloat32s([]float32{1.0, 2.0, 3.0})
	accB.FoldFloat32s([]float32{1.0, 2.5, 3.0})

	diff := cudacheck.Triage(accA, accB)
	fmt.Printf("Diverged at fold #%d\n", diff.DiffIndex)
	// Output:
	// Diverged at fold #1
}

func ExampleCompare() {
	// Without a physical device, Compare reports StatusUnmeasured with the CPU reference hash.
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{1}, cudair.Float32)
	act := g.AddUnary("relu", x)
	_ = g.SetOutputs(act)

	inputs := map[string][]float32{"X": {1.0}}

	res, err := cudacheck.Compare(context.Background(), g, cudacheck.WithInputs(inputs))
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println(res.Status.String())
	// Output:
	// Unmeasured
}
