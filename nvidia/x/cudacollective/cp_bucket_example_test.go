package cudacollective_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacollective"
)

func ExamplePrepareCPMirrorIndices() {
	perRank, gathered, err := cudacollective.ComputeCPBucketBytes(cudacollective.CPBucketSpec{
		MaxBatchTokens: 1000,
		CPSize:         4,
		HiddenDim:      128,
		ElemBytes:      2,
		PadMultiple:    64,
	})
	if err != nil {
		panic(err)
	}
	plan, err := cudacollective.PrepareCPMirrorIndices(16, 4, 1)
	if err != nil {
		panic(err)
	}
	fmt.Printf("perRank=%d gathered=%d local=%v mirror=%v\n",
		perRank, gathered, plan.LocalIndices, plan.MirrorIndices)
	// Output:
	// perRank=65536 gathered=262144 local=[2 3] mirror=[12 13]
}
