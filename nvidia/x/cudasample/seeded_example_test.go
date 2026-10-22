// SPDX-License-Identifier: Apache-2.0

package cudasample_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasample"
)

func ExampleSampleSeededCPU() {
	logits := []float32{0.1, 3.5, 2.0, -1.0}
	params := cudasample.SequenceSampleParams{
		SeqSeed:        42,
		StepOffset:     10,
		Temperature:    0.8,
		TopK:           2,
		TopP:           0.95,
		AllowedBitmask: []uint32{(1 << 0) | (1 << 2) | (1 << 3)}, // disallow token 1
	}
	tok, _, err := cudasample.SampleSeededCPU(logits, params)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("sampled token:", tok)
	// Output:
	// sampled token: 2
}

func ExampleVerifySpeculativeSeededCPU() {
	targetProbs := [][]float32{
		{0.1, 0.8, 0.1},
		{0.0, 0.2, 0.8},
	}
	draftProbs := [][]float32{
		{0.2, 0.5, 0.3},
		{0.9, 0.1, 0.0},
	}
	draftTokens := []int32{1, 0}

	accepted, bonus, err := cudasample.VerifySpeculativeSeededCPU(targetProbs, draftProbs, draftTokens, 42, 0)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("accepted=%v bonus=%d\n", accepted, bonus)
	// Output:
	// accepted=[1] bonus=2
}
