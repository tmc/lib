// SPDX-License-Identifier: Apache-2.0

package cudaagreement_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
)

func ExampleVerifyBatchInvariance() {
	inputs := [][]float32{
		{1.0, 2.5, -0.5, 4.0, 0.25, -1.25, 3.0, 0.5},
		{-2.0, 0.75, 1.25, -0.5, 2.0, 1.5, -0.25, 0.125},
		{0.5, -1.5, 3.25, 0.25, -0.75, 2.25, 1.0, -0.5},
	}

	report, err := cudaagreement.VerifyBatchInvariance(
		inputs,
		[]int{1, 2, 4},
		func(batch [][]float32) ([][]float32, error) {
			out := make([][]float32, len(batch))
			for r, row := range batch {
				out[r] = []float32{cudaagreement.FixedSplitKReductionCPU(row, 4)}
			}
			return out, nil
		},
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("passed=%v identical=%d/%d mismatches=%d\n",
		report.Passed(), report.BitIdenticalCases, report.TotalCases, len(report.Mismatches))
	// Output:
	// passed=true identical=33/33 mismatches=0
}
