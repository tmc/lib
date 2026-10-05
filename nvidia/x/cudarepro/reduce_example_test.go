package cudarepro

import (
	"context"
	"fmt"
)

func ExampleReduce() {
	// This fixture demonstrates reduction without a device. A native adapter
	// must supply actual campaign run records, not these simulated receipts.
	original := reductionCase()
	r, err := Reduce(context.Background(), original, mismatchRecord(0),
		ReduceOptions{Repeats: 3, MaxCandidates: 100}, hostReplay)
	fmt.Println(len(original.Inputs), len(r.Reduced.Inputs))
	fmt.Println(original.Block[0], r.Reduced.Block[0])
	fmt.Println(r.Baseline.Reproductions, len(r.Baseline.Runs), err)
	// Output:
	// 4 1
	// 64 2
	// 3 3 <nil>
}
