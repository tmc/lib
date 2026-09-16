package cudatrial_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func Example() {
	arms := []cudatrial.Arm{
		{
			Name: "baseline",
			Run: func(ctx context.Context) (cudatrial.Receipt, error) {
				return cudatrial.Receipt{
					Complete: true,
					Digest:   "sha256:abcd",
				}, nil
			},
		},
		{
			Name: "candidate",
			Run: func(ctx context.Context) (cudatrial.Receipt, error) {
				return cudatrial.Receipt{
					Complete: true,
					Digest:   "sha256:abcd",
				}, nil
			},
		},
	}

	receipts, err := cudatrial.Interleave(context.Background(), arms, 2)
	if err != nil {
		panic(err)
	}

	exact, _ := cudatrial.SameArmExact(receipts)
	fmt.Printf("Receipts: %d\n", len(receipts))
	fmt.Printf("SameArmExact: %v\n", exact)

	for _, r := range receipts {
		fmt.Printf("Round %d %s: complete=%v digest=%s\n", r.Round, r.Name, r.Complete, r.Digest)
	}

	// Output:
	// Receipts: 4
	// SameArmExact: true
	// Round 1 baseline: complete=true digest=sha256:abcd
	// Round 1 candidate: complete=true digest=sha256:abcd
	// Round 2 candidate: complete=true digest=sha256:abcd
	// Round 2 baseline: complete=true digest=sha256:abcd
}
