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

func ExampleSameArmDisagreements() {
	const giB = 1 << 30
	// Two rounds of the same arm disagreed. The later round started with a
	// neighbouring arm's 72.56 GiB of artifacts still retained.
	quiet := cudatrial.Conditions{}
	_ = quiet.Record(cudatrial.ConditionHostAvailableBytes, 100*giB)
	_ = quiet.Record(cudatrial.ConditionRetainedBytes, 0)

	pressed := cudatrial.Conditions{}
	_ = pressed.Record(cudatrial.ConditionHostAvailableBytes, 27*giB)
	_ = pressed.Record(cudatrial.ConditionRetainedBytes, 77918175232)

	receipts := []cudatrial.Receipt{
		{Name: "moe", Round: 1, Complete: true, Digest: "sha:aaa", Before: quiet},
		{Name: "moe", Round: 3, Complete: true, Digest: "sha:bbb", Before: pressed},
	}

	for _, d := range cudatrial.SameArmDisagreements(receipts, 1*giB) {
		fmt.Printf("confounded=%v\n", d.Confounded())
		for _, c := range d.Changed {
			fmt.Println(c)
		}
	}

	// Output:
	// confounded=true
	// host_available_bytes -78383153152 (107374182400 -> 28991029248)
	// retained_bytes +77918175232 (0 -> 77918175232)
}
