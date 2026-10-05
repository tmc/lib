package cudaoutputcoverage

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func Example() {
	t := recorded(3)
	t.ReadbackA[2], t.ReadbackB[2] = t.FillA, t.FillB
	r, err := ClassifyTwoFill(t)
	fmt.Println(r.Unwritten, err)
	// Output: [{2 3}] <nil>
}

func ExampleClassifyTwoFill() {
	t := recorded(1003)
	for i := 768; i < 1003; i++ {
		t.ReadbackA[i], t.ReadbackB[i] = t.FillA, t.FillB
	}
	r, err := ClassifyTwoFill(t)
	fmt.Println(r.Unwritten, err)
	// Output: [{768 1003}] <nil>
}

func ExampleObserve() {
	_, _, err := Observe(context.Background(), recorded(3), nil)
	fmt.Println(errors.Is(err, cudaerrors.ErrHardwarePending))
	// Output: true
}

func ExampleCheckOutput() {
	t := recorded(3)
	t.ReadbackA[2], t.ReadbackB[2] = t.FillA, t.FillB
	r, err := CheckOutput(t, t.Reference, policy(t))
	fmt.Println(r.Written.Unwritten, r.NumericalMismatch == nil, err)
	// Output: [{2 3}] true <nil>
}

func ExampleContract_Validate() {
	c := Contract{Buffer: "out", Elements: 4, Domain: []Range{{1, 3}}, Kind: AtLeastOnce, Deterministic: true}
	fmt.Println(c.Validate())
	// Output: <nil>
}
