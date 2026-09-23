package cudaliveness_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
)

// A probe reporting no defects has either found none or not run. Assert
// refuses to call that clean without controls that say which.
func Example() {
	err := cudaliveness.Assert(
		cudaliveness.Planted{Site: "score_tile", Reported: 1, Ran: true},
		cudaliveness.Adjacent{Site: "score_tile+1", Reported: 0, Ran: true},
		cudaliveness.Counted{Site: "score_tile", Want: 2, Got: 2,
			Why: "warm-up launch: one read, one write", Ran: true},
		cudaliveness.Refused{Site: "sm_90 object", Rejected: true, Ran: true},
		cudaliveness.Undisturbed{Site: "logits", ObserverOff: "d0", ObserverOn: "d0", Ran: true},
	)
	fmt.Println(err)
	// Output: <nil>
}

// A clean report with no control is the failure this package exists to catch.
func ExampleAssert() {
	fmt.Println(cudaliveness.Assert())
	// Output: cudaliveness: no controls: a clean report with no control is not a measurement
}

// A control nobody ran is the absence of a measurement, not a negative one.
func ExampleAssert_unrun() {
	err := cudaliveness.Assert(cudaliveness.Planted{Site: "score_tile"})
	fmt.Println(errors.Is(err, cudaerrors.ErrHardwarePending), errors.Is(err, cudaliveness.ErrNotLive))
	// Output: true false
}

// A count is evidence only with its derivation. Matching numbers nobody can
// account for are refused.
func ExampleCounted() {
	unexplained := cudaliveness.Counted{Site: "score_tile", Want: 2, Got: 2, Ran: true}
	fmt.Println(unexplained.Check())
	// Output: instrumentation count at score_tile: want 2: cudaliveness: expected count has no derivation
}
