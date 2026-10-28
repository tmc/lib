// SPDX-License-Identifier: Apache-2.0

package cudafuse_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudafuse"
)

func ExampleHCLazyState_FusedStepIntoNext() {
	// 1 token, 4 streams, hiddenDim 2.
	residuals := []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}
	state, err := cudafuse.NewHCLazyState(residuals, 1, 4, 2)
	if err != nil {
		panic(err)
	}
	preMix0, err := cudafuse.IdentityPreMix(1, 4, 0.25)
	if err != nil {
		panic(err)
	}
	x0, err := state.FusedStepIntoNext(preMix0)
	if err != nil {
		panic(err)
	}
	fmt.Printf("x0=[%.1f %.1f] pending=%v\n", x0[0], x0[1], state.Pending)
	// Output:
	// x0=[4.0 5.0] pending=false
}
