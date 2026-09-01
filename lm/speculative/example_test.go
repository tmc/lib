// SPDX-License-Identifier: Apache-2.0

package speculative_test

import (
	"fmt"

	"github.com/tmc/lib/lm/speculative"
)

func ExampleVerifyGreedy() {
	targetPreds := []int32{101, 102, 103, 104}
	draftTokens := []int32{101, 102}

	res := speculative.VerifyGreedy(targetPreds, draftTokens)
	fmt.Printf("Accepted: %v, Bonus: %d\n", res.AcceptedTokens, res.BonusToken)
	// Output:
	// Accepted: [101 102], Bonus: 103
}
