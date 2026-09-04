package speculative_test

import (
	"fmt"

	cudaspec "github.com/tmc/lib/lm/speculative"
)

func Example() {
	drafter := cudaspec.NewNgramDrafter(3)

	// Ingest past prompt/system context
	drafter.Feed([]int32{100, 200, 300, 400, 500, 600})

	// Current context ends with [200, 300, 400]
	draft := drafter.Draft([]int32{200, 300, 400}, 2)
	fmt.Printf("Draft tokens: %v\n", draft)

	// Target model verifies candidate sequence in one batched forward pass
	targetVerification := []int32{500, 600, 700}
	res := cudaspec.VerifyGreedy(targetVerification, draft)

	fmt.Printf("Accepted: %v, Bonus: %d, Emitted Count: %d\n", res.AcceptedTokens, res.BonusToken, res.TotalEmitted)

	// Output:
	// Draft tokens: [500 600]
	// Accepted: [500 600], Bonus: 700, Emitted Count: 3
}
