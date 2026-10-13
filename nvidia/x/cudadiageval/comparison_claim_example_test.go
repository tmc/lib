package cudadiageval

import (
	"context"
	"fmt"
)

func ExampleCompareRegisteredPair() {
	_, err := CompareRegisteredPair(context.Background(), ValidatedRegisteredPair{})
	fmt.Println(err)
	// Output: paired comparison: invalid handle
}

func ExampleFormulateComparisonClaim() {
	_, err := FormulateComparisonClaim(context.Background(), RegisteredComparison{}, EvidenceAuthority{}, nil)
	fmt.Println(err)
	// Output: cudadiageval: claim missing required bounded fields: invalid comparison handle
}

func ExampleRegisteredComparison_Results() {
	var comparison RegisteredComparison
	fmt.Println(len(comparison.Results()))
	// Output: 0
}
