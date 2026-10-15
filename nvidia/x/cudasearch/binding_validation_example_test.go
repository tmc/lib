package cudasearch_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasearch"
)

func ExampleWorkloadBindings_Validate() {
	b := cudasearch.WorkloadBindings{Name: "weighted-input", InputNames: []string{"input", "weight"},
		InputIdentities: map[string]string{"input": "input-v1"}, WeightIdentities: map[string]string{"weight": "weight-v1"},
		NumericalPolicy: cudasearch.NumericalPolicyExact}
	fmt.Println(b.Validate())
	delete(b.WeightIdentities, "weight")
	fmt.Println(errors.Is(b.Validate(), cudasearch.ErrMissingBinding))
	// Output:
	// <nil>
	// true
}
