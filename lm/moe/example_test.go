// SPDX-License-Identifier: Apache-2.0

package moe_test

import (
	"fmt"

	"github.com/tmc/lib/lm/moe"
)

func ExampleGroupedTopKRouter() {
	router := moe.NewGroupedTopKRouter(moe.MoEConfig{
		NumRoutedExperts: 4,
		TopK:             2,
	})

	decision := router.Route([]float32{0.5, 3.2, 1.1, 4.0})
	fmt.Printf("Top experts: %v\n", decision.SelectedExperts)
	// Output:
	// Top experts: [3 1]
}
