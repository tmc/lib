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

func ExampleGroupedTopKRouter_SetBiases() {
	router := moe.NewGroupedTopKRouter(moe.MoEConfig{
		NumRoutedExperts: 4,
		TopK:             2,
	})
	logits := []float32{1.0, 0.9, 0.2, 0.1}
	fmt.Println(router.Route(logits).SelectedExperts)

	// A load-balancing bias steers selection toward the idle expert 2 without
	// changing the weight any chosen expert receives.
	if err := router.SetBiases([]float32{0, 0, 1.5, 0}); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(router.Route(logits).SelectedExperts)
	// Output:
	// [0 1]
	// [2 0]
}
