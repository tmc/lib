// SPDX-License-Identifier: Apache-2.0

package grammar_test

import (
	"fmt"

	"github.com/tmc/lib/lm/grammar"
)

func ExampleJSONConstraint() {
	c := grammar.NewJSONConstraint()
	c.Accept(`{"key": "value"}`)
	fmt.Println(c.State() == grammar.StateComplete)
	// Output:
	// true
}

func ExampleBitmaskTable() {
	// Two FSM states over a four-token vocabulary; only token 2 is allowed in
	// state 1.
	table := grammar.NewBitmaskTable(2, 4)
	table.SetValid(1, 2)

	logits := []float32{1, 2, 3, 4}
	table.ApplyMask(1, logits)
	fmt.Println(logits)
	// Output:
	// [-1e+09 -1e+09 3 -1e+09]
}
