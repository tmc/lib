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
