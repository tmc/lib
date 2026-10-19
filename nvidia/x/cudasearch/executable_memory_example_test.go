package cudasearch

import "fmt"

func ExampleExecutableEvaluator_Close() {
	// A failed executable cleanup can be retried without closing its borrowed
	// baseline. Last-attempt metadata is retired only after cleanup succeeds.
	evaluator := new(ExecutableEvaluator)
	fmt.Println(evaluator.Close())
	fmt.Println(evaluator.Close())
	// Output:
	// <nil>
	// <nil>
}
