package cudasanitizer_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudasanitizer"
)

func ExampleTraceNovelty() {
	g, _ := cudasanitizer.NewTraceNovelty(cudasanitizer.GatingSimplifyTrace, 4, 8)
	first, _ := g.Evaluate([]byte{0, 1, 0, 0})
	duplicate, _ := g.Evaluate([]byte{0, 8, 0, 0})
	newEdge, _ := g.Evaluate([]byte{0, 8, 1, 0})
	fmt.Println(first, duplicate, newEdge)
	// Output:
	// true false true
}
