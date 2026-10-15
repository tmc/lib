package cudaptx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func ExampleTargetDeclaration() {
	m, err := cudaptx.ParseString("// .target sm_70\n.target sm_90, debug\n.target sm_90a\n")
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, declaration := range m.TargetDeclarations {
		fmt.Println(declaration.Line, declaration.Target)
	}
	fmt.Println("last:", m.Target)
	// Output:
	// 2 sm_90, debug
	// 3 sm_90a
	// last: sm_90a
}
