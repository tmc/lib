package tileirdisasm_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/tileirdisasm"
)

func ExampleParseString() {
	tileIR := `entry @vector_add(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) {
  %c0 = constant <i32: 0> : tile<i32> loc(#loc)
  return loc(#loc)
}
#loc = loc("vector_add.cu":12:4)
`

	mod, err := tileirdisasm.ParseString(tileIR)
	if err != nil {
		panic(err)
	}

	fn := mod.FindFunction("vector_add")
	fmt.Printf("Function: %s, Params: %d, Ops: %d\n", fn.Name, len(fn.Params), len(fn.Operations))
	fmt.Printf("First Op: %s at %s\n", fn.Operations[0].OpCode, fn.Operations[0].Loc)

	// Output:
	// Function: vector_add, Params: 2, Ops: 2
	// First Op: constant at vector_add.cu:12:4
}
