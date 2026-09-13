package tileir_test

import (
	"fmt"
	"log"
	"strings"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func Example() {
	const code = `
cuda_tile.module @demo {
  cuda_tile.entry @scale(%A: !cuda_tile.tile<ptr<f32>>) {
    %cst = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>
    cuda_tile.return
  }
}
`
	mod, err := tileir.ParseString(code)
	if err != nil {
		log.Fatal(err)
	}

	fn := mod.FindFunction("scale")
	fmt.Println("Function:", fn.Name, "Entry:", fn.Entry)
	for _, op := range fn.Operations {
		fmt.Printf("Op: %s %s\n", op.OpCode, strings.Join(op.Results, ", "))
	}

	// Output:
	// Function: scale Entry: true
	// Op: cuda_tile.constant %cst
	// Op: cuda_tile.return
}

func ExampleParseType() {
	t1, _ := tileir.ParseType("!cuda_tile.tile<16x32xbf16>")
	t2, _ := tileir.ParseType("ptr<f32>")

	fmt.Printf("%s (isTile=%v)\n", t1, t1.IsTile())
	fmt.Printf("%s (isTile=%v)\n", t2, t2.IsTile())

	// Output:
	// !cuda_tile.tile<16x32xbf16> (isTile=true)
	// !cuda_tile.ptr<f32> (isTile=false)
}
