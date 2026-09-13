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

func ExampleLookupDoc() {
	doc := tileir.LookupDoc("cuda_tile.mma_f")
	fmt.Printf("%s: %s [%s]\n", doc.OpCode, doc.Category, doc.HardwareLowering)

	// Output:
	// cuda_tile.mma_f: Tensor Core MMA [wgmma.mma_async / mma.sync]
}

func ExampleSplitOperands() {
	ops := tileir.SplitOperands("%0, !cuda_tile.tile<64x64xf16>, {inbounds = true}")
	for _, op := range ops {
		fmt.Println(op)
	}

	// Output:
	// %0
	// !cuda_tile.tile<64x64xf16>
	// {inbounds = true}
}

func ExampleModule_AssertOpcodes() {
	const code = `
cuda_tile.module @gemm {
  cuda_tile.entry @matmul(%A: !cuda_tile.tile<ptr<f16>>, %B: !cuda_tile.tile<ptr<f16>>, %C: !cuda_tile.tile<ptr<f32>>) {
    %tok0 = cuda_tile.make_token : !cuda_tile.token
    %vA = cuda_tile.load_view_tko %A, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %vB = cuda_tile.load_view_tko %B, %tok0 : (!cuda_tile.tile<ptr<f16>>, !cuda_tile.token) -> !cuda_tile.tile<64x64xf16>
    %cst = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %acc = cuda_tile.mma_f %vA, %vB, %cst : (!cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf16>, !cuda_tile.tile<64x64xf32>) -> !cuda_tile.tile<64x64xf32>
    cuda_tile.return
  }
}
`
	mod, _ := tileir.ParseString(code)
	err := mod.AssertOpcodes("matmul", []string{"cuda_tile.mma_f"}, tileir.ForbidScalarMemoryFallbacks)
	fmt.Println("Audit passed:", err == nil)

	// Output:
	// Audit passed: true
}
