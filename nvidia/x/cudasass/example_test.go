package cudasass_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func ExampleParseString() {
	sass := `
--- Kernel: LinearGEMV ---
/* 0x7f001230 */   LDG.E.CONSTANT.F32 R4, [R2.64] ;    // Global load
/* 0x7f001250 */   FFMA R0, R4, R5, R0 ;               // Fused multiply-add
`

	d, err := cudasass.ParseString(sass)
	if err != nil {
		panic(err)
	}

	fn := d.Function("LinearGEMV")
	fmt.Printf("Kernel: %s (instructions: %d)\n", fn.Name, len(fn.Instructions))
	first := fn.Instructions[0]
	fmt.Printf("First instruction: [0x%x] %s %s\n", first.Address, first.Opcode, first.Operands)
	// Output:
	// Kernel: LinearGEMV (instructions: 2)
	// First instruction: [0x7f001230] LDG.E.CONSTANT.F32 R4, [R2.64]
}

func ExampleDisassembly_AssertOpcodes() {
	sass := `
--- Kernel: QuantizedDotProduct ---
/* 0010 */   LDG.E.CONSTANT.F32 R4, [R2.64] ;
/* 0020 */   IDP4A.U8.U8 R0, R4, R5, R0 ;
`

	d, err := cudasass.ParseString(sass)
	if err != nil {
		panic(err)
	}

	// Verify the kernel uses hardware SIMD dot product and no scalar integer multiplication.
	err = d.AssertOpcodes("QuantizedDotProduct",
		[]string{"IDP4A"}, // Required
		[]string{"IMAD"},  // Forbidden
	)
	fmt.Println("Assert SIMD:", err == nil)

	// Attempting to assert a missing instruction reports a clear error.
	err = d.AssertOpcodes("QuantizedDotProduct", []string{"HMMA"}, nil)
	fmt.Println("Assert Tensor Core:", err)

	// Output:
	// Assert SIMD: true
	// Assert Tensor Core: cudasass: kernel "QuantizedDotProduct" missing required opcode "HMMA"
}
