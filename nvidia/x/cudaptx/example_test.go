package cudaptx_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func ExampleModule_AssertOpcodes() {
	ptxText := `.version 8.0
.target sm_90
.address_size 64

.visible .entry VectorAdd(.param .u64 in1) {
	fma.rn.f32 %f1, %f2, %f3, %f4;
	ret;
}`

	mod, err := cudaptx.ParseString(ptxText)
	if err != nil {
		log.Fatalf("parse PTX: %v", err)
	}

	err = mod.AssertOpcodes("VectorAdd", []string{"fma"}, []string{"st.global"})
	if err != nil {
		fmt.Printf("Audit failed: %v\n", err)
	} else {
		fmt.Println("Audit passed successfully")
	}

	// Output:
	// Audit passed successfully
}

func ExampleInstructionCategory() {
	for _, mnemonic := range []string{"fma", "ld", "tcgen05", "fabric", "bra"} {
		cat := cudaptx.InstructionCategory(mnemonic)
		fmt.Printf("%s: %s (compute=%t, memory=%t)\n", mnemonic, cat, cat.IsCompute(), cat.IsMemory())
	}

	// Output:
	// fma: FloatingPoint (compute=true, memory=false)
	// ld: DataMovement (compute=false, memory=true)
	// tcgen05: TensorCore5thGen (compute=true, memory=false)
	// fabric: Fabric (compute=false, memory=true)
	// bra: ControlFlow (compute=false, memory=false)
}

func ExampleOpcodeSummary_CategoryCount() {
	ptxText := `.version 9.0
.target sm_90
.address_size 64

.visible .entry saxpy() {
	ld.global.f32 %f1, [%rd1];
	ld.global.f32 %f2, [%rd2];
	fma.rn.f32    %f3, %f1, %f0, %f2;
	st.global.f32 [%rd3], %f3;
	ret;
}`

	mod, err := cudaptx.ParseString(ptxText)
	if err != nil {
		log.Fatalf("parse PTX: %v", err)
	}

	sum, err := mod.Summary("saxpy")
	if err != nil {
		log.Fatalf("summary: %v", err)
	}

	fmt.Printf("Memory instructions: %d\n", sum.CategoryCount(cudaptx.CatDataMovement))
	fmt.Printf("Floating point: %d\n", sum.CategoryCount(cudaptx.CatFloatingPoint))
	fmt.Printf("Control flow: %d\n", sum.CategoryCount(cudaptx.CatControlFlow))

	// Output:
	// Memory instructions: 3
	// Floating point: 1
	// Control flow: 1
}

func ExampleModule_AuditArchitecture() {
	ptxText := `.version 9.0
.target sm_90
.address_size 64

.visible .entry kernel() {
	wgmma.fence.sync.aligned;
	ret;
}`

	mod, err := cudaptx.ParseString(ptxText)
	if err != nil {
		log.Fatalf("parse PTX: %v", err)
	}

	err = mod.AuditArchitecture("kernel")
	if err != nil {
		fmt.Println("Audit failed:", err)
	} else {
		fmt.Println("Architecture audit passed")
	}

	// Output:
	// Architecture audit passed
}
