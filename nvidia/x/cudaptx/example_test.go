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
