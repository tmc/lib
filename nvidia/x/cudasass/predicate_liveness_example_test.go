package cudasass_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A guarded write may not happen, so the value it would overwrite stays live
// before it.
func ExampleLiveness_predicated() {
	b := cudasass.BasicBlock{Instructions: []cudasass.Instruction{
		{Address: 0x0, Opcode: "NOP"},
		{Address: 0x10, Predicate: "@P0", Opcode: "IADD", Operands: "R4, R2, UR6"},
	}}
	r4 := cudasass.Register{Class: cudasass.GPR, Num: 4}
	live, err := cudasass.Liveness(b, cudasass.LiveSet{r4: true}, cudasass.SM121a)
	if err != nil {
		log.Fatal(err)
	}
	for i, s := range live {
		fmt.Printf("after %d: %v\n", i, s.Registers())
	}
	// Output:
	// after 0: [R2 R4 P0 UR6]
	// after 1: [R4]
}
