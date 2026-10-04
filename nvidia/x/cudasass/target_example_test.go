package cudasass_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func ExampleDisassembly_Target() {
	d, err := cudasass.ParseString(".target sm_121\nFunction : kernel\n/*0000*/ EXIT;\n")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(d.Target)
	// Output: sm_121
}
