package nvfatbinx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvfatbinx"
)

func ExampleBuild() {
	_, err := nvfatbinx.Build(nil)
	fmt.Println(err)
	// Output: nvfatbinx: no inputs
}

func ExampleInput() {
	input := nvfatbinx.Input{Kind: nvfatbinx.PTX, Name: "kernel", Arch: "80"}
	fmt.Println(input.Name, input.Arch)
	// Output: kernel 80
}

func ExampleResult_Close() {
	var result nvfatbinx.Result
	fmt.Println(result.Close())
	// Output: <nil>
}
