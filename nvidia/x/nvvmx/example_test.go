package nvvmx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvvmx"
)

func ExampleCompile() {
	// Input validation does not require the native library.
	_, err := nvvmx.Compile(nil)
	fmt.Println(err)
	// Output: nvvmx: no inputs
}

func ExampleInput() {
	input := nvvmx.Input{Name: "libdevice", Data: []byte{1, 2, 3}, Lazy: true}
	fmt.Println(input.Name, input.Lazy)
	// Output: libdevice true
}

func ExampleResult_Close() {
	var result nvvmx.Result
	fmt.Println(result.Close())
	// Output: <nil>
}
