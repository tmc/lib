package nvrtcx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func ExampleWithStandard() {
	opt := nvrtcx.WithStandard("c++17")
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithFastMath() {
	opt := nvrtcx.WithFastMath()
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithDefaultDevice() {
	opt := nvrtcx.WithDefaultDevice()
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithDefines() {
	opt := nvrtcx.WithDefines("BLOCK_SIZE=256", "DEBUG")
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithOptimizationLevel() {
	opt := nvrtcx.WithOptimizationLevel(3)
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithIncludePaths() {
	opt := nvrtcx.WithIncludePaths("/usr/local/cuda/include")
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithLineInfo() {
	opt := nvrtcx.WithLineInfo()
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithRelocatableDeviceCode() {
	opt := nvrtcx.WithRelocatableDeviceCode(true)
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithMaxRegisters() {
	opt := nvrtcx.WithMaxRegisters(64)
	fmt.Println(opt != nil)
	// Output: true
}
