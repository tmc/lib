package nvrtcx_test

import (
	"context"
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

func ExampleCompileFlags() {
	flags, err := nvrtcx.CompileFlags(
		nvrtcx.WithStandard("c++17"),
		nvrtcx.WithFastMath(),
		nvrtcx.WithOptimizationLevel(3),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(flags)
	// Output: [-std=c++17 --use_fast_math -O3 --generate-line-info]
}

func ExampleWithFastCompile() {
	opt := nvrtcx.WithFastCompile("max")
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithRandomSeed() {
	opt := nvrtcx.WithRandomSeed(12345)
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithPCH() {
	opt := nvrtcx.WithPCH()
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithBundledHeaders() {
	opt := nvrtcx.WithBundledHeaders()
	fmt.Println(opt != nil)
	// Output: true
}

func ExampleWithContext() {
	ctx := context.Background()
	opt := nvrtcx.WithContext(ctx)
	fmt.Println(opt != nil)
	// Output: true
}
