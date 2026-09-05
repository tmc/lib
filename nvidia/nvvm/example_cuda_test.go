//go:build cuda

package nvvm_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/nvvm"
	"os"
	"unsafe"
)

func Example() {
	program, err := nvvm.CreateProgram()
	if err != nil {
		panic(err)
	}
	defer nvvm.DestroyProgram(&program)
	fmt.Println(program != 0)
	// Output: true
}

func ExampleProgram_Compile() {
	program, err := nvvm.CreateProgram()
	if err != nil {
		panic(err)
	}
	defer nvvm.DestroyProgram(&program)
	for _, name := range []string{os.Getenv("CUDA_LIBDEVICE"), "testdata/sin.ll"} {
		data, err := os.ReadFile(name)
		if err != nil {
			panic(err)
		}
		if err := program.AddModuleTo(unsafe.Pointer(unsafe.SliceData(data)), uint(len(data)), name); err != nil {
			panic(err)
		}
	}
	if err := program.Compile([]string{"-arch=compute_120", "-gen-lto"}); err != nil {
		panic(err)
	}
	lto, err := program.GetCompiledResult()
	if err != nil {
		panic(err)
	}
	fmt.Println(len(lto) > 0)
	// Output: true
}
