package nvvm_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvvm"
)

// Example compiles a small NVVM IR module to PTX. It returns early when the
// NVVM library is not available. The PTX can then be loaded with the driver
// package, or linked with libdevice and other inputs first.
func Example() {
	if _, _, err := nvvm.Version(); err != nil {
		fmt.Println("NVVM unavailable:", err)
		return
	}

	const module = `
target datalayout = "e-p:64:64:64-i64:64:64-f64:64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @zero(float* %out) {
  store float 0.0, float* %out
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{void (float*)* @zero, !"kernel", i32 1}
`
	prog, err := nvvm.CreateProgram()
	if err != nil {
		fmt.Println("create:", err)
		return
	}
	defer nvvm.DestroyProgram(&prog)

	ir := []byte(module)
	if err := prog.AddModuleTo(unsafe.Pointer(&ir[0]), uint(len(ir)), "zero.ll"); err != nil {
		fmt.Println("add module:", err)
		return
	}
	if err := prog.Compile([]string{"-arch=compute_70"}); err != nil {
		log, _ := prog.GetProgramLog()
		fmt.Printf("compile: %v\n%s", err, log)
		return
	}
	ptx, err := prog.GetCompiledResult()
	if err != nil {
		fmt.Println("result:", err)
		return
	}
	fmt.Println("compiled", len(ptx), "bytes of PTX")
}
