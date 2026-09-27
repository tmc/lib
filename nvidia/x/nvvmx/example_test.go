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

func ExampleCompileWithOptions() {
	// Options are validated before native library calls.
	_, err := nvvmx.CompileWithOptions(nil, nvvmx.WithArch("compute_80"), nvvmx.WithOpt(3))
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

func ExampleInferAddressSpaces() {
	ir := `
@smem = internal addrspace(3) global [16 x float] undef, align 4

define void @example() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
`
	_, report, err := nvvmx.InferAddressSpaces([]byte(ir))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	info := report.Slots["%slot"]
	fmt.Printf("%s: state=%s effective=addrspace(%d)\n", info.Name, info.State, info.Effective)
	// Output: %slot: state=shared effective=addrspace(3)
}
