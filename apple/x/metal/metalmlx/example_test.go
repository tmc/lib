package metalmlx_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/apple/x/metal/metalmlx"
)

const source = `
#include <metal_stdlib>
kernel void scale(device float* out [[buffer(0)]],
                  constant float& by [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
  out[i] = float(i) * by;
}`

// Example compiles a kernel, allocates from the caching allocator, and
// dispatches through the command encoder.
func Example() {
	d, err := metalmlx.GetDevice()
	if err != nil {
		fmt.Println("no Metal device")
		return
	}
	lib, err := d.LibraryFromSource("example", metalmlx.CompileOptions{},
		func() (string, error) { return source, nil })
	if err != nil {
		fmt.Println("compile:", err)
		return
	}
	kernel, err := d.Kernel(lib, "scale")
	if err != nil {
		fmt.Println("kernel:", err)
		return
	}

	e, err := metalmlx.NewCommandEncoder(d)
	if err != nil {
		fmt.Println("encoder:", err)
		return
	}
	defer e.Close()

	const n = 8
	a := d.Allocator()
	out, err := a.Malloc(n * 4)
	if err != nil {
		fmt.Println("malloc:", err)
		return
	}
	defer a.Free(out)

	by := float32(2)
	e.SetComputePipelineState(kernel)
	e.SetOutputBuffer(out, 0, 0)
	arg, err := metalmlx.ValueBytes(&by)
	if err != nil {
		fmt.Println("argument:", err)
		return
	}
	e.SetBytes(arg, 1)
	e.DispatchThreads(metalmlx.Size{Width: n}, metalmlx.Size{Width: n})
	if err := e.Synchronize(); err != nil {
		fmt.Println("synchronize:", err)
		return
	}

	fmt.Println(unsafe.Slice((*float32)(out.Contents()), n))
	// Output: [0 2 4 6 8 10 12 14]
}
