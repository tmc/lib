package nvjitlink_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvjitlink"
)

// Example links one PTX input into a cubin. It returns early when the
// nvJitLink library is not available. Real callers add several inputs —
// PTX from nvrtc, LTOIR, and device libraries — before calling Complete;
// the cubin can then be loaded with the driver package's ModuleLoadData.
func Example() {
	if _, _, err := nvjitlink.Version(); err != nil {
		fmt.Println("nvJitLink unavailable:", err)
		return
	}

	const ptx = `
.version 7.0
.target sm_70
.address_size 64
.visible .entry noop() { ret; }
`
	h, err := nvjitlink.Create([]string{"-arch=sm_70"})
	if err != nil {
		fmt.Println("create:", err)
		return
	}
	defer nvjitlink.Destroy(&h)

	code := []byte(ptx)
	if err := h.AddData(nvjitlink.InputPtx, unsafe.Pointer(&code[0]), uint(len(code)), "noop.ptx"); err != nil {
		fmt.Println("add:", err)
		return
	}
	if err := h.Complete(); err != nil {
		log, _ := h.GetErrorLog()
		fmt.Printf("link: %v\n%s", err, log)
		return
	}
	cubin, err := h.GetLinkedCubin()
	if err != nil {
		fmt.Println("cubin:", err)
		return
	}
	fmt.Println("linked", len(cubin), "bytes of cubin")
}
