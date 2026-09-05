//go:build cuda

package nvjitlink_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/nvjitlink"
	"unsafe"
)

func Example() {
	linker, err := nvjitlink.Create([]string{"-arch=sm_75"})
	if err != nil {
		panic(err)
	}
	defer nvjitlink.Destroy(&linker)
	ptx := []byte(".version 8.0\n.target sm_75\n.address_size 64\n.visible .entry noop() { ret; }\n\x00")
	if err := linker.AddData(nvjitlink.InputPtx, unsafe.Pointer(unsafe.SliceData(ptx)), uint(len(ptx)), "noop.ptx"); err != nil {
		panic(err)
	}
	if err := linker.Complete(); err != nil {
		panic(err)
	}
	cubin, err := linker.GetLinkedCubin()
	if err != nil {
		panic(err)
	}
	fmt.Println(len(cubin) >= 4 && string(cubin[:4]) == "\x7fELF")
	// Output: true
}
