//go:build cuda

// This example needs libnvfatbin installed but no GPU: nvFatbin is a host-side
// packaging library. It is behind the cuda build tag because `go test` on a
// machine without the CUDA libraries would otherwise fail rather than skip.
//
// It has not been executed. The tree it was written in has no CUDA libraries.

package nvfatbin_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvfatbin"
)

// Package PTX for one architecture and read back the fatbinary.
func Example() {
	handle, err := nvfatbin.Create([]string{"-64", "-cuda"})
	if err != nil {
		panic(err)
	}
	defer nvfatbin.Destroy(&handle)

	ptx := ".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry noop() { ret; }\n"
	// arch is the numeric part of sm_90, and must agree with .target above.
	if err := handle.AddPTXString(ptx, "90", "noop.ptx", ""); err != nil {
		panic(err)
	}

	size, err := handle.Size()
	if err != nil {
		panic(err)
	}
	fatbin, err := handle.Get()
	if err != nil {
		panic(err)
	}
	fmt.Println(size > 0 && uint(len(fatbin)) == size)
	// Output: true
}
