//go:build linux
// +build linux

package cufile_test

import (
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/cufile"
)

// Example opens the cuFile driver and registers a file for GPUDirect
// Storage I/O. It returns early when the cuFile library is not available.
// Handle.Read and Handle.Write transfer directly between the file and
// CUDA device memory, e.g. from the driver package's MemAlloc.
func Example() {
	if err := cufile.DriverOpen(); err != nil {
		fmt.Println("cuFile unavailable:", err)
		return
	}
	defer cufile.DriverClose()

	f, err := os.Open("/path/to/data")
	if err != nil {
		fmt.Println("open:", err)
		return
	}
	defer f.Close()

	descr := cufile.Descr{
		Type:   cufile.HandleTypeOpaqueFd,
		Handle: [1]uint64{uint64(f.Fd())},
	}
	h, err := cufile.NewHandleRegister(&descr)
	if err != nil {
		fmt.Println("register:", err)
		return
	}
	defer h.Deregister()
	fmt.Println("file registered for GPUDirect Storage")
}
