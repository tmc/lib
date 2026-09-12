//go:build linux
// +build linux

package cufile_test

import (
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/cufile"
)

// Example is an illustrative registration sequence; replace /path/to/data
// with a file on a cuFile-supported filesystem before running it. It opens the cuFile driver and registers a file for GPUDirect
// Storage I/O. It returns early when the cuFile library is not available.
// Handle.Read and Handle.Write transfer between the file and CUDA device
// memory. The transfer path depends on the system and cuFile configuration.
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
