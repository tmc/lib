package nccl_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nccl"
)

// Example queries the NCCL version and creates a single-process
// communicator spanning the local GPUs. It returns early when the NCCL
// library is not available.
func Example() {
	version, err := nccl.GetVersion()
	if err != nil {
		fmt.Println("NCCL unavailable:", err)
		return
	}
	fmt.Println("NCCL version:", version)

	devs := []int32{0}
	comms := make([]nccl.Comm, len(devs))
	err = nccl.CommInitAll(&comms[0], int32(len(devs)), &devs[0])
	if err != nil {
		fmt.Println("comm init:", err)
		return
	}
	defer comms[0].Destroy()

	n, _ := comms[0].Count()
	fmt.Println("communicator ranks:", n)
}
