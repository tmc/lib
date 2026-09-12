package cudamem_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func Example() {
	if err := cuda.Init(); err != nil {
		fmt.Printf("CUDA not available: %v\n", err)
		return
	}

	devs, err := cuda.Devices()
	if err != nil || len(devs) == 0 {
		fmt.Println("No CUDA devices found")
		return
	}

	dev := devs[0]
	slice, err := cudamem.Alloc[float32](dev, 4)
	if err != nil {
		fmt.Printf("Alloc failed: %v\n", err)
		return
	}
	defer slice.Free()

	// Copy data to device
	src := []float32{1.0, 2.0, 3.0, 4.0}
	if err := slice.CopyFromHost(src); err != nil {
		fmt.Printf("CopyFromHost failed: %v\n", err)
		return
	}

	// Copy data back from device
	dst := make([]float32, 4)
	if err := slice.CopyToHost(dst); err != nil {
		fmt.Printf("CopyToHost failed: %v\n", err)
		return
	}

	fmt.Printf("Received: %v\n", dst)
}

// An Arena gives a group of buffers one error check and one cleanup path.
func ExampleArena() {
	devs, err := cuda.Devices()
	if err != nil || len(devs) == 0 {
		fmt.Println("No CUDA devices found")
		return
	}

	a := cudamem.NewArena(devs[0])
	defer a.Free()
	x := cudamem.AllocIn[float32](a, "x", 4096)
	y := cudamem.AllocIn[float32](a, "y", 4096)
	tokens := cudamem.AllocIn[int32](a, "tokens", 1)
	if err := a.Err(); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(x.Len(), y.Len(), tokens.Len())
}
