package ncclep_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/ncclep"
)

func Example() {
	dims := [2]uint{32, 128}
	tensor := ncclep.NewTensor()
	tensor.Ndim = 2
	tensor.Datatype = nccl.Float32
	tensor.Sizes = &dims[0]
	// Set Data or WinHdl to device storage before submitting tensor operations.
	fmt.Println(tensor.Ndim, tensor.Datatype, *tensor.Sizes)
	// Output: 2 Float32 32
}

func ExampleNewTensor() {
	v := ncclep.NewTensor()
	fmt.Println(v.Size)
	// Output: 48
}

func ExampleNewTensorAllocConfig() {
	v := ncclep.NewTensorAllocConfig()
	fmt.Println(v.Size)
	// Output: 8
}

func ExampleNewGroupConfig() {
	v := ncclep.NewGroupConfig()
	fmt.Println(v.Size)
	// Output: 112
}

func ExampleNewLayoutInfo() {
	v := ncclep.NewLayoutInfo()
	fmt.Println(v.Size)
	// Output: 48
}

func ExampleNewDispatchInputs() {
	v := ncclep.NewDispatchInputs()
	fmt.Println(v.Size)
	// Output: 32
}

func ExampleNewDispatchOutputs() {
	v := ncclep.NewDispatchOutputs()
	fmt.Println(v.Size)
	// Output: 40
}

func ExampleNewCombineInputs() {
	v := ncclep.NewCombineInputs()
	fmt.Println(v.Size)
	// Output: 32
}

func ExampleNewCombineOutputs() {
	v := ncclep.NewCombineOutputs()
	fmt.Println(v.Size)
	// Output: 24
}

func ExampleNewHandleConfig() {
	v := ncclep.NewHandleConfig()
	fmt.Println(v.Size)
	// Output: 16
}

func ExampleNewDispatchConfig() {
	v := ncclep.NewDispatchConfig()
	fmt.Println(v.Size)
	// Output: 24
}

func ExampleNewCombineConfig() {
	v := ncclep.NewCombineConfig()
	fmt.Println(v.Size)
	// Output: 20
}

func ExampleNewCompleteConfig() {
	v := ncclep.NewCompleteConfig()
	fmt.Println(v.Size)
	// Output: 8
}
