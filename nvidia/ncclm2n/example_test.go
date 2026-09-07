package ncclm2n_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/ncclm2n"
)

func Example() {
	shape := [1]uint{128}
	placement := [1]int32{ncclm2n.Replicate}
	tensor := ncclm2n.NewDistTensor()
	tensor.Ndims = 1
	tensor.Dtype = nccl.Float32
	tensor.LocalShape = &shape[0]
	tensor.Placements = &placement[0]
	// Set Mesh and DataPtr before submitting reshard operations.
	fmt.Println(tensor.Dtype, *tensor.LocalShape, *tensor.Placements)
	// Output: Float32 128 -1
}

func ExampleNewMesh() {
	v := ncclm2n.NewMesh()
	fmt.Println(v.Size)
	// Output: 32
}

func ExampleNewDistTensor() {
	v := ncclm2n.NewDistTensor()
	fmt.Println(v.Size)
	// Output: 56
}

func ExampleNewConfig() {
	v := ncclm2n.NewConfig()
	fmt.Println(v.Size)
	// Output: 24
}
