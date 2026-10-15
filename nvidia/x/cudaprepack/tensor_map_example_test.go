package cudaprepack_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprepack"
)

func ExampleBoxDescriptor_ValidateTensorMap() {
	b := cudaprepack.BoxDescriptor{
		Rank: 2, DataType: cudaprepack.TypeFloat32, GlobalAddress: 4096,
		GlobalDims: [3]uint64{256, 8, 1}, GlobalStrides: [3]int64{1024, 0, 0},
		BoxDims: [3]uint32{256, 4, 1}, ElementStrides: [3]uint32{1, 1, 1},
	}
	fmt.Println(b.ValidateTensorMap(cudaprepack.ArchSM121a, cudaprepack.TensorMapTiledLoad))
	fmt.Println(b.ValidateTensorMap(cudaprepack.ArchSM121a, cudaprepack.TensorMapTiledStore))
	// Output:
	// <nil>
	// <nil>
}
