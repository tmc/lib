package cudashape_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

func ExampleInferGrid_tiled() {
	tensor, _ := cudashape.New(1000, 96)
	tile, _ := cudashape.New(128, 32)
	grid, err := cudashape.InferGrid(tensor, tile)
	fmt.Println(grid, err)
	// Output: (8, 3, 1) <nil>
}

func ExampleGrid3D() {
	grid := cudashape.Grid3D{X: 8, Y: 3, Z: 1}
	fmt.Println(grid.Dims(), grid.Equal(cudashape.Grid3D{X: 8, Y: 3, Z: 1}), grid.IsZero())
	// Output: [8 3 1] true false
}
