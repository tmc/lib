package numopt_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/egraph/numopt"
)

func ExampleSolveParetoFrontier() {
	tiles := []numopt.TileSensitivity{
		{TileID: "k0", HessianTrace: 1, WeightKurtosis: 3},
		{TileID: "k1", HessianTrace: 1, WeightKurtosis: 3},
		{TileID: "k2", HessianTrace: 100, WeightKurtosis: 25},
	}
	best, frontier, err := numopt.SolveParetoFrontier(tiles, "sm_100a", 0.1)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(frontier), "frontier points")
	fmt.Println(best.Assignments["k0"], best.Assignments["k1"], best.Assignments["k2"])
	// Output:
	// 14 frontier points
	// NVFP4ScaleVec4X NVFP4ScaleVec4X MXFP8E4M3
}
