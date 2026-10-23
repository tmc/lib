package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleTileNormalizedProjection() {
	shape := egraph.TileNormalizedProjection{M: 2, N: 17, K: 65}
	expr := &egraph.Expr{Op: "tile_rmsnorm_proj", Data: shape, Children: []*egraph.Expr{
		egraph.NewLeafNode("x"), egraph.NewLeafNode("norm"), egraph.NewLeafNode("eps"), egraph.NewLeafNode("projection"),
	}}
	mod, err := egraph.ToTileIR(expr, egraph.CodegenOptions{})
	if err != nil {
		panic(err)
	}
	wire, err := mod.EncodeBytecode()
	if err != nil {
		panic(err)
	}
	grid, err := shape.Grid()
	if err != nil {
		panic(err)
	}
	fmt.Println(len(wire) > 0, grid)
	// Output: true [2 2 1]
}

func ExampleTileNormalizedProjection_Grid() {
	grid, err := (egraph.TileNormalizedProjection{M: 3, N: 65, K: 17}).Grid()
	fmt.Println(grid, err)
	// Output: [5 3 1] <nil>
}
