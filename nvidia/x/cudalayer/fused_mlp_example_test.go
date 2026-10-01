package cudalayer_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalayer"
)

func ExampleNewFusedMLP() {
	// Construct a 3-layer width-64 MLP
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	unit, err := cudair.EmitTree(mlp, "fused_mlp_eval")
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	fmt.Printf("num symbols: %d\n", len(unit.Symbols))
	// Output:
	// num symbols: 4
}
