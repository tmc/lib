package cudamega_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamega"
)

func ExampleCompileModel() {
	arch := cudamega.ModelArchitecture{
		Dim:        64,
		HiddenDim:  128,
		NumLayers:  2,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    16,
		VocabSize:  256,
		TileSize:   32,
	}

	layers := make([]cudamega.LayerPointers, arch.NumLayers)
	scratch := cudamega.ScratchPointers{}

	queue, err := cudamega.CompileModel(arch, layers, scratch, 0)
	if err != nil {
		fmt.Printf("Compile failed: %v\n", err)
		return
	}

	fmt.Printf("Compiled Instructions: %d\n", queue.Len())
	fmt.Printf("Total Work Tiles: %d\n", queue.TotalTileCount())
	fmt.Printf("Total Dynamic Barriers: %d\n", queue.TotalBarriers)

	// Output:
	// Compiled Instructions: 33
	// Total Work Tiles: 86
	// Total Dynamic Barriers: 19
}
