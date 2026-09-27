package cudalayout_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func ExampleNewPartition() {
	tensorShape, err := cudashape.New(1024, 1024)
	if err != nil {
		panic(err)
	}
	tileShape, err := cudashape.New(64, 64)
	if err != nil {
		panic(err)
	}
	part, err := cudalayout.NewPartition(driver.DevicePtr(0x1000), tensorShape, tileShape, 4)
	if err != nil {
		panic(err)
	}
	grid, err := part.Grid()
	if err != nil {
		panic(err)
	}
	fmt.Printf("grid: %s, tile stride: %d\n", grid, part.TileStride())
	// Output:
	// grid: (16, 16, 1), tile stride: 1024
}
