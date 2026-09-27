package cudalaunch_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func ExampleInferLaunchGrid() {
	tensorShape, err := cudashape.New(1024, 2048)
	if err != nil {
		panic(err)
	}
	tileShape, err := cudashape.New(64, 128)
	if err != nil {
		panic(err)
	}
	partA, err := cudalayout.NewPartition(driver.DevicePtr(0x1000), tensorShape, tileShape, 4)
	if err != nil {
		panic(err)
	}
	partB, err := cudalayout.NewPartition(driver.DevicePtr(0x2000), tensorShape, tileShape, 4)
	if err != nil {
		panic(err)
	}

	grid, err := cudalaunch.InferLaunchGrid(cudashape.Grid3D{}, []cudalayout.Partition{partA, partB})
	if err != nil {
		panic(err)
	}
	fmt.Printf("inferred grid: %s\n", grid)
	// Output:
	// inferred grid: (16, 16, 1)
}

func ExampleConfig_AutoGrid() {
	tensorShape, err := cudashape.New(512, 512)
	if err != nil {
		panic(err)
	}
	tileShape, err := cudashape.New(32, 32)
	if err != nil {
		panic(err)
	}
	part, err := cudalayout.NewPartition(driver.DevicePtr(0x1000), tensorShape, tileShape, 4)
	if err != nil {
		panic(err)
	}

	var cfg cudalaunch.Config
	if err := cfg.AutoGrid(part); err != nil {
		panic(err)
	}
	fmt.Printf("grid: %s\n", cfg.Grid)
	// Output:
	// grid: (16, 16, 1)
}
