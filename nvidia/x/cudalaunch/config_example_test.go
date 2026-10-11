package cudalaunch_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func ExampleConfig() {
	config := cudalaunch.Config{Grid: cudashape.Grid3D{X: 8, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), SharedMem: 1024}
	fmt.Println(config.Grid, config.SharedMem)
	// Output: (8, 1, 1) 1024
}

func ExampleLauncher_Config() {
	launcher := cudalaunch.Kernel(0).Config(cudalaunch.Config{Grid: cudashape.Grid3D{X: 8, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1)})
	fmt.Println(launcher != nil)
	// Output: true
}
