package nvtelemetry_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleNewNVMLFabricReader() {
	_, err := nvtelemetry.NewNVMLFabricReader(0, nil, nvtelemetry.RealClock{}, "node", "epoch", nvtelemetry.MIGScopeUnknown, 1000000000)
	fmt.Println(err)
	// Output: fabric reader requires a device
}

func ExampleNVMLFabricReader_Read() {
	var reader nvtelemetry.NVMLFabricReader
	_, err := reader.Read(context.Background())
	fmt.Println(err)
	// Output: invalid fabric reader
}
