package cudagroup_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudagroup"
)

func ExampleGenerateReductionFixtureWithCapability() {
	// Recorded launch limits, not a native execution result.
	warp, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	record, _ := cudadevicetable.Lookup("sm_121")
	f, err := cudagroup.GenerateReductionFixtureWithCapability(warp, "sm_121a", record)
	fmt.Println(f.Target, f.Launch.Block, err)
	// Output: sm_121a 32x1x1 <nil>
}

func ExampleFixture_ValidateCapability() {
	warp, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	f, _ := cudagroup.GenerateReductionFixtureForTarget(warp, "sm_121a")
	record, _ := cudadevicetable.Lookup("sm_121a") // Same device limits as sm_121.
	fmt.Println(f.ValidateCapability(record))
	err := f.ValidateCapability(cudadevicetable.Architecture{})
	fmt.Println(errors.Is(err, cudagroup.ErrCapabilityAbsent), errors.Is(err, cudagroup.ErrCapabilityUnsupported))
	// Output:
	// <nil>
	// true false
}
