package cudaoccupancy_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

func ExampleKernelResource_OccupancyValue() {
	digest := strings.Repeat("1", 64)
	source := cudaevidence.Source{
		Identity: cudaevidence.Identity{ArtifactDigest: digest, Entry: "kernel", Target: "sm_90", CompilerKey: digest, WorkloadKey: digest},
		Producer: "compiled fixture", Locator: "kernel record",
	}
	k := cudaoccupancy.KernelResource{Name: "kernel", Registers: 32}
	value, err := k.OccupancyValue(source, 128, 0, true)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(value.Record().Class)
	// Output: derived
}
