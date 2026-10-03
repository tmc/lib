package cudasass_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func ExamplePeakGPRValue() {
	digest := strings.Repeat("1", 64)
	source := cudaevidence.Source{
		Identity: cudaevidence.Identity{ArtifactDigest: digest, Entry: "kernel", Target: "sm_90", CompilerKey: digest, WorkloadKey: digest},
		Producer: "compiled fixture", Locator: "kernel record",
	}
	value, err := cudasass.PeakGPRValue(nil, source)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(value.Record().Number == nil)
	// Output: true
}
