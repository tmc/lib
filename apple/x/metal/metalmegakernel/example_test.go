package metalmegakernel_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/apple/x/metal/metalmegakernel"
)

func Example() {
	cfg := metalmegakernel.LayerConfig{
		HiddenDim:       64,
		IntermediateDim: 128,
		RmsNormEps:      1e-5,
	}

	msl := metalmegakernel.GenerateLayerMegakernelMSL(cfg)
	hasKernel := strings.Contains(msl, "MetalLayerMegakernel")
	fmt.Printf("Generated MSL contains Megakernel: %t\n", hasKernel)

	// Output:
	// Generated MSL contains Megakernel: true
}
