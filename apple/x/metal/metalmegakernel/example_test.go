package metalmegakernel_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/apple/x/metal/metalmegakernel"
)

func Example() {
	cfg := metalmegakernel.FusedFFNConfig{
		HiddenDim:       64,
		IntermediateDim: 128,
		RMSNormEpsilon:  1e-5,
	}

	msl, err := metalmegakernel.GenerateFusedFFNMSL(cfg)
	fmt.Println(err)
	fmt.Println(strings.Contains(msl, "MetalFusedFFN"))
	fmt.Println(cfg.ScratchBytes())

	// Output:
	// <nil>
	// true
	// 768
}
