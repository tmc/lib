package cudaroofline_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func Example() {
	model, err := cudaroofline.NewModel("sm_121")
	if err != nil {
		log.Fatal(err)
	}

	// Saturated Q8_0 matrix-vector launch:
	// 2,000,000 FLOPs, 1,000,000 bytes transferred in 4,000 nanoseconds.
	pt := model.Evaluate("ds4_q8_0_matvec", 2000000, 1000000, 4000)

	fmt.Printf("%s: %.1f TFLOPS, %.1f FLOP/B, %s (%.0f%% efficiency)\n",
		pt.Name, pt.AttainedTFLOPS, pt.Intensity, pt.Classification, pt.Efficiency*100)

	// Output:
	// ds4_q8_0_matvec: 0.5 TFLOPS, 2.0 FLOP/B, memory_bound (92% efficiency)
}
