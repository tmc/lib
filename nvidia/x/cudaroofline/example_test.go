package cudaroofline_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func Example() {
	// ds4 reads its weights from an mmap'd model file, so the bytes travel
	// through host page tables, not device memory.
	model, err := cudaroofline.NewModel("sm_121", cudaroofline.HostPageTableResident)
	if err != nil {
		log.Fatal(err)
	}

	// Q8_0 matrix-vector launch: 2,000,000 FLOPs over 1,000,000 bytes in
	// 7,000 nanoseconds.
	pt := model.Evaluate("ds4_q8_0_matvec", 2000000, 1000000, 7000)

	fmt.Printf("%s: %.1f TFLOPS, %.1f FLOP/B, %s (%.0f%% efficiency)\n",
		pt.Name, pt.AttainedTFLOPS, pt.Intensity, pt.Classification, pt.Efficiency*100)

	// Output:
	// ds4_q8_0_matvec: 0.3 TFLOPS, 2.0 FLOP/B, memory_bound (88% efficiency)
}

// ExampleModel_Evaluate shows a launch that beats its ceiling. The efficiency
// is reported as it was computed; the verdict is about the ceiling.
func ExampleModel_Evaluate() {
	model := cudaroofline.NewModelWithCeiling(cudaroofline.Ceiling{
		Architecture:      "sm_121a",
		Residency:         cudaroofline.HostPageTableResident,
		PeakTFLOPS:        40.0,
		PeakMemoryGBps:    162.0,
		ComputeProvenance: cudaroofline.Unverified,
		MemoryProvenance:  cudaroofline.Measured,
	})

	// 1,648,000 bytes in 10,000 nanoseconds: 164.8 GB/s.
	pt := model.Evaluate("ds4_q8_0_matvec", 3296000, 1648000, 10000)

	fmt.Printf("%.1f GB/s, %.1f%% of ceiling, %s\n",
		pt.AttainedGBps, pt.Efficiency*100, pt.Classification)

	// Output:
	// 164.8 GB/s, 101.7% of ceiling, above_ceiling
}

// ExampleLookupCeiling shows the same part reaching different bandwidth on
// different memory paths.
func ExampleLookupCeiling() {
	for _, r := range []cudaroofline.Residency{
		cudaroofline.DeviceResident,
		cudaroofline.HostPageTableResident,
	} {
		c, err := cudaroofline.LookupCeiling("sm_121a", r)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%-16s %.1f GB/s (%s)\n", c.Residency, c.PeakMemoryGBps, c.MemoryProvenance)
	}

	// Output:
	// device           239.6 GB/s (measured)
	// host_page_table  162.4 GB/s (measured)
}
