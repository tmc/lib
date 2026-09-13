package cudaoccupancy_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

func Example() {
	// A sample compiled kernel allocating 40 registers and 1024 bytes static shared memory
	kernel := cudaoccupancy.KernelResource{
		Name:            "matvec_q8_0",
		Registers:       40,
		StaticSharedMem: 1024,
	}

	// Calculate theoretical occupancy on Blackwell (sm_121) with 256 threads/block
	report, err := kernel.Calculate("sm_121", 256, 0)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s: %d/%d warps (%.0f%%) limited by %s\n",
		report.Architecture, report.TheoreticalWarps, report.MaxWarps, report.Occupancy*100, report.LimitingFactor)

	// Add 32 KiB dynamic shared memory
	reportSmem, err := kernel.Calculate("sm_121", 256, 32768)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s (+32KB smem): %d/%d warps (%.0f%%) limited by %s\n",
		reportSmem.Architecture, reportSmem.TheoreticalWarps, reportSmem.MaxWarps, reportSmem.Occupancy*100, reportSmem.LimitingFactor)

	// Output:
	// sm_121: 48/48 warps (100%) limited by none
	// sm_121 (+32KB smem): 24/48 warps (50%) limited by shared_memory
}
