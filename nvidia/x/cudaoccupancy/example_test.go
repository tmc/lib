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

func ExampleKernelResource_Evidence() {
	kernel := cudaoccupancy.KernelResource{
		Name:            "gemv_q8",
		Registers:       48,
		StaticSharedMem: 2048,
		StackFrameBytes: 0,
	}

	ev, err := kernel.Evidence("sm_121", 256, 1024)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s: %d regs, %d bytes local, %d warps/SM\n",
		ev.Kernel, ev.Registers, ev.LocalMemory, ev.Occupancy.TheoreticalWarps)
	// Output:
	// gemv_q8: 48 regs, 0 bytes local, 40 warps/SM
}

func ExampleKernelResource_Calculate_tensorMemory() {
	// A tcgen05 GEMM that allocates 256 TMEM columns per CTA fits two CTAs
	// on an SM of 512 columns, whatever its registers allow.
	kernel := cudaoccupancy.KernelResource{Name: "umma_gemm", Registers: 64, TMEMColumns: 256}
	report, err := kernel.Calculate("sm_103a", 128, 0)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(report)
	// Output:
	// sm_103: 8/64 warps (12.5%, 2 blocks/SM) limited by tensor_memory
}
