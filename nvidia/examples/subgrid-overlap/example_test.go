package main

import (
	"fmt"
)

func Example() {
	ptx := ModulePTX(DefaultPTXConfig())
	m, err := AuditModule(ptx)
	if err != nil {
		fmt.Printf("audit error: %v\n", err)
		return
	}
	fmt.Printf("Audit passed: target %s, %d kernels\n", m.Target, len(m.Kernels))
	// Output:
	// Audit passed: target sm_90, 2 kernels
}

func Example_pdlOverlap() {
	hw := DefaultHardware()
	wl := DefaultWorkload()

	// Run Sequential Graph
	graphRes, err := Simulate(SimConfig{
		Hardware: hw,
		Workload: wl,
		Mode:     ModeSequentialGraph,
	})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	// Run PDL Overlap
	pdlRes, err := Simulate(SimConfig{
		Hardware: hw,
		Workload: wl,
		Mode:     ModePDLOverlap,
	})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	speedup := float64(graphRes.TotalCycles) / float64(pdlRes.TotalCycles)

	fmt.Printf("Sequential Graph Latency: %.2f us, Overlap: %.2f us\n",
		float64(graphRes.TotalDuration.Microseconds()),
		float64(graphRes.OverlapDuration.Microseconds()))
	fmt.Printf("PDL Overlap Latency: %.2f us, Overlap: %.2f us\n",
		float64(pdlRes.TotalDuration.Microseconds()),
		float64(pdlRes.OverlapDuration.Microseconds()))
	fmt.Printf("Speedup: %.2fx\n", speedup)

	// Output:
	// Sequential Graph Latency: 114.00 us, Overlap: 0.00 us
	// PDL Overlap Latency: 85.00 us, Overlap: 28.00 us
	// Speedup: 1.34x
}
