// Package cudamemtrack tracks GPU memory allocations, calculates heap
// fragmentation ratios, and detects memory leaks at teardown.
//
// Unlike [github.com/tmc/lib/nvidia/x/cudamem] which owns typed allocations,
// cudamemtrack acts as an external monitor and lifecycle auditor for
// physical and virtual device memory.
//
// # Basic Usage
//
// Track allocations and assert no memory is leaked at test teardown:
//
//	tracker := cudamemtrack.New()
//	tracker.Alloc(0x1000, 1024*1024, "weights")
//	// ... run workload ...
//	tracker.Free(0x1000)
//	if err := tracker.CheckLeaks(); err != nil {
//		log.Fatal(err)
//	}
//
// # Fragmentation Index
//
// Compute real-time fragmentation from total and contiguous free memory:
//
//	// 100 MB free, but largest contiguous block is only 20 MB:
//	f := cudamemtrack.Fragmentation(100*1024*1024, 20*1024*1024)
//	fmt.Printf("Fragmentation: %.1f%%\n", f*100) // 80.0%
package cudamemtrack
