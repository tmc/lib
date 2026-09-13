package cudamemtrack_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudamemtrack"
)

func Example() {
	tracker := cudamemtrack.New()

	// Track device allocations
	tracker.Alloc(0x7f10000000, 1024*1024, "weights")
	tracker.Alloc(0x7f20000000, 2048*1024, "activation")

	s := tracker.Summary()
	fmt.Printf("Active: %d allocations (%d bytes)\n", s.ActiveAllocations, s.ActiveBytes)

	// Free memory
	if err := tracker.Free(0x7f10000000); err != nil {
		log.Fatal(err)
	}
	if err := tracker.Free(0x7f20000000); err != nil {
		log.Fatal(err)
	}

	// Verify no leaks remain
	if err := tracker.CheckLeaks(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("No leaks detected")

	// Output:
	// Active: 2 allocations (3145728 bytes)
	// No leaks detected
}
