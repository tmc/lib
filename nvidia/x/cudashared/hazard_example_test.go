package cudashared_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudashared"
)

func ExampleHazardMatrix() {
	m := cudashared.NewHazardMatrix("sm_90")

	// Thread 0 reads byte 0
	_ = m.AddAccess(cudashared.MemoryAccess{
		Scope:     cudashared.ScopeCTA,
		ThreadID:  0,
		Space:     cudashared.MemoryShared,
		Offset:    0x1000,
		SizeBytes: 1,
		IsWrite:   false,
		Stage:     0,
	})

	// Thread 32 writes byte 1 (adjacent byte in same 32-bit word)
	_ = m.AddAccess(cudashared.MemoryAccess{
		Scope:     cudashared.ScopeCTA,
		ThreadID:  32,
		Space:     cudashared.MemoryShared,
		Offset:    0x1001,
		SizeBytes: 1,
		IsWrite:   true,
		Stage:     0,
	})

	records, err := m.Evaluate()
	if err != nil {
		panic(err)
	}

	for _, r := range records {
		fmt.Printf("WordConflict=%v ByteConflict=%v IsRace=%v\n", r.WordConflict, r.ByteConflict, r.IsRace)
	}

	// Output:
	// WordConflict=true ByteConflict=false IsRace=false
}
