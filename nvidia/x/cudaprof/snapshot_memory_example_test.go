package cudaprof_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// fixedMemory admits reservations until its limit is reached.
type fixedMemory struct{ limit, live int64 }

func (m *fixedMemory) Reserve(n int64) error {
	if n > m.limit-m.live {
		return errors.New("limit reached")
	}
	m.live += n
	return nil
}

func (m *fixedMemory) Release(n int64) { m.live -= n }

func ExampleProfileSnapshot_GuidanceBounded() {
	snap := validProfileSnapshot()
	for pc := uint64(0); pc < 100; pc++ {
		snap.PCStalls[pc] = map[string]int64{"stall_mem": int64(pc)}
	}
	target := cudaprof.ProfileTarget{CodeDigest: snap.CodeDigest, Region: snap.Region, CollectorMode: snap.CollectorMode, Workload: snap.Workload, Device: snap.Device}
	for _, limit := range []int64{1 << 10, 1 << 20} {
		memory := &fixedMemory{limit: limit}
		scratch, _ := egraphcore.NewMemoryScope(memory)
		result, _ := egraphcore.NewMemoryScope(memory)
		g, err := snap.GuidanceBounded(target, nil, scratch, result)
		if err != nil {
			fmt.Println(limit, "refused:", err)
		} else {
			fmt.Println(limit, "admitted:", len(g.PCStalls), g.HotPCs["stall_mem"][:3])
		}
		scratch.Close()
		result.Close()
		fmt.Println("live", memory.live)
	}
	// Output:
	// 1024 refused: limit reached
	// live 0
	// 1048576 admitted: 100 [99 98 97]
	// live 0
}
