package cuptiactivity

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/cupti"
)

// AllocationLifetime links observed allocation and release records by native
// address within one process generation, device, context and memory kind.
// Indices refer to the input of MemoryLifetimes. A missing side has index -1.
// Generation counts observed allocations at that address, starting at one;
// it is zero for a release whose allocation was not observed. An unreleased
// allocation is not proof of a leak or of physical memory still being resident.
type AllocationLifetime struct {
	AllocationIndex int
	ReleaseIndex    int
	Generation      uint64
}

// MemoryLifetimes reconstructs observed address generations from memory records.
// It sorts by native timestamp within the supplied session/run/process scopes,
// so callback delivery order does not determine pairing. Other kinds are ignored.
// It rejects missing identities/timestamps/addresses, unknown operations,
// simultaneous operations at the same address, and overlapping allocations.
// These conditions need a partial/ambiguous disposition rather than a guessed
// lifetime. Input records remain unchanged on success or failure.
//
// Results alone do not prove complete allocation coverage: the caller must
// retain collector losses and capture boundaries. Async timestamps describe
// native memory API operations, not device completion or physical residency.
func MemoryLifetimes(records []ScopedRecord) ([]AllocationLifetime, error) {
	type key struct {
		session, run, process string
		device, context       uint32
		kind                  cupti.ActivityMemoryKind
		address               uint64
	}
	keys := make(map[int]key)
	var order []int
	for i, r := range records {
		a := r.Record
		if a.Kind != cupti.ActivityKindMemory2 {
			continue
		}
		if r.Session == "" || r.Run == "" || r.Process == "" || a.Timestamp == 0 || a.Address == 0 {
			return nil, fmt.Errorf("cuptiactivity: memory record %d lacks identity or timestamp", i)
		}
		if a.MemoryOperation != cupti.ActivityMemoryOperationTypeAllocation && a.MemoryOperation != cupti.ActivityMemoryOperationTypeRelease {
			return nil, fmt.Errorf("cuptiactivity: memory record %d has unknown operation", i)
		}
		keys[i] = key{r.Session, r.Run, r.Process, a.DeviceID, a.ContextID, a.MemoryKind, a.Address}
		order = append(order, i)
	}
	sort.SliceStable(order, func(i, j int) bool { return records[order[i]].Record.Timestamp < records[order[j]].Record.Timestamp })
	last := make(map[key]uint64)
	live := make(map[key]int)
	generation := make(map[key]uint64)
	var out []AllocationLifetime
	for _, i := range order {
		a, k := records[i].Record, keys[i]
		if last[k] == a.Timestamp {
			return nil, fmt.Errorf("cuptiactivity: ambiguous simultaneous memory operation at record %d", i)
		}
		last[k] = a.Timestamp
		if a.MemoryOperation == cupti.ActivityMemoryOperationTypeAllocation {
			if _, ok := live[k]; ok {
				return nil, fmt.Errorf("cuptiactivity: overlapping allocation at record %d", i)
			}
			generation[k]++
			live[k] = len(out)
			out = append(out, AllocationLifetime{AllocationIndex: i, ReleaseIndex: -1, Generation: generation[k]})
		} else if allocation, ok := live[k]; ok {
			out[allocation].ReleaseIndex = i
			delete(live, k)
		} else {
			out = append(out, AllocationLifetime{AllocationIndex: -1, ReleaseIndex: i})
		}
	}
	return out, nil
}
