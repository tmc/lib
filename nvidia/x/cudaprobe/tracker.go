package cudaprobe

import (
	"fmt"
	"sort"
)

// Allocation is an observed allocation without a matched successful free.
// It is not a leak diagnosis; unobserved paths can allocate or free memory.
type Allocation struct {
	AddressSpace, Address, Bytes, Sequence uint64
}

type allocationKey struct{ space, address uint64 }

// Tracker models synchronous CUDA allocation calls for one process lifetime.
// It is not safe for concurrent use. Construct it with NewTracker. It retains
// at most the configured number of allocations, with no history of freed ones.
// Runtime and driver probes for the same operation must not both feed it:
// nested calls are separate observations, not independent allocations.
type Tracker struct {
	process    Process
	capacity   int
	sequence   uint64
	active     map[allocationKey]Allocation
	incomplete bool
	exited     bool
}

// NewTracker creates a bounded allocation tracker for process.
func NewTracker(process Process, capacity int) (*Tracker, error) {
	if process.PID == 0 || process.Generation == 0 || capacity <= 0 {
		return nil, fmt.Errorf("cudaprobe: invalid process or allocation capacity")
	}
	return &Tracker{process: process, capacity: capacity, active: make(map[allocationKey]Allocation)}, nil
}

// Observe applies one process-ordered observation. Sequence gaps, undecoded
// calls, duplicate live addresses, unknown frees and cap exhaustion make the
// report incomplete. Errors do not advance the sequence or alter allocations.
// Failed API calls do not allocate or free. Async allocation API observations are not
// applied because API return is not their lifetime completion boundary.
func (t *Tracker) Observe(e Event) error {
	if t == nil || t.active == nil {
		return fmt.Errorf("cudaprobe: uninitialized tracker")
	}
	if t.exited {
		return fmt.Errorf("cudaprobe: process has exited")
	}
	if e.Identity.Process != t.process {
		return fmt.Errorf("cudaprobe: process identity mismatch")
	}
	if e.Sequence == 0 || e.Sequence <= t.sequence {
		return fmt.Errorf("cudaprobe: sequence is not increasing")
	}
	if e.Sequence != t.sequence+1 {
		t.incomplete = true
	}
	t.sequence = e.Sequence
	if !e.Decoded {
		t.incomplete = true
		return nil
	}
	if e.AsyncMemory != nil || (e.Control != nil && e.Control.Scope == "context" && e.Control.Operation != "get_current") {
		t.incomplete = true
		return nil
	}
	if !e.ReturnKnown || e.Status != 0 {
		return nil
	}
	key := allocationKey{e.Identity.AddressSpace, e.Address}
	switch e.Identity.Symbol {
	case "cudaMalloc", "cuMemAlloc_v2":
		if e.Bytes == 0 {
			return nil
		}
		if key.space == 0 || e.Address == 0 || e.Bytes > ^uint64(0)-e.Address {
			t.incomplete = true
			return nil
		}
		if _, ok := t.active[key]; ok {
			t.incomplete = true
		}
		if _, ok := t.active[key]; !ok && len(t.active) >= t.capacity {
			t.incomplete = true
			return nil
		}
		t.active[key] = Allocation{AddressSpace: key.space, Address: e.Address, Bytes: e.Bytes, Sequence: e.Sequence}
	case "cudaFree", "cuMemFree_v2":
		if e.Address == 0 {
			return nil
		}
		if key.space == 0 {
			t.incomplete = true
			return nil
		}
		if _, ok := t.active[key]; !ok {
			t.incomplete = true
		}
		delete(t.active, key)
	}
	return nil
}

// Gap marks the tracking interval incomplete when the collector reports loss
// without a subsequent event, for example a dropped tail before process exit.
func (t *Tracker) Gap() { t.incomplete = true }

// Report copies the current unmatched allocations in address order. Incomplete
// remains true after any observation gap, even if all known allocations free.
func (t *Tracker) Report() (unmatched []Allocation, incomplete bool) {
	for _, a := range t.active {
		unmatched = append(unmatched, a)
	}
	sort.Slice(unmatched, func(i, j int) bool {
		if unmatched[i].AddressSpace != unmatched[j].AddressSpace {
			return unmatched[i].AddressSpace < unmatched[j].AddressSpace
		}
		return unmatched[i].Address < unmatched[j].Address
	})
	return unmatched, t.incomplete
}

// Exit closes the process lifetime. Remaining allocations are unmatched at exit,
// not leaks. Subsequent observations are rejected even if the PID is reused.
func (t *Tracker) Exit() { t.exited = true }
