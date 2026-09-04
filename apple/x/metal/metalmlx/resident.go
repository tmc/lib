package metalmlx

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"

	"github.com/tmc/apple/metal"
	"github.com/tmc/apple/objc"
)

// noSet marks an allocation that is tracked but is in no set, because it did
// not fit in the wired limit.
const noSet = ^uint32(0)

// maxSets caps the number of sets: a command queue accepts a limited number
// of residency sets and every set is attached to every queue. The limit is
// from the Metal feature set tables.
const maxSets = 32

// ResidencySets keeps allocations GPU-resident, up to the wired limit set by
// SetWiredLimit (0 by default, so nothing is wired unless asked for). Port of
// mlx/backend/metal/resident.cpp.
//
// Within that budget the allocations are distributed over several size-capped
// residency sets rather than one large one. macOS makes residency decisions
// per set, so when a set loses residency under memory pressure only its
// allocations have to be made resident again.
type ResidencySets struct {
	dev     metal.MTLDevice
	enabled bool

	mu             sync.Mutex
	maxBytesPerSet uint64 // per-set cap, 0 for a single set
	capacity       uint64
	totalWired     uint64
	sets           []residencySet         // indexed by id, append-only
	placement      map[objc.ID]*placement // every tracked allocation
	allocs         map[objc.ID]metal.MTLAllocation

	// len(sets), published so a queue can check for new sets without the lock.
	numSets atomic.Uint64
}

type residencySet struct {
	set  metal.MTLResidencySet
	size uint64
}

// placement records the set an allocation lives in and the size it was
// inserted with. Sizes are recorded rather than read back at erase, so the
// running totals cannot drift.
type placement struct {
	setID uint32
	bytes uint64
}

func newResidencySets(dev metal.MTLDevice) (*ResidencySets, error) {
	r := &ResidencySets{
		placement: make(map[objc.ID]*placement),
		allocs:    make(map[objc.ID]metal.MTLAllocation),
	}
	if !dev.SupportsFamily(metal.MTLGPUFamilyMetal3) {
		return r, nil // disabled, everything below is a no-op
	}
	r.dev = dev
	r.enabled = true
	if pct := envInt("MLX_RESIDENCY_SET_MAX_PCT", 5); pct > 0 && pct < 100 {
		// Floor the cap so a small working set or percentage cannot ask for
		// an absurd number of sets.
		ws := dev.RecommendedMaxWorkingSetSize()
		r.maxBytesPerSet = max(ws/100*uint64(pct), 64<<20)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	// Set 0 always exists and is the fallback when a later set cannot be
	// made, so failing to create it is fatal.
	if err := r.addSetLocked(); err != nil {
		return nil, fmt.Errorf("metalmlx: unable to construct residency set: %w", err)
	}
	return r, nil
}

// Enabled reports whether residency sets are available on this device.
func (r *ResidencySets) Enabled() bool { return r.enabled }

// NumSets returns the number of residency sets created so far.
func (r *ResidencySets) NumSets() int { return int(r.numSets.Load()) }

// WiredSize returns the total bytes currently wired across all sets.
func (r *ResidencySets) WiredSize() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.totalWired
}

func (r *ResidencySets) addSetLocked() error {
	set, err := r.dev.NewResidencySetWithDescriptorError(metal.NewMTLResidencySetDescriptor())
	if err != nil {
		return err
	}
	if set == nil {
		return fmt.Errorf("metalmlx: newResidencySet returned nil")
	}
	// A standing request, so allocations added later are covered without
	// requesting residency again on every insert.
	set.RequestResidency()
	r.sets = append(r.sets, residencySet{set: set})
	r.numSets.Store(uint64(len(r.sets)))
	if envInt("MLX_RESIDENCY_DEBUG", 0) != 0 {
		fmt.Fprintf(os.Stderr, "[residency] created residency set %d (max_bytes_per_set=%d MB)\n",
			len(r.sets)-1, r.maxBytesPerSet>>20)
	}
	return nil
}

func (r *ResidencySets) chooseSetLocked(bytes uint64) uint32 {
	if r.maxBytesPerSet == 0 {
		return 0
	}
	none := uint32(len(r.sets))
	empty, emptiest := none, uint32(0)
	for i := range r.sets {
		size := r.sets[i].size
		if size+bytes <= r.maxBytesPerSet {
			return uint32(i) // first fit
		}
		// Only reached by an allocation larger than the cap, which needs a
		// set of its own. Reusing an emptied one keeps the count from growing.
		if i != 0 && size == 0 && empty == none {
			empty = uint32(i)
		}
		if size < r.sets[emptiest].size {
			emptiest = uint32(i)
		}
	}
	if empty != none {
		return empty
	}
	if len(r.sets) < maxSets && r.addSetLocked() == nil {
		return uint32(len(r.sets) - 1)
	}
	// At the cap, or the driver would not make another set. Filling the
	// emptiest set keeps them evenly sized.
	return emptiest
}

func (r *ResidencySets) addToSetLocked(buf metal.MTLAllocation, at *placement) uint32 {
	idx := r.chooseSetLocked(at.bytes)
	s := &r.sets[idx]
	s.set.AddAllocation(buf)
	s.size += at.bytes
	r.totalWired += at.bytes
	at.setID = idx
	return idx
}

func (r *ResidencySets) removeFromSetLocked(buf metal.MTLAllocation, at *placement) {
	s := &r.sets[at.setID]
	s.set.RemoveAllocation(buf)
	s.size -= at.bytes
	r.totalWired -= at.bytes
	at.setID = noSet
}

// Insert starts tracking an allocation, and wires it if it fits in the limit.
func (r *ResidencySets) Insert(buf metal.MTLAllocation, bytes uint64) {
	if !r.enabled {
		return
	}
	id := buf.GetID()
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.placement[id]; ok {
		return // already tracked
	}
	at := &placement{setID: noSet, bytes: bytes}
	r.placement[id] = at
	r.allocs[id] = buf
	// Stay within the wired limit. The excess is tracked but left out of any
	// set, and resize() adds it to one if the limit is raised later.
	if r.totalWired+bytes > r.capacity {
		return
	}
	idx := r.addToSetLocked(buf, at)
	r.sets[idx].set.Commit()
}

// Erase stops tracking an allocation and unwires it.
func (r *ResidencySets) Erase(buf metal.MTLAllocation) {
	if !r.enabled {
		return
	}
	id := buf.GetID()
	r.mu.Lock()
	defer r.mu.Unlock()
	at, ok := r.placement[id]
	if !ok {
		return
	}
	if at.setID != noSet {
		idx := at.setID
		r.removeFromSetLocked(buf, at)
		r.sets[idx].set.Commit()
	}
	delete(r.placement, id)
	delete(r.allocs, id)
}

// Resize sets the wired budget, wiring or unwiring allocations to match. A
// bulk resize costs one commit per set rather than one per allocation.
func (r *ResidencySets) Resize(size uint64) {
	if !r.enabled {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.capacity == size {
		return
	}
	r.capacity = size

	touched := make([]bool, len(r.sets))
	if r.totalWired < r.capacity {
		// The budget grew: add allocations that now fit.
		for id, at := range r.placement {
			if at.setID != noSet || r.totalWired+at.bytes > r.capacity {
				continue
			}
			idx := r.addToSetLocked(r.allocs[id], at)
			if int(idx) >= len(touched) {
				touched = append(touched, make([]bool, len(r.sets)-len(touched))...)
			}
			touched[idx] = true
		}
	} else {
		// The budget shrank: remove allocations until we are back under it.
		for id, at := range r.placement {
			if r.totalWired <= r.capacity {
				break
			}
			if at.setID == noSet {
				continue
			}
			touched[at.setID] = true
			r.removeFromSetLocked(r.allocs[id], at)
		}
	}
	for i, t := range touched {
		if t {
			r.sets[i].set.Commit()
		}
	}
}

// attachNewSets attaches the sets this queue has not seen yet. It is called
// before every command buffer commit, because Metal locks a command buffer's
// residency at commit time.
func (r *ResidencySets) attachNewSets(q metal.MTLCommandQueue, attached *uint64) {
	// Lock-free when there is nothing new, which is the common case. It also
	// covers the disabled case, where the count stays 0.
	if r.numSets.Load() == *attached {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	sets := make([]metal.MTLResidencySet, 0, uint64(len(r.sets))-*attached)
	for id := *attached; id < uint64(len(r.sets)); id++ {
		sets = append(sets, r.sets[id].set)
	}
	q.AddResidencySetsCount(sets, uint(len(sets)))
	*attached = uint64(len(r.sets))
}

// setMaxBytesPerSet sets the per-set cap for subsequent inserts, bypassing
// the size floor so tests can reach the multi-set paths cheaply.
func (r *ResidencySets) setMaxBytesPerSet(bytes uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.maxBytesPerSet = bytes
}

func (r *ResidencySets) close() {
	if r == nil || !r.enabled {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := range r.sets {
		r.sets[i].set.EndResidency()
		release(r.sets[i].set)
	}
	r.sets = nil
	clear(r.placement)
	clear(r.allocs)
	r.totalWired = 0
	r.numSets.Store(0)
}
