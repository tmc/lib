package dsv4

import (
	"fmt"
	"math"
)

// ContextMemory counts live request-owned numeric backing storage in bytes.
// It excludes shared weights, allocator overhead and transient Eval locals.
// Compressed includes indexer and rolling compressor storage; Scratch includes
// retained logits and steering. Values describe capacity, not populated rows.
type ContextMemory struct {
	Raw, Compressed, Scratch int64
}

// Total returns the checked sum of the three disjoint storage classes.
func (m ContextMemory) Total() (int64, error) {
	var total int64
	for _, n := range []int64{m.Raw, m.Compressed, m.Scratch} {
		if n < 0 || n > math.MaxInt64-total {
			return 0, fmt.Errorf("context memory: invalid byte count")
		}
		total += n
	}
	return total, nil
}

// MemoryReporter is an optional State or Snapshot capability. Implementations
// enumerate actual owned backing storage after synchronizing device work.
// A retained owner whose Close failed must remain accounted for until release.
// Unsupported or stale observations return an error, never placeholder zeros.
type MemoryReporter interface {
	MemoryStats() (ContextMemory, error)
}

// SnapshotSizer reports actual serialized snapshot payload bytes, not arena
// capacity. A closed snapshot must return ErrClosed.
type SnapshotSizer interface {
	SnapshotBytes() (int64, error)
}
