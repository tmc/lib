package cudacapture

import (
	"errors"
	"fmt"
	"sync"
)

// ErrModuleGenerationReuse indicates that a module base address was reused with a different code digest
// without an explicit generation bump.
var ErrModuleGenerationReuse = errors.New("cudacapture: module address reused with different code digest")

// ModuleEntry tracks an immutable module generation loaded at a specific virtual address.
type ModuleEntry struct {
	ID          string
	BaseAddress uint64
	Generation  uint64
	ImageDigest string
	Code        CodeFiles
}

// ModuleTracker tracks loaded modules and their generation history by virtual address.
// When an address is reused with a different binary digest, it detects stale lookups
// and requires generation tracking.
type ModuleTracker struct {
	mu          sync.Mutex
	generations map[uint64][]ModuleEntry // base address -> slice of generations in order
}

// NewModuleTracker creates a new module tracker.
func NewModuleTracker() *ModuleTracker {
	return &ModuleTracker{
		generations: make(map[uint64][]ModuleEntry),
	}
}

// Load registers a module image at baseAddress.
// If baseAddress is already in use:
//   - With the identical digest: returns the existing generation number.
//   - With a different digest: increments the generation and records the new entry.
func (mt *ModuleTracker) Load(id string, baseAddress uint64, digest string, code CodeFiles) (uint64, error) {
	if baseAddress == 0 {
		return 0, errors.New("cudacapture: invalid zero module base address")
	}
	if digest == "" {
		return 0, errors.New("cudacapture: empty module code digest")
	}

	mt.mu.Lock()
	defer mt.mu.Unlock()

	history := mt.generations[baseAddress]
	if len(history) > 0 {
		latest := history[len(history)-1]
		if latest.ImageDigest == digest {
			return latest.Generation, nil
		}
		newGen := latest.Generation + 1
		entry := ModuleEntry{
			ID:          id,
			BaseAddress: baseAddress,
			Generation:  newGen,
			ImageDigest: digest,
			Code:        code,
		}
		mt.generations[baseAddress] = append(history, entry)
		return newGen, nil
	}

	entry := ModuleEntry{
		ID:          id,
		BaseAddress: baseAddress,
		Generation:  1,
		ImageDigest: digest,
		Code:        code,
	}
	mt.generations[baseAddress] = []ModuleEntry{entry}
	return 1, nil
}

// Verify checks that a specific generation at baseAddress matches expectedDigest.
// If the address was reused by a different digest and a stale/incorrect generation is queried,
// it returns ErrModuleGenerationReuse.
func (mt *ModuleTracker) Verify(baseAddress uint64, generation uint64, expectedDigest string) error {
	mt.mu.Lock()
	defer mt.mu.Unlock()

	history, ok := mt.generations[baseAddress]
	if !ok || len(history) == 0 {
		return fmt.Errorf("cudacapture: no module registered at address 0x%x", baseAddress)
	}

	for _, entry := range history {
		if entry.Generation == generation {
			if entry.ImageDigest != expectedDigest {
				return fmt.Errorf("%w: generation %d at 0x%x has digest %q, want %q",
					ErrModuleGenerationReuse, generation, baseAddress, entry.ImageDigest, expectedDigest)
			}
			return nil
		}
	}

	// Address exists, but generation does not match; address was reused across generations
	latest := history[len(history)-1]
	return fmt.Errorf("%w: stale generation %d for address 0x%x (latest generation is %d with digest %s)",
		ErrModuleGenerationReuse, generation, baseAddress, latest.Generation, latest.ImageDigest)
}

// Lookup returns the module entry for a given base address and generation.
func (mt *ModuleTracker) Lookup(baseAddress uint64, generation uint64) (*ModuleEntry, error) {
	mt.mu.Lock()
	defer mt.mu.Unlock()

	history, ok := mt.generations[baseAddress]
	if !ok || len(history) == 0 {
		return nil, fmt.Errorf("cudacapture: no module registered at address 0x%x", baseAddress)
	}

	for _, entry := range history {
		if entry.Generation == generation {
			entryCopy := entry
			return &entryCopy, nil
		}
	}

	return nil, fmt.Errorf("cudacapture: generation %d not found for address 0x%x", generation, baseAddress)
}
