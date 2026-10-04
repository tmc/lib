package cudahotswap

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

var errGenerationPinned = errors.New("kernel generation is pinned")

// A generation owns one loaded module. Fields other than pins and retired
// are immutable after publication. The table lock protects retired and owner.
type generation struct {
	version   uint64
	digest    string
	flags     []string
	signature cudalaunch.Signature
	function  driver.Function
	owner     io.Closer
	pins      atomic.Int64
	retired   bool
}

type kernelEntry struct {
	current  atomic.Pointer[generation]
	original uint64
	revision uint64
	versions map[uint64]*generation
}

// generationTable separates immutable compiled versions from the monotonic
// mutation revision. Selecting a retained version does not create an aliasing
// second module owner. A raw handle remains valid only through its pin.
type generationTable struct {
	mu      sync.RWMutex
	entries map[string]*kernelEntry
	next    uint64
	closed  bool
}

type generationStamp struct {
	revision uint64
	version  uint64
}

// publish transfers candidate's module ownership only on success. expected
// is the version against which compilation and checking were performed; zero
// registers a new kernel. The caller owns failed candidates, including retry
// cleanup when their module cannot yet be unloaded.
func (t *generationTable) publish(name string, candidate *generation, expected uint64) (generationStamp, error) {
	if name == "" || candidate == nil || candidate.owner == nil || candidate.function == 0 || candidate.digest == "" || candidate.version != 0 || candidate.retired || candidate.pins.Load() != 0 {
		return generationStamp{}, errors.New("invalid kernel generation")
	}
	if err := CheckABI(candidate.signature, candidate.signature); err != nil {
		return generationStamp{}, err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return generationStamp{}, errors.New("kernel table is closed")
	}
	e := t.entries[name]
	if e == nil {
		if expected != 0 {
			return generationStamp{}, errors.New("kernel disappeared during compilation")
		}
	} else {
		current := e.current.Load()
		if current == nil || expected != current.version {
			return generationStamp{}, errors.New("kernel changed during compilation")
		}
		if err := CheckABI(current.signature, candidate.signature); err != nil {
			return generationStamp{}, err
		}
	}
	if t.next == math.MaxUint64 {
		return generationStamp{}, errors.New("kernel generation counter exhausted")
	}
	t.next++
	// Copy compiler metadata before publication; caller mutations must not
	// change the ABI or the flags recorded for the selected generation.
	candidate.flags = append([]string(nil), candidate.flags...)
	candidate.signature.Params = append([]cudalaunch.Param(nil), candidate.signature.Params...)
	candidate.version = t.next
	if t.entries == nil {
		t.entries = make(map[string]*kernelEntry)
	}
	if e == nil {
		e = &kernelEntry{original: candidate.version, versions: make(map[uint64]*generation)}
		t.entries[name] = e
	}
	e.versions[candidate.version] = candidate
	e.revision = t.next
	e.current.Store(candidate)
	return generationStamp{e.revision, candidate.version}, nil
}

func (t *generationTable) selectVersion(name string, version uint64) (generationStamp, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.selectLocked(name, version)
}

func (t *generationTable) selectLocked(name string, version uint64) (generationStamp, error) {
	if t.closed {
		return generationStamp{}, errors.New("kernel table is closed")
	}
	e := t.entries[name]
	if e == nil || e.versions[version] == nil || e.versions[version].retired {
		return generationStamp{}, errors.New("kernel version is not retained")
	}
	if t.next == math.MaxUint64 {
		return generationStamp{}, errors.New("kernel generation counter exhausted")
	}
	t.next++
	e.revision = t.next
	e.current.Store(e.versions[version])
	return generationStamp{e.revision, version}, nil
}

func (t *generationTable) revert(name string) (generationStamp, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	e := t.entries[name]
	if e == nil {
		return generationStamp{}, errors.New("kernel is not registered")
	}
	return t.selectLocked(name, e.original)
}

type generationPin struct {
	table    *generationTable
	gen      *generation
	released atomic.Bool
}

func (t *generationTable) pin(name string) (*generationPin, generationStamp, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.closed {
		return nil, generationStamp{}, errors.New("kernel table is closed")
	}
	e := t.entries[name]
	if e == nil {
		return nil, generationStamp{}, errors.New("kernel is not registered")
	}
	g := e.current.Load()
	if g == nil || g.retired {
		return nil, generationStamp{}, errors.New("kernel version is not retained")
	}
	g.pins.Add(1)
	return &generationPin{table: t, gen: g}, generationStamp{e.revision, g.version}, nil
}

// release must follow actual launch or graph completion. Cancellation of an
// observation is not permission to release the pin. Failed cleanup leaves the
// module owned by the table, where collect or close can retry it.
func (p *generationPin) release() error {
	if p == nil || p.released.Swap(true) {
		return nil
	}
	t := p.table
	t.mu.Lock()
	defer t.mu.Unlock()
	p.gen.pins.Add(-1)
	return t.collectLocked()
}

func (t *generationTable) drop(name string, version uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return errors.New("kernel table is closed")
	}
	e := t.entries[name]
	if e == nil || e.versions[version] == nil {
		return errors.New("kernel version is not retained")
	}
	if e.original == version || e.current.Load() == e.versions[version] {
		return errors.New("cannot drop original or current kernel version")
	}
	e.versions[version].retired = true
	return t.collectLocked()
}

func (t *generationTable) collectLocked() error {
	var result error
	for name, e := range t.entries {
		for version, g := range e.versions {
			if !g.retired || g.pins.Load() != 0 {
				continue
			}
			if err := g.owner.Close(); err != nil {
				result = errors.Join(result, fmt.Errorf("close %q version %d: %w", name, version, err))
				continue
			}
			g.owner = nil
			delete(e.versions, version)
		}
		if len(e.versions) == 0 {
			delete(t.entries, name)
		}
	}
	return result
}

func (t *generationTable) close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	pinned := false
	for _, e := range t.entries {
		e.current.Store(nil)
		for _, g := range e.versions {
			g.retired = true
			pinned = pinned || g.pins.Load() != 0
		}
	}
	err := t.collectLocked()
	if pinned {
		err = errors.Join(err, errGenerationPinned)
	}
	return err
}
