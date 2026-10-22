package dsv4expertcache

import (
	"errors"
	"fmt"
	"math"
	"reflect"
)

// Identity binds a cache to immutable model contents and a quantized layout.
type Identity struct{ Model, Quant, Layout [32]byte }

// Key names one routed expert within an immutable weight layout.
type Key struct {
	Identity      Identity
	Layer, Expert int
}

// Completion waits until all work represented by a backend event is complete.
// Upload events must cover writes to the entire expert triplet. Consumer events
// must cover every use of the lease's slots, including graph replays.
// Wait must be repeatable: Commit failure followed by Abort waits again. A
// successful Wait must remain successful. Adapters retain the event until the
// lease ends; Wait must not destroy it.
type Completion interface{ Wait() error }

// Upload identifies a reserved slot. Generation prevents stale lease reuse.
type Upload struct {
	Slot       int
	Generation uint64
	Key        Key
}

type slot struct {
	key   Key
	valid bool
	used  uint64
}

// Cache owns slot metadata. Its zero value is unusable. Calls must be serialized.
type Cache struct {
	identity   Identity
	slots      []slot
	generation uint64
	active     *Lease
	closed     bool
}

// New validates a fixed-size cache against its byte budget. Expert bytes include
// gate, up and down; backend allocation and alignment remain adapter concerns.
func New(identity Identity, slots int, expertBytes, budget uint64) (*Cache, error) {
	if identity.Model == ([32]byte{}) || identity.Quant == ([32]byte{}) || identity.Layout == ([32]byte{}) {
		return nil, errors.New("missing weight identity")
	}
	if slots <= 0 || expertBytes == 0 || uint64(slots) > budget/expertBytes {
		return nil, errors.New("invalid cache budget")
	}
	return &Cache{identity: identity, slots: make([]slot, slots)}, nil
}

// Lease protects all selected slots until consumer completion. It is created
// only by Acquire. Returned slices are detached from planner state.
type Lease struct {
	cache           *Cache
	generation      uint64
	remap           []int
	uploads         []Upload
	events          map[int]Completion
	committed, done bool
}

// Acquire protects every hit before assigning LRU victims to unique misses.
// A rejected request leaves cache metadata unchanged.
func (c *Cache) Acquire(keys []Key) (*Lease, error) {
	if c == nil || c.closed || len(c.slots) == 0 {
		return nil, errors.New("cache closed or uninitialized")
	}
	if c.active != nil {
		return nil, errors.New("cache lease active")
	}
	if c.generation == math.MaxUint64 {
		return nil, errors.New("cache generation exhausted")
	}
	unique := make(map[Key]int)
	for _, k := range keys {
		if k.Identity != c.identity || k.Layer < 0 || k.Expert < 0 {
			return nil, errors.New("invalid expert key")
		}
		unique[k] = -1
	}
	if len(keys) == 0 || len(unique) > len(c.slots) {
		return nil, errors.New("request exceeds slot capacity")
	}
	protected := make([]bool, len(c.slots))
	for i, s := range c.slots {
		if s.valid {
			if _, ok := unique[s.key]; ok {
				unique[s.key] = i
				protected[i] = true
			}
		}
	}
	l := &Lease{cache: c, generation: c.generation + 1, events: make(map[int]Completion)}
	for _, k := range keys {
		i := unique[k]
		if i < 0 {
			i = -1
			for j, s := range c.slots {
				if !protected[j] && (i < 0 || !s.valid || (c.slots[i].valid && s.used < c.slots[i].used)) {
					i = j
					if !s.valid {
						break
					}
				}
			}
			protected[i] = true
			unique[k] = i
			l.uploads = append(l.uploads, Upload{i, l.generation, k})
		}
		l.remap = append(l.remap, i)
	}
	c.generation = l.generation
	c.active = l
	return l, nil
}

// Slots returns remapped IDs in the original request order. Before Commit these
// are staging addresses only, not consumable mappings. A nil lease returns nil.
func (l *Lease) Slots() []int {
	if l == nil {
		return nil
	}
	return append([]int(nil), l.remap...)
}

// Uploads returns the unique misses in first-request order.
// A nil lease returns nil.
func (l *Lease) Uploads() []Upload {
	if l == nil {
		return nil
	}
	return append([]Upload(nil), l.uploads...)
}

func nilCompletion(event any) bool {
	if event == nil {
		return true
	}
	v := reflect.ValueOf(event)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}

func (l *Lease) check() error {
	if l == nil || l.cache == nil || l.done || l.cache.closed || l.cache.active != l || l.generation != l.cache.generation {
		return errors.New("stale cache lease")
	}
	return nil
}

// Started reserves a victim for writing and registers its completion before
// submission. The adapter must make Wait cover every submitted write, including
// partially failed submissions. Registering a victim invalidates it on Abort
// even when submission writes nothing; this conservative loss is intentional.
func (l *Lease) Started(u Upload, event Completion) error {
	if err := l.check(); err != nil {
		return err
	}
	if l.committed || nilCompletion(event) {
		return errors.New("invalid upload completion")
	}
	for _, want := range l.uploads {
		if want == u {
			if _, ok := l.events[u.Slot]; ok {
				return errors.New("upload already started")
			}
			l.events[u.Slot] = event
			return nil
		}
	}
	return errors.New("upload does not belong to lease")
}

// Commit waits for all missing triplets before publishing their mappings.
// On failure the caller must Abort; no consumer may use this lease yet.
func (l *Lease) Commit() error {
	if err := l.check(); err != nil {
		return err
	}
	if l.committed {
		return errors.New("lease already committed")
	}
	for _, u := range l.uploads {
		if l.events[u.Slot] == nil {
			return errors.New("upload incomplete")
		}
	}
	for _, u := range l.uploads {
		if err := l.events[u.Slot].Wait(); err != nil {
			return fmt.Errorf("wait for upload: %w", err)
		}
	}
	for _, u := range l.uploads {
		l.cache.slots[u.Slot] = slot{u.Key, true, l.generation}
	}
	for _, i := range l.remap {
		l.cache.slots[i].used = l.generation
	}
	l.committed = true
	return nil
}

// Abort waits for started writes and invalidates only their victims. A failed
// wait closes the cache so uncertain in-flight writes can never be reused.
func (l *Lease) Abort() error {
	if err := l.check(); err != nil {
		return err
	}
	if l.committed {
		return errors.New("committed lease requires release")
	}
	var err error
	for i, event := range l.events {
		err = errors.Join(err, event.Wait())
		l.cache.slots[i].valid = false
	}
	if err != nil {
		l.cache.closed = true
	}
	l.done = true
	l.cache.active = nil
	return err
}

// Release waits for actual consumer completion before allowing slot reuse.
// A failed wait leaves the lease active; no slot can be reused.
func (l *Lease) Release(event Completion) error {
	if err := l.check(); err != nil {
		return err
	}
	if !l.committed || nilCompletion(event) {
		return errors.New("invalid consumer completion")
	}
	if err := event.Wait(); err != nil {
		return fmt.Errorf("wait for consumer: %w", err)
	}
	l.done = true
	l.cache.active = nil
	return nil
}

// Close rejects active leases. Device-memory teardown remains the backend's job.
func (c *Cache) Close() error {
	if c == nil {
		return nil
	}
	if c.active != nil {
		return errors.New("cache lease active")
	}
	c.closed = true
	return nil
}
