package dsv4expertcache

import (
	"errors"
	"fmt"
	"math"
)

// MetalUpload owns bounded shared expert storage. Calls must be serialized.
// It retains storage until every consumer lease has completed.
type MetalUpload struct {
	backend                 metalUploadBackend
	strides                 [3]int
	slots                   int
	lease                   *Lease
	hotness                 MetalHotness
	boundCache              *Cache
	staging                 *Lease
	closed, closing, failed bool
}

// MetalPinFailure reports a failed slot pin without claiming a completed write.
// LockedBytes and LockedSlots describe the already pinned cache, not staging.
type MetalPinFailure struct {
	LockedSlots uint32
	LockedBytes uint64
	Err         error
}

func (e *MetalPinFailure) Error() string {
	if e == nil || e.Err == nil {
		return "pin Metal expert slot"
	}
	return "pin Metal expert slot: " + e.Err.Error()
}
func (e *MetalPinFailure) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

type metalUploadBackend interface {
	pageSize() uint64
	allocate(int, [3]int) error
	stage() []byte
	write(int, int, []byte) error
	close() error
}

// metalAllocationBytes counts all slot padding and the separately owned staging
// reservation. The budget covers shared expert slabs; staging is additional.
func metalAllocationBytes(slots uint64, strides [3]uint64, page, budget uint64) (slot, shared, stage uint64, err error) {
	if slots == 0 || page == 0 || page&(page-1) != 0 {
		return 0, 0, 0, errors.New("invalid Metal allocation shape")
	}
	var logical, largest uint64
	for _, n := range strides {
		if n == 0 || logical > math.MaxUint64-n {
			return 0, 0, 0, errors.New("Metal triplet extent overflow")
		}
		logical += n
		largest = max(largest, n)
	}
	if logical > math.MaxUint64-(page-1) || largest > math.MaxUint64-(page-1) {
		return 0, 0, 0, errors.New("Metal page extent overflow")
	}
	slot = (logical + page - 1) &^ (page - 1)
	stage = (largest + page - 1) &^ (page - 1)
	if slots > math.MaxUint64/slot {
		return 0, 0, 0, errors.New("Metal slab extent overflow")
	}
	shared = slots * slot
	if shared > budget {
		return 0, 0, 0, errors.New("Metal page-aligned experts exceed budget")
	}
	if shared > math.MaxUint64-stage {
		return 0, 0, 0, errors.New("Metal allocation reserve overflow")
	}
	return slot, shared, stage, nil
}

func newMetalUpload(b metalUploadBackend, slots int, strides [3]int, budget uint64) (*MetalUpload, error) {
	if nilCompletion(b) || slots <= 0 || slots > 80*384 {
		return nil, errors.New("invalid Metal upload")
	}
	var bytes uint64
	for _, n := range strides {
		if n <= 0 || uint64(slots) > uint64(^uint(0)>>1)/uint64(n) {
			return nil, errors.New("invalid Metal expert stride")
		}
		size := uint64(slots) * uint64(n)
		if bytes > budget || size > budget-bytes {
			return nil, errors.New("Metal experts exceed budget")
		}
		bytes += size
	}
	if _, _, _, err := metalAllocationBytes(uint64(slots), [3]uint64{uint64(strides[0]), uint64(strides[1]), uint64(strides[2])}, b.pageSize(), budget); err != nil {
		return nil, err
	}
	u := &MetalUpload{backend: b, slots: slots, strides: strides}
	if err := b.allocate(slots, strides); err != nil {
		if cleanup := u.Close(); cleanup != nil {
			return u, errors.Join(err, cleanup)
		}
		return nil, err
	}
	return u, nil
}

type metalWriteCompletion struct{}

func (metalWriteCompletion) Wait() error { return nil }

// Submit copies a component into CPU-visible shared storage synchronously.
// GPU consumer completion is separate and must cover every later device read.
func (u *MetalUpload) Submit(upload Upload, component int, src []byte) (Completion, error) {
	if u == nil || u.closed || u.closing || u.failed || component < 0 || component >= 3 || upload.Slot < 0 || upload.Slot >= u.slots || len(src) != u.strides[component] || !u.acceptUpload(upload) {
		return nil, errors.New("invalid Metal expert write")
	}
	err := u.backend.write(upload.Slot, component, src)
	if err != nil {
		var pin *MetalPinFailure
		u.failed = !errors.As(err, &pin)
	}
	return metalWriteCompletion{}, err
}

// PinBackoff applies the pinned source's failure cap after staging has aborted.
// It refuses active consumers and never moves a live slot or rewrites a key.
// Zero means no reduced cap was established; mapped views can serve the request.
func (u *MetalUpload) PinBackoff(c *Cache, failure *MetalPinFailure) (int, error) {
	if u == nil || u.closed || u.closing || u.failed || c == nil || c != u.boundCache || c.closed || c.active != nil || failure == nil || (u.lease != nil && !u.lease.done) {
		return 0, errors.New("Metal pin backoff requires drained ownership")
	}
	entries := uint32(0)
	for _, s := range c.slots {
		if s.valid {
			entries++
		}
	}
	var expert uint64
	for _, n := range u.strides {
		expert += uint64(n)
	}
	cap := MetalPinCapacity(entries, failure.LockedSlots, failure.LockedBytes, expert)
	if cap == 0 {
		return 0, nil
	}
	cap = min(cap, uint32(len(c.slots)), uint32(u.slots))
	c.slots = c.slots[:cap]
	u.slots = int(cap)
	return int(cap), nil
}

func (u *MetalUpload) acceptUpload(upload Upload) bool {
	if u.staging == nil || u.staging.check() != nil || upload.Generation != u.staging.generation {
		return false
	}
	for _, expected := range u.staging.uploads {
		if upload == expected {
			return true
		}
	}
	return false
}

// Stage commits a lease only after every shared-buffer write completes.
// Graph replay with evictable slots is refused before any model read.
func (u *MetalUpload) Stage(source *Source, lease *Lease, graphCapture bool) error {
	if u == nil || u.closed || u.closing || u.failed {
		return errors.New("Metal upload closed")
	}
	if u.lease != nil && !u.lease.done {
		return errors.New("Metal consumer lease active")
	}
	if lease == nil || lease.check() != nil || (u.boundCache != nil && u.boundCache != lease.cache) {
		return errors.New("foreign Metal cache lease")
	}
	u.boundCache = lease.cache
	u.staging = lease
	defer func() { u.staging = nil }()
	if err := source.Stage(lease, u.backend.stage(), u, graphCapture); err != nil {
		return err
	}
	u.lease = lease
	return nil
}

// MetalSelection protects shared buffers until Release succeeds.
// An errored selection can still own a committed lease; keep it until teardown.
type MetalSelection struct {
	owner *MetalUpload
	lease *Lease
}

// Ready reports whether a selection has consumable committed buffers.
func (s *MetalSelection) Ready() bool { return s.check() == nil }

// Discard releases an uncommitted selection after a failed stage. It refuses
// committed selections, which require actual consumer completion via Release.
func (s *MetalSelection) Discard() error {
	if s == nil || s.lease == nil || s.lease.done {
		return nil
	}
	if s.lease.committed {
		return errors.New("committed Metal selection requires completion")
	}
	return s.lease.Abort()
}

// Select stages one layer's unique experts, preserving duplicate route order.
func (u *MetalUpload) Select(source *Source, cache *Cache, layer int, ids []int, graphCapture bool) (*MetalSelection, error) {
	return u.selectExperts(source, cache, layer, ids, graphCapture, true)
}

// SelectRoutes counts every routed occurrence, including mapped overflow, and
// reserves resident hits before admitting misses. Returned IDs identify the
// selected cache views; remaining routes must use the mapped source.
func (u *MetalUpload) SelectRoutes(source *Source, cache *Cache, layer int, ids []int) (*MetalSelection, []int, error) {
	if u == nil || u.closed || u.closing || u.failed || source == nil || cache == nil || (u.boundCache != nil && u.boundCache != cache) || source.identity != cache.identity || cache.active != nil || cache.closed {
		return nil, nil, errors.New("invalid Metal routes")
	}
	tr, ok := source.layers[layer]
	if !ok || layer < 0 || layer >= 80 {
		return nil, nil, errors.New("missing expert layer")
	}
	seen := make(map[int]bool)
	var unique []int
	for _, id := range ids {
		if id < 0 || id >= tr.Gate.Experts || id >= 384 {
			return nil, nil, errors.New("invalid selected expert")
		}
		if !seen[id] {
			unique = append(unique, id)
			seen[id] = true
		}
	}
	for _, id := range ids {
		p := &u.hotness.routes[layer][id]
		if *p != ^uint32(0) {
			*p++
		}
	}
	var chosen []int
	resident := make(map[int]bool)
	for _, id := range unique {
		for _, s := range cache.slots {
			if s.valid && s.key.Layer == layer && s.key.Expert == id {
				chosen = append(chosen, id)
				resident[id] = true
				break
			}
		}
	}
	for _, id := range unique {
		if len(chosen) == len(cache.slots) {
			break
		}
		if !resident[id] {
			chosen = append(chosen, id)
		}
	}
	if len(chosen) == 0 {
		return nil, nil, nil
	}
	s, err := u.selectExperts(source, cache, layer, chosen, false, false)
	return s, chosen, err
}

// SeedPriority sets an admitted hotlist priority after its upload has drained.
// It never changes a slot's identity or makes an absent expert resident.
func (u *MetalUpload) SeedPriority(layer, expert int, priority uint32) error {
	if u == nil || u.closed || u.closing || u.failed || u.boundCache == nil || u.boundCache.active != nil || layer < 0 || layer >= 80 || expert < 0 || expert >= 384 || priority == 0 {
		return errors.New("invalid Metal hotlist priority")
	}
	u.hotness.routes[layer][expert] = priority
	return nil
}

// NoteToken advances decode hotness aging at the first routed layer. Prefill
// callers do not advance this clock.
func (u *MetalUpload) NoteToken(layer int) error {
	if u == nil || u.closed || u.closing || u.failed {
		return errors.New("Metal hotness unavailable")
	}
	return u.hotness.NoteToken(layer)
}

// NoteMappedRoutes counts a mixed-size layer's routes without reserving slots.
// The whole request is validated before changing any hotness counter.
func (u *MetalUpload) NoteMappedRoutes(source *Source, layer int, ids []int32) error {
	if u == nil || u.closed || u.closing || u.failed || source == nil || (u.boundCache != nil && u.boundCache.identity != source.identity) || layer < 0 || layer >= 80 {
		return errors.New("Metal mapped routes unavailable")
	}
	tr, ok := source.layers[layer]
	if !ok {
		return errors.New("missing expert layer")
	}
	for _, id := range ids {
		if id < 0 || int(id) >= tr.Gate.Experts || id >= 384 {
			return errors.New("invalid selected expert")
		}
	}
	for _, id := range ids {
		p := &u.hotness.routes[layer][id]
		if *p != ^uint32(0) {
			*p++
		}
	}
	return nil
}

func (u *MetalUpload) selectExperts(source *Source, cache *Cache, layer int, ids []int, graphCapture, count bool) (*MetalSelection, error) {
	if graphCapture {
		return nil, errors.New("Metal streaming graph replay unqualified")
	}
	if u == nil || u.closed || u.closing || u.failed || source == nil || cache == nil || source.identity != cache.identity {
		return nil, errors.New("invalid Metal selection")
	}
	tr, ok := source.layers[layer]
	if !ok {
		return nil, errors.New("missing expert layer")
	}
	keys := make([]Key, len(ids))
	for i, id := range ids {
		if id < 0 || id >= tr.Gate.Experts {
			return nil, errors.New("invalid selected expert")
		}
		keys[i] = Key{Identity: source.identity, Layer: layer, Expert: id}
	}
	l, err := u.hotness.acquire(cache, keys, count)
	if err != nil {
		return nil, err
	}
	s := &MetalSelection{u, l}
	if err = u.Stage(source, l, false); err != nil {
		if !l.done {
			err = errors.Join(err, l.Abort())
		}
		return s, err
	}
	return s, nil
}

func (s *MetalSelection) check() error {
	if s == nil || s.owner == nil || s.lease == nil || s.owner.closed || s.owner.closing || s.owner.failed || s.owner.lease != s.lease || !s.lease.committed {
		return errors.New("inactive Metal selection")
	}
	return s.lease.check()
}

// Slots returns detached remapped IDs for the active committed selection.
func (s *MetalSelection) Slots() ([]int, error) {
	if err := s.check(); err != nil {
		return nil, err
	}
	return s.lease.Slots(), nil
}

// Release waits for all device reads before permitting eviction or teardown.
// A failed wait retains ownership and can be retried.
func (s *MetalSelection) Release(done Completion) error {
	if err := s.check(); err != nil {
		return err
	}
	return s.lease.Release(done)
}

// Close refuses active consumers and preserves partially released ownership.
func (u *MetalUpload) Close() error {
	if u == nil || u.closed {
		return nil
	}
	if u.lease != nil && !u.lease.done {
		return errors.New("Metal consumer lease active")
	}
	u.closing = true
	if err := u.backend.close(); err != nil {
		return fmt.Errorf("close Metal expert storage: %w", err)
	}
	u.closed = true
	return nil
}
