package cuptipcsampling

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

var (
	// ErrRingBufferFull indicates that all slots in the circular ring buffer
	// are currently in flight and no free buffer is available.
	ErrRingBufferFull = errors.New("cuptipcsampling: ring buffer full")

	// ErrRingBufferClosed indicates that the ring buffer has been shut down.
	ErrRingBufferClosed = errors.New("cuptipcsampling: ring buffer closed")
)

// MultiBuffer is an alias for [RingBuffer] for multi-buffering PC sampling and
// CUPTI activity streams.
type MultiBuffer = RingBuffer

// RingBufferStats holds performance and diagnostic counters for a [RingBuffer].
type RingBufferStats struct {
	Capacity      int    // Total number of buffers configured in the ring
	Available     int    // Number of free buffers ready for acquisition
	InFlight      int    // Number of buffers currently in flight (acquired or queued)
	Ready         int    // Number of filled buffers waiting for consumption
	HighWaterMark int    // Maximum number of concurrently in-flight buffers observed
	Overflows     uint64 // Count of acquire attempts that failed due to a full ring
	Drops         uint64 // Cumulative count of dropped records or samples
}

// RingBufferConfig configures a circular [RingBuffer].
// At least one of PCCount or ActivitySize must be positive.
type RingBufferConfig struct {
	Capacity     int    // Buffer count in the ring; must be at least 2
	PCCount      int    // PC record capacity per buffer (for PC sampling)
	ReasonCount  int    // Reason capacity per PC (for PC sampling)
	MaxBytes     uint64 // Maximum bytes per PC sampling buffer
	ActivitySize int    // Raw activity buffer byte capacity (multiple of 8)
}

// RingSlot is a pinned buffer slot managed by a [RingBuffer].
// Its zero value is not usable.
type RingSlot struct {
	index       int
	sampling    *samplingBuffer
	activity    []byte
	words       []uint64
	activityPin runtime.Pinner
	validBytes  int
	dropped     uint64
}

// Index returns the zero-based slot index in the ring buffer.
func (s *RingSlot) Index() int { return s.index }

// SamplingBuffer returns the underlying PC sampling buffer, or nil if this slot
// was configured solely for activity streams.
func (s *RingSlot) SamplingBuffer() *samplingBuffer { return s.sampling }

// ActivityBuffer returns the raw activity byte slice, or nil if this slot
// was configured solely for PC sampling.
func (s *RingSlot) ActivityBuffer() []byte { return s.activity }

// ActivityPointer returns the pinned uintptr to the start of the activity
// buffer, or 0 if unconfigured.
func (s *RingSlot) ActivityPointer() uintptr {
	if len(s.words) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.words[0]))
}

// ActivitySize returns the capacity of the raw activity buffer in bytes.
func (s *RingSlot) ActivitySize() int { return len(s.activity) }

// ValidBytes returns the number of valid bytes written into the activity buffer.
func (s *RingSlot) ValidBytes() int { return s.validBytes }

// SetValidBytes sets the number of valid bytes written into the activity buffer.
func (s *RingSlot) SetValidBytes(n int) { s.validBytes = n }

// Dropped returns the number of dropped samples recorded for this slot.
func (s *RingSlot) Dropped() uint64 { return s.dropped }

// SetDropped records dropped samples associated with this slot.
func (s *RingSlot) SetDropped(n uint64) { s.dropped = n }

// Fetch drains CUPTI PC sampling records directly into this slot's sampling buffer.
func (s *RingSlot) Fetch(ctx cupti.Context, version uint32, query func(*cupti.PCSamplingGetDataParams) error) (samplingBatch, error) {
	if s.sampling == nil {
		return samplingBatch{}, errors.New("cuptipcsampling: slot has no sampling buffer")
	}
	return s.sampling.fetch(ctx, version, query)
}

// Snapshot decodes PC sampling records from this slot without modifying name ownership.
func (s *RingSlot) Snapshot(copyName func(uintptr) (string, error)) (samplingBatch, error) {
	if s.sampling == nil {
		return samplingBatch{}, errors.New("cuptipcsampling: slot has no sampling buffer")
	}
	return s.sampling.snapshot(copyName)
}

// SnapshotOwned decodes PC sampling records and registers native function names.
func (s *RingSlot) SnapshotOwned(names *samplingNames) (samplingBatch, error) {
	if s.sampling == nil {
		return samplingBatch{}, errors.New("cuptipcsampling: slot has no sampling buffer")
	}
	return s.sampling.snapshotOwned(names)
}

func (s *RingSlot) reset() {
	if s.sampling != nil {
		s.sampling.reset()
	}
	s.validBytes = 0
	s.dropped = 0
}

func (s *RingSlot) release() {
	if s.sampling != nil {
		s.sampling.release()
	}
	if len(s.words) > 0 {
		s.activityPin.Unpin()
		s.activity = nil
		s.words = nil
	}
}

// RingBuffer coordinates zero-copy circular multi-buffering for PC sampling and
// CUPTI activity streams. Its zero value is not usable; use [NewRingBuffer] or
// [NewActivityRingBuffer].
type RingBuffer struct {
	mu            sync.Mutex
	cond          *sync.Cond
	cfg           RingBufferConfig
	slots         []RingSlot
	freeQueue     []*RingSlot
	readyQueue    []*RingSlot
	freeHead      int
	freeTail      int
	freeCount     int
	readyHead     int
	readyTail     int
	readyCount    int
	highWaterMark int
	overflows     uint64
	drops         uint64
	closed        bool
	released      bool
}

// NewRingBuffer allocates a zero-copy circular ring buffer with capacity pinned
// PC sampling buffers. Capacity must be at least 2.
func NewRingBuffer(capacity int, pcCount, reasonCount int, maxBytes uint64) (*RingBuffer, error) {
	return NewRingBufferWithConfig(RingBufferConfig{
		Capacity:    capacity,
		PCCount:     pcCount,
		ReasonCount: reasonCount,
		MaxBytes:    maxBytes,
	})
}

// NewMultiBuffer creates a zero-copy circular multi-buffer for PC sampling.
func NewMultiBuffer(capacity int, pcCount, reasonCount int, maxBytes uint64) (*MultiBuffer, error) {
	return NewRingBuffer(capacity, pcCount, reasonCount, maxBytes)
}

// NewActivityRingBuffer allocates a zero-copy circular ring buffer with capacity
// pinned buffers for CUPTI activity streams. BufferBytes must be a positive multiple of 8.
func NewActivityRingBuffer(capacity int, bufferBytes int) (*RingBuffer, error) {
	return NewRingBufferWithConfig(RingBufferConfig{
		Capacity:     capacity,
		ActivitySize: bufferBytes,
	})
}

// NewRingBufferWithConfig constructs a ring buffer with the given configuration.
func NewRingBufferWithConfig(cfg RingBufferConfig) (*RingBuffer, error) {
	if cfg.Capacity < 2 {
		return nil, errors.New("cuptipcsampling: ring buffer capacity must be at least 2")
	}
	if cfg.PCCount <= 0 && cfg.ActivitySize <= 0 {
		return nil, errors.New("cuptipcsampling: ring buffer requires positive PC count or activity size")
	}
	if cfg.ActivitySize > 0 && (cfg.ActivitySize < 8 || cfg.ActivitySize%8 != 0) {
		return nil, errors.New("cuptipcsampling: activity buffer size must be a positive multiple of 8")
	}

	rb := &RingBuffer{
		cfg:        cfg,
		slots:      make([]RingSlot, cfg.Capacity),
		freeQueue:  make([]*RingSlot, cfg.Capacity),
		readyQueue: make([]*RingSlot, cfg.Capacity),
		freeCount:  cfg.Capacity,
	}
	rb.cond = sync.NewCond(&rb.mu)

	// Pre-allocate and pin all buffers up front to guarantee zero allocations
	// during steady-state buffer rotation.
	for i := 0; i < cfg.Capacity; i++ {
		slot := &rb.slots[i]
		slot.index = i

		if cfg.PCCount > 0 {
			buf, err := newSamplingBuffer(cfg.PCCount, cfg.ReasonCount, cfg.MaxBytes)
			if err != nil {
				rb.teardown()
				return nil, fmt.Errorf("cuptipcsampling: allocate sampling buffer %d: %w", i, err)
			}
			slot.sampling = buf
		}

		if cfg.ActivitySize > 0 {
			words := make([]uint64, cfg.ActivitySize/8)
			slot.activityPin.Pin(&words[0])
			slot.words = words
			slot.activity = unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), cfg.ActivitySize)
		}

		rb.freeQueue[i] = slot
	}

	return rb, nil
}

// Acquire retrieves the next free buffer slot from the ring buffer without blocking.
// If all buffers are currently in flight, Acquire increments the overflow counter
// and returns ErrRingBufferFull.
func (r *RingBuffer) Acquire() (*RingSlot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil, ErrRingBufferClosed
	}
	if r.freeCount == 0 {
		r.overflows++
		return nil, ErrRingBufferFull
	}

	slot := r.freeQueue[r.freeHead]
	r.freeQueue[r.freeHead] = nil
	r.freeHead = (r.freeHead + 1) % len(r.slots)
	r.freeCount--

	inFlight := len(r.slots) - r.freeCount
	if inFlight > r.highWaterMark {
		r.highWaterMark = inFlight
	}
	return slot, nil
}

// Produce hands a filled buffer to the ready queue for concurrent draining
// by a background consumer and signals any waiting consumers.
func (r *RingBuffer) Produce(slot *RingSlot) error {
	if slot == nil {
		return errors.New("cuptipcsampling: nil slot")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return ErrRingBufferClosed
	}

	r.readyQueue[r.readyTail] = slot
	r.readyTail = (r.readyTail + 1) % len(r.slots)
	r.readyCount++
	r.cond.Signal()
	return nil
}

// Handoff provides an atomic, non-blocking handoff: it enqueues the filled buffer
// to the ready queue for concurrent background draining and retrieves the next free
// buffer to hand to CUPTI. If all remaining buffers are full, the filled buffer is still
// safely queued for consumption, the overflow counter is incremented, and ErrRingBufferFull is returned.
func (r *RingBuffer) Handoff(filled *RingSlot) (*RingSlot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil, ErrRingBufferClosed
	}

	if filled != nil {
		r.readyQueue[r.readyTail] = filled
		r.readyTail = (r.readyTail + 1) % len(r.slots)
		r.readyCount++
		r.cond.Signal()
	}

	if r.freeCount == 0 {
		r.overflows++
		return nil, ErrRingBufferFull
	}

	next := r.freeQueue[r.freeHead]
	r.freeQueue[r.freeHead] = nil
	r.freeHead = (r.freeHead + 1) % len(r.slots)
	r.freeCount--

	inFlight := len(r.slots) - r.freeCount
	if inFlight > r.highWaterMark {
		r.highWaterMark = inFlight
	}
	return next, nil
}

// Consume waits for and removes the next filled buffer from the ready queue.
// It blocks until a buffer is produced or the ring buffer is closed.
// When the ring buffer is closed and all queued buffers have been consumed,
// Consume returns ErrRingBufferClosed.
func (r *RingBuffer) Consume() (*RingSlot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for r.readyCount == 0 && !r.closed {
		r.cond.Wait()
	}

	if r.readyCount == 0 && r.closed {
		return nil, ErrRingBufferClosed
	}

	slot := r.readyQueue[r.readyHead]
	r.readyQueue[r.readyHead] = nil
	r.readyHead = (r.readyHead + 1) % len(r.slots)
	r.readyCount--
	return slot, nil
}

// TryConsume retrieves the next filled buffer from the ready queue without blocking.
// If no buffer is ready, it returns nil, nil.
func (r *RingBuffer) TryConsume() (*RingSlot, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.readyCount == 0 {
		if r.closed {
			return nil, ErrRingBufferClosed
		}
		return nil, nil
	}

	slot := r.readyQueue[r.readyHead]
	r.readyQueue[r.readyHead] = nil
	r.readyHead = (r.readyHead + 1) % len(r.slots)
	r.readyCount--
	return slot, nil
}

// Release resets a drained buffer and returns it to the free queue for reuse.
// It performs zero allocations.
func (r *RingBuffer) Release(slot *RingSlot) {
	if slot == nil {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.released {
		return
	}

	slot.reset()
	r.freeQueue[r.freeTail] = slot
	r.freeTail = (r.freeTail + 1) % len(r.slots)
	r.freeCount++

	if r.closed && r.freeCount == len(r.slots) {
		r.teardown()
	}
}

// Abandon returns an acquired buffer slot that was not produced (for example,
// when a native fetch call encountered an error) back to the free queue.
func (r *RingBuffer) Abandon(slot *RingSlot) {
	if slot == nil {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.released {
		return
	}

	slot.reset()
	r.freeQueue[r.freeTail] = slot
	r.freeTail = (r.freeTail + 1) % len(r.slots)
	r.freeCount++

	if r.closed && r.freeCount == len(r.slots) {
		r.teardown()
	}
}

// RecordDrops atomically records sample drops (e.g. from CUPTI hardware exhaustion).
func (r *RingBuffer) RecordDrops(n uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.drops += n
}

// Stats returns a snapshot of ring buffer counters and watermarks.
func (r *RingBuffer) Stats() RingBufferStats {
	r.mu.Lock()
	defer r.mu.Unlock()

	return RingBufferStats{
		Capacity:      len(r.slots),
		Available:     r.freeCount,
		InFlight:      len(r.slots) - r.freeCount,
		Ready:         r.readyCount,
		HighWaterMark: r.highWaterMark,
		Overflows:     r.overflows,
		Drops:         r.drops,
	}
}

// teardown unpins all allocated buffers. Must be called while holding r.mu or during init.
func (r *RingBuffer) teardown() {
	if r.released {
		return
	}
	r.released = true
	for i := range r.slots {
		r.slots[i].release()
	}
}

// Close marks the ring buffer as closed, wakes any blocked consumers, and unpins
// all underlying buffers once all in-flight buffers have returned. It is idempotent.
func (r *RingBuffer) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil
	}
	r.closed = true
	r.cond.Broadcast()
	if r.freeCount == len(r.slots) {
		r.teardown()
	}
	return nil
}
