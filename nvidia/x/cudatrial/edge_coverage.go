package cudatrial

import "fmt"

// EdgeBitmap holds saturating edge counters. It models a trace supplied by a
// producer; its contents alone do not prove device execution or CFG coverage.
type EdgeBitmap struct{ counts []byte }

// NewEdgeBitmap allocates an even-sized map, bounded to 16 MiB.
func NewEdgeBitmap(size int) (*EdgeBitmap, error) {
	if size < 2 || size > 16<<20 || size%2 != 0 {
		return nil, fmt.Errorf("edge bitmap: size must be even and between 2 and 16777216")
	}
	return &EdgeBitmap{counts: make([]byte, size)}, nil
}

// Bytes returns a copy of the raw counters.
func (b *EdgeBitmap) Bytes() []byte { return append([]byte(nil), b.counts...) }

// Reset clears counters. It does not reset a CUDA context or recover a fault.
func (b *EdgeBitmap) Reset() { clear(b.counts) }

// Merge takes the counter-wise maximum. A mismatched map is rejected before
// changing the receiver.
func (b *EdgeBitmap) Merge(other *EdgeBitmap) error {
	if b == nil || other == nil || len(b.counts) != len(other.counts) || len(b.counts) == 0 {
		return fmt.Errorf("edge bitmap: mismatched map size")
	}
	for i, n := range other.counts {
		if n > b.counts[i] {
			b.counts[i] = n
		}
	}
	return nil
}

// Quantized returns AFL-style hit-count classes: 0, 1, 2, 4, 8, 16, 32, 64,
// and 128. Repeated calls do not alter the raw counters.
func (b *EdgeBitmap) Quantized() []byte {
	out := b.Bytes()
	for i, n := range out {
		switch {
		case n <= 1:
		case n <= 3:
			out[i] = 2
		case n <= 7:
			out[i] = 4
		case n <= 15:
			out[i] = 8
		case n <= 31:
			out[i] = 16
		case n <= 63:
			out[i] = 32
		case n <= 127:
			out[i] = 64
		default:
			out[i] = 128
		}
	}
	return out
}

// ThreadEdgeMap tracks an independent predecessor for each declared thread.
// It is not safe for concurrent use. Block identities must come from verified
// CFG instrumentation, never CUDA thread-block entry notifications.
type ThreadEdgeMap struct {
	bitmap   *EdgeBitmap
	previous []uint64
}

// NewThreadEdgeMap bounds both the bitmap and the number of thread states.
func NewThreadEdgeMap(size, threads int) (*ThreadEdgeMap, error) {
	if threads <= 0 || threads > 1<<20 {
		return nil, fmt.Errorf("thread edge map: invalid thread count")
	}
	b, err := NewEdgeBitmap(size)
	if err != nil {
		return nil, err
	}
	return &ThreadEdgeMap{bitmap: b, previous: make([]uint64, threads)}, nil
}

// Record applies cuFuzz's predecessor XOR formula in the map's upper half.
// Counters saturate at 255. The block becomes this thread's predecessor.
func (m *ThreadEdgeMap) Record(thread int, block uint64) (int, error) {
	if m == nil || thread < 0 || thread >= len(m.previous) {
		return 0, fmt.Errorf("thread edge map: invalid thread")
	}
	half := uint64(len(m.bitmap.counts) / 2)
	i := int(half + (m.previous[thread]^block)%half)
	if m.bitmap.counts[i] < 255 {
		m.bitmap.counts[i]++
	}
	m.previous[thread] = block
	return i, nil
}

// Snapshot returns an independent bitmap copy.
func (m *ThreadEdgeMap) Snapshot() *EdgeBitmap { return &EdgeBitmap{counts: m.bitmap.Bytes()} }

// Reset starts a new trace by clearing counters and per-thread predecessors.
func (m *ThreadEdgeMap) Reset() { m.bitmap.Reset(); clear(m.previous) }
