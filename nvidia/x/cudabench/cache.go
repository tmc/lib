package cudabench

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Evict runs fn between timed iterations to disturb device cache state. Its
// cost is excluded from the reported time: once an evictor is installed, each
// iteration is timed with its own event pair and the reported total is the sum
// of those spans, so the eviction happens in the gaps between them.
//
// Installing an evictor changes what the benchmark measures, and is meant to.
// A tight repeat loop over a working set that fits in L2 measures a
// progressively warmer cache and converges on a number that says nothing about
// the kernel's cost in a program where that data is cold. Which of the two is
// the right question depends on the workload; the point is that without an
// evictor the choice is made silently.
//
// fn must enqueue its work on the stream it is given, and must not synchronize.
func (b *B) Evict(fn func(stream *cuda.Stream) error) {
	b.evict = fn
	b.evictName = "caller-supplied"
}

// EvictL2 installs an evictor that overwrites a buffer larger than the device's
// L2 cache, which displaces whatever the previous iteration left resident. It
// returns the buffer's free function, which the caller must defer.
//
// The multiple of L2 size is the caller's: 1 is not enough, since a write
// stream and the working set share the cache. 2 is the usual choice.
//
// What this does and does not establish: it makes L2 residency of the previous
// iteration's data unlikely. It is not a cache flush, there is no CUDA API for
// one, and it does not touch L1 or the DRAM row buffers. A benchmark that needs
// to prove eviction happened must measure it with traffic counters, which is
// cuptibench's tier, not this package's.
func EvictL2(b *B, multiple int) (free func() error, err error) {
	if multiple < 1 {
		return nil, fmt.Errorf("cudabench: L2 evictor multiple %d must be at least 1", multiple)
	}
	l2, err := b.dev.Attribute(driver.DevAttrL2CacheSize)
	if err != nil {
		return nil, fmt.Errorf("cudabench: L2 cache size: %w", err)
	}
	if l2 <= 0 {
		return nil, fmt.Errorf("cudabench: device reports L2 cache size %d; cannot size an evictor", l2)
	}
	n := l2 * multiple
	buf, err := cudamem.Alloc[byte](b.dev, n)
	if err != nil {
		return nil, fmt.Errorf("cudabench: allocate %d-byte evictor: %w", n, err)
	}
	b.evict = func(stream *cuda.Stream) error {
		return buf.Ptr().MemsetD8Async(0, uint(n), stream.Raw())
	}
	b.evictName = fmt.Sprintf("L2 overwrite, %d bytes (%dx the %d-byte L2)", n, multiple, l2)
	return buf.Free, nil
}
