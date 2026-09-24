package cudasass

// CoverageMapSize is the number of counters in an edge coverage map, cuFuzz's
// MAP_SIZE.
const CoverageMapSize = 65536

// EdgeIndex returns the coverage map index of the transition into block from
// the location prev, and the location to pass for the next transition.
//
// It is the AFL edge hash as cuFuzz's record_coverage_edge_count computes it:
// the index is (MAP_SIZE/2) + ((prev ^ block) % (MAP_SIZE/2)), with block
// sign-extended to 64 bits as C converts an int, and the next location is
// block >> 1, an arithmetic shift. The shift makes A→B and B→A different
// edges. Indices therefore lie in the upper half of the map.
//
// Block IDs are the caller's. cuFuzz derives them from a hash of the function
// name and the block's position in NVBit's CFG_t, and a [CFG] block index can
// be used the same way.
func EdgeIndex(prev uint64, block int32) (index int, next uint64) {
	const half = CoverageMapSize / 2
	index = half + int((prev^uint64(int64(block)))%half)
	return index, uint64(int64(block >> 1))
}

// An EdgeCoverage accumulates edge counts for many threads, each with its own
// previous location. Its zero value is empty and ready to use.
type EdgeCoverage struct {
	counts [CoverageMapSize]uint32
	prev   map[uint64]uint64
}

// Record records that thread entered block and returns the index it counted.
// A thread's first transition is from location 0.
func (c *EdgeCoverage) Record(thread uint64, block int32) int {
	if c.prev == nil {
		c.prev = make(map[uint64]uint64)
	}
	index, next := EdgeIndex(c.prev[thread], block)
	c.prev[thread] = next
	c.counts[index]++
	return index
}

// RecordWarp records that the active lanes of one warp, threads[0] first,
// entered block together, as cuFuzz's injected function does: every lane
// advances its own previous location, and only the first lane's edge is
// counted, once. It returns that index, or -1 if threads is empty.
func (c *EdgeCoverage) RecordWarp(threads []uint64, block int32) int {
	if len(threads) == 0 {
		return -1
	}
	if c.prev == nil {
		c.prev = make(map[uint64]uint64)
	}
	first, _ := EdgeIndex(c.prev[threads[0]], block)
	for _, t := range threads {
		_, c.prev[t] = EdgeIndex(c.prev[t], block)
	}
	c.counts[first]++
	return first
}

// Count returns the number of transitions counted at index.
func (c *EdgeCoverage) Count(index int) uint32 {
	return c.counts[index]
}

// Bitmap returns the map as CoverageMapSize bytes, each count reduced to
// cuFuzz's bucket: 0, 1 and 2 are themselves, then 3 for 3-511, 4 for
// 512-4095, 5 for 4096-16383, 6 for 16384-65535 and 7 above.
func (c *EdgeCoverage) Bitmap() []byte {
	b := make([]byte, CoverageMapSize)
	for i, n := range c.counts {
		b[i] = bucket(n)
	}
	return b
}

func bucket(n uint32) byte {
	switch {
	case n <= 2:
		return byte(n)
	case n < 512:
		return 3
	case n < 4096:
		return 4
	case n < 16384:
		return 5
	case n < 65536:
		return 6
	}
	return 7
}
