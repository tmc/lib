//go:build linux

package main

// A cudaWeightTraffic records the weight bytes each matvec kernel must read.
//
// The decode step is weight-streaming: every matvec reads a whole weight plane
// to produce one output column, and nothing it reads is reused before it is
// evicted. So the bytes a step moves are the sum of the planes its matvecs
// touch, and that sum is known at the launch site from the tensor's own extent
// -- no profiler, no sampling, and no dependence on the driver reporting DRAM
// counters. Activations are deliberately excluded: they are kilobytes against
// megabytes and they hit in L2.
//
// This exists to settle one question. Dividing device time by an ASSUMED
// bytes/token says whether we are near the bandwidth roofline, and that reading
// is only as good as the assumption. If we move the bytes we think we do, the
// remaining gap is efficiency; if we move more, it is traffic, and the two have
// different remedies.
//
// The zero value counts nothing. Counting costs a map store per matvec launch,
// against a kernel that runs for tens of microseconds, so it is left on.
type cudaWeightTraffic struct {
	bytes    map[string]uint64
	launches map[string]uint64
}

// countWeightBytes adds one matvec launch to the ledger when a caller has
// opened one. kernel is the device kernel's own name, so the ledger can be
// joined against a CUPTI activity trace by name.
func (r *cudaFoundation) countWeightBytes(kernel string, bytes uint64) {
	if r.traffic == nil {
		return
	}
	r.traffic.bytes[kernel] += bytes
	r.traffic.launches[kernel]++
}

// openWeightTraffic starts a ledger, discarding any previous one, and returns
// the totals accumulated since the last call.
func (r *cudaFoundation) openWeightTraffic() {
	r.traffic = &cudaWeightTraffic{
		bytes:    make(map[string]uint64),
		launches: make(map[string]uint64),
	}
}

// weightTraffic returns the open ledger, or nil.
func (r *cudaFoundation) weightTraffic() *cudaWeightTraffic {
	return r.traffic
}

// total is the ledger's summed bytes.
func (t *cudaWeightTraffic) total() uint64 {
	var n uint64
	for _, b := range t.bytes {
		n += b
	}
	return n
}
