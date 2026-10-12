package cudamega

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// PolicyFor returns the selection policy g's shape calls for: [WavePolicy]
// when every rank holds one task, and [CountedPolicy] when some rank holds
// more than one.
//
// A rank of width one has nothing to run beside its task, so dealing it the
// whole grid loses nothing; a wider rank has independent tasks that only
// counted dispatch runs at the same time. The choice reads rank widths and
// nothing else. Whether counted dispatch is faster than the wave at a given
// rank width is unmeasured: the policy follows from which tasks may overlap,
// not from a timing.
//
// A graph with no tasks has no shape and is refused, rather than given the
// zero Policy.
func PolicyFor(g cudataskgraph.Graph) (Policy, error) {
	widths := RankWidths(g)
	if len(widths) == 0 {
		return WavePolicy, fmt.Errorf("graph has no tasks to choose a policy for")
	}
	for _, w := range widths {
		if w > 1 {
			return CountedPolicy, nil
		}
	}
	return WavePolicy, nil
}

// An Occupancy is the driver's occupancy measurement of one compiled worker
// on one device, taken by the caller. This package makes no device calls.
//
// Zero in any field is refused by [AdmissionFromOccupancy]. It cannot tell an
// unmeasured field from a measured zero, and needs neither: no grid can be
// derived from either.
type Occupancy struct {
	// BlocksPerMultiprocessor is what
	// cuOccupancyMaxActiveBlocksPerMultiprocessor reports for the worker at
	// WorkerThreads threads and SharedBytes of dynamic shared memory.
	BlocksPerMultiprocessor uint32

	// Multiprocessors is the device's
	// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
	Multiprocessors uint32

	// WorkerThreads and SharedBytes are the block size and dynamic shared
	// memory the occupancy was queried at.
	WorkerThreads uint32
	SharedBytes   uint32
}

// AdmissionFromOccupancy derives the worker admission from an occupancy
// measurement: the resident CTAs are the blocks per multiprocessor times the
// multiprocessor count, exactly, and the thread and shared-memory figures are
// the ones the measurement was taken at.
//
// It refuses a zero field, a product that does not fit in 32 bits, and an
// admission [Admission.Validate] rejects. On refusal the returned Admission is
// the zero value, which Validate also rejects.
func AdmissionFromOccupancy(o Occupancy) (Admission, error) {
	for _, f := range []struct {
		name string
		v    uint32
	}{
		{"blocks per multiprocessor", o.BlocksPerMultiprocessor},
		{"multiprocessor count", o.Multiprocessors},
		{"worker threads", o.WorkerThreads},
		{"shared bytes", o.SharedBytes},
	} {
		if f.v == 0 {
			return Admission{}, fmt.Errorf("occupancy %s is zero: unmeasured, or the worker does not fit", f.name)
		}
	}
	ctas := uint64(o.BlocksPerMultiprocessor) * uint64(o.Multiprocessors)
	if ctas > math.MaxUint32 {
		return Admission{}, fmt.Errorf("occupancy %d blocks on %d multiprocessors overflows the resident CTA count", o.BlocksPerMultiprocessor, o.Multiprocessors)
	}
	a := Admission{ResidentCTAs: uint32(ctas), WorkerThreads: o.WorkerThreads, SharedBytes: o.SharedBytes}
	if err := a.Validate(); err != nil {
		return Admission{}, fmt.Errorf("occupancy: %w", err)
	}
	return a, nil
}
