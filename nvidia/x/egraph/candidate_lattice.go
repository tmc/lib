package egraph

import (
	"fmt"

	"github.com/tmc/lib/egraphcore"
)

// CandidateLatticeFinalist is one individually viable NVIDIA finalist.
type CandidateLatticeFinalist[T any] = egraphcore.CandidateLatticeFinalist[T]

// CandidateLatticeBucket supplies ranked, individually viable finalists.
type CandidateLatticeBucket[T any] = egraphcore.CandidateLatticeBucket[T]

// CandidateLattice proposes bounded best-first combinations of bucket
// finalists. The implementation is shared with other backends.
type CandidateLattice[T any] = egraphcore.CandidateLattice[T]

// NewCandidateLattice creates a bounded best-first coordinator over ranked
// bucket finalists. aggregate must be deterministic, leave its input
// unchanged, and be coordinate-monotone: increasing one bucket score must not
// lower the result.
func NewCandidateLattice[T any](buckets []CandidateLatticeBucket[T], aggregate func([]float64) float64, maxSets, maxFinalistsPerBucket int) (*CandidateLattice[T], error) {
	return egraphcore.NewCandidateLattice(buckets, aggregate, maxSets, maxFinalistsPerBucket)
}

// TuningResultsBucket adapts one measured NVIDIA ranking to a lazy lattice
// finalist source. The results must come from the same workload and be in
// latency order. Expression nodes are cloned; payload data follows the graph's
// cloning rules and must be treated as immutable.
func TuningResultsBucket(results []RankedTuningResult) (CandidateLatticeBucket[TuningCandidate], error) {
	if len(results) == 0 {
		return CandidateLatticeBucket[TuningCandidate]{}, fmt.Errorf("egraph: tuning result bucket is empty")
	}
	cloned := make([]RankedTuningResult, len(results))
	for i, result := range results {
		if result.Latency <= 0 {
			return CandidateLatticeBucket[TuningCandidate]{}, fmt.Errorf("egraph: tuning result %d has non-positive latency", i)
		}
		if i > 0 && result.Latency < results[i-1].Latency {
			return CandidateLatticeBucket[TuningCandidate]{}, fmt.Errorf("egraph: tuning results are not ranked")
		}
		cloned[i] = RankedTuningResult{Candidate: cloneTuningCandidate(result.Candidate), Latency: result.Latency}
	}
	return CandidateLatticeBucket[TuningCandidate]{
		LoadFinalist: func(rank int) (CandidateLatticeFinalist[TuningCandidate], bool, error) {
			if rank < 0 {
				return CandidateLatticeFinalist[TuningCandidate]{}, false, fmt.Errorf("egraph: negative finalist rank %d", rank)
			}
			if rank >= len(cloned) {
				return CandidateLatticeFinalist[TuningCandidate]{}, false, nil
			}
			result := cloned[rank]
			return CandidateLatticeFinalist[TuningCandidate]{
				Score: float64(result.Latency),
				Value: cloneTuningCandidate(result.Candidate),
			}, true, nil
		},
	}, nil
}
