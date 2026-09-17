package cudacmp

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudacensus"
)

// SequenceConfig specifies parameters for launch sequence and gap analysis.
type SequenceConfig struct {
	// MinGapNs is the minimum duration in nanoseconds between consecutive launches
	// on the same stream to be reported as an observed gap. Defaults to 1,000 ns (1 µs).
	MinGapNs int64

	// MinSequenceLen is the minimum pattern length (number of kernels) for repeated sequence detection.
	// Defaults to 2.
	MinSequenceLen int

	// MinRepetitions is the minimum number of consecutive repetitions of a pattern to report.
	// Defaults to 2.
	MinRepetitions int

	// CandidateBatchingMinLaunches is the minimum number of consecutive kernels in a candidate batching region.
	// Defaults to 3.
	CandidateBatchingMinLaunches int

	// CandidateBatchingMaxGapNs is the maximum allowable gap duration between adjacent launches within a candidate
	// batching region. Defaults to 5,000 ns (5 µs).
	CandidateBatchingMaxGapNs int64
}

func (cfg *SequenceConfig) withDefaults() SequenceConfig {
	c := *cfg
	if c.MinGapNs <= 0 {
		c.MinGapNs = 1_000
	}
	if c.MinSequenceLen <= 0 {
		c.MinSequenceLen = 2
	}
	if c.MinRepetitions <= 0 {
		c.MinRepetitions = 2
	}
	if c.CandidateBatchingMinLaunches <= 0 {
		c.CandidateBatchingMinLaunches = 3
	}
	if c.CandidateBatchingMaxGapNs <= 0 {
		c.CandidateBatchingMaxGapNs = 5_000
	}
	return c
}

// Gap records an observed execution gap between consecutive kernel launches on the same stream.
type Gap struct {
	StreamID        int64
	StartNs         int64
	EndNs           int64
	GapNs           int64
	PrecedingKernel string
	FollowingKernel string
	MissingEvidence []string
}

// RepeatedSequence records a pattern of kernel launches repeating consecutively on the same stream.
type RepeatedSequence struct {
	StreamID        int64
	Pattern         []string
	Repetitions     int
	TotalDurationNs int64
	Intervals       []Interval
	MissingEvidence []string
}

// BatchingCandidate identifies a contiguous region of closely spaced kernel launches on a stream
// that may be candidates for CUDA graph capture or operator batching.
type BatchingCandidate struct {
	StreamID        int64
	StartNs         int64
	EndNs           int64
	LaunchCount     int
	Kernels         []string
	Hypothesis      string
	MissingEvidence []string
}

// SequenceAnalysis contains the structural launch analysis across streams.
type SequenceAnalysis struct {
	StreamIDs          []int64
	Gaps               []Gap
	RepeatedSequences  []RepeatedSequence
	BatchingCandidates []BatchingCandidate
}

// AnalyzeLaunchSequences performs a pure analysis pass over kernel launch records,
// identifying per-stream repeated sequences, execution gaps, and candidate batching regions.
//
// Because activity records provide only timing and symbols, findings are reported as hypotheses
// accompanied by explicit lists of missing evidence (e.g. unobserved host activity or unverified
// graph safety), rather than speculative promises of recoverable latency.
func AnalyzeLaunchSequences(launches []cudacensus.Launch, cfg SequenceConfig) (SequenceAnalysis, error) {
	c := cfg.withDefaults()

	// Validate launch intervals
	for i, l := range launches {
		if l.EndNs < l.StartNs {
			return SequenceAnalysis{}, fmt.Errorf("cudacmp: launch %d %q ends before start (%d < %d)", i, l.Name, l.EndNs, l.StartNs)
		}
		if l.StartNs < 0 || l.EndNs < 0 {
			return SequenceAnalysis{}, fmt.Errorf("cudacmp: launch %d %q has negative timestamp", i, l.Name)
		}
	}

	byStream := make(map[int64][]cudacensus.Launch)
	for _, l := range launches {
		byStream[l.StreamID] = append(byStream[l.StreamID], l)
	}

	streamIDs := make([]int64, 0, len(byStream))
	for sid := range byStream {
		streamIDs = append(streamIDs, sid)
	}
	sort.Slice(streamIDs, func(i, j int) bool { return streamIDs[i] < streamIDs[j] })

	var allGaps []Gap
	var allRepeated []RepeatedSequence
	var allCandidates []BatchingCandidate

	for _, sid := range streamIDs {
		streamLaunches := byStream[sid]
		sort.SliceStable(streamLaunches, func(i, j int) bool {
			if streamLaunches[i].StartNs != streamLaunches[j].StartNs {
				return streamLaunches[i].StartNs < streamLaunches[j].StartNs
			}
			return streamLaunches[i].EndNs < streamLaunches[j].EndNs
		})

		n := len(streamLaunches)
		if n == 0 {
			continue
		}

		// 1. Detect observed gaps between consecutive launches
		for i := 0; i < n-1; i++ {
			prev := streamLaunches[i]
			next := streamLaunches[i+1]
			gapNs := next.StartNs - prev.EndNs
			if gapNs >= c.MinGapNs {
				allGaps = append(allGaps, Gap{
					StreamID:        sid,
					StartNs:         prev.EndNs,
					EndNs:           next.StartNs,
					GapNs:           gapNs,
					PrecedingKernel: prev.Name,
					FollowingKernel: next.Name,
					MissingEvidence: []string{
						"host_cpu_activity_unobserved",
						"stream_synchronization_unobserved",
						"data_hazard_unobserved",
					},
				})
			}
		}

		// 2. Detect repeated launch sequences
		names := make([]string, n)
		for i, l := range streamLaunches {
			names[i] = l.Name
		}

		maxPatLen := n / c.MinRepetitions
		for patLen := c.MinSequenceLen; patLen <= maxPatLen; patLen++ {
			for i := 0; i <= n-patLen*c.MinRepetitions; {
				pat := names[i : i+patLen]
				reps := 1
				for j := i + patLen; j+patLen <= n; j += patLen {
					if equalSlice(names[j:j+patLen], pat) {
						reps++
					} else {
						break
					}
				}
				if reps >= c.MinRepetitions {
					startIdx := i
					endIdx := i + reps*patLen - 1
					intervals := make([]Interval, reps)
					for r := 0; r < reps; r++ {
						rStart := streamLaunches[i+r*patLen].StartNs
						rEnd := streamLaunches[i+(r+1)*patLen-1].EndNs
						intervals[r] = Interval{StartNs: rStart, EndNs: rEnd}
					}
					allRepeated = append(allRepeated, RepeatedSequence{
						StreamID:        sid,
						Pattern:         append([]string(nil), pat...),
						Repetitions:     reps,
						TotalDurationNs: streamLaunches[endIdx].EndNs - streamLaunches[startIdx].StartNs,
						Intervals:       intervals,
						MissingEvidence: []string{
							"argument_mutation_unobserved",
							"grid_dimension_equivalence_unverified",
						},
					})
					i += reps * patLen
				} else {
					i++
				}
			}
		}

		// 3. Detect candidate batching regions
		// A candidate batching region is a contiguous run of at least CandidateBatchingMinLaunches
		// where each gap between consecutive launches is <= CandidateBatchingMaxGapNs.
		start := 0
		for i := 0; i < n; i++ {
			if i == n-1 || streamLaunches[i+1].StartNs-streamLaunches[i].EndNs > c.CandidateBatchingMaxGapNs {
				count := i - start + 1
				if count >= c.CandidateBatchingMinLaunches {
					kNames := make([]string, count)
					for idx := 0; idx < count; idx++ {
						kNames[idx] = streamLaunches[start+idx].Name
					}
					allCandidates = append(allCandidates, BatchingCandidate{
						StreamID:    sid,
						StartNs:     streamLaunches[start].StartNs,
						EndNs:       streamLaunches[i].EndNs,
						LaunchCount: count,
						Kernels:     kNames,
						Hypothesis:  "contiguous rapid execution cadence suggests candidate for graph capture or operator batching",
						MissingEvidence: []string{
							"host_side_effects_unverified",
							"graph_capture_safety_unverified",
							"parameter_stability_unverified",
						},
					})
				}
				start = i + 1
			}
		}
	}

	return SequenceAnalysis{
		StreamIDs:          streamIDs,
		Gaps:               allGaps,
		RepeatedSequences:  allRepeated,
		BatchingCandidates: allCandidates,
	}, nil
}

func equalSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
