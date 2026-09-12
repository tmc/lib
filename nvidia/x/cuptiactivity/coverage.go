package cuptiactivity

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/cupti"
)

// ActivityTotals accounts for activity records and valid native intervals.
// Duration is the sum of intervals in nanoseconds, not elapsed wall time:
// activities may overlap. InvalidIntervals are retained but excluded from
// Duration. These values do not account for unobserved or dropped records.
type ActivityTotals struct {
	Records          uint64
	TimedRecords     uint64
	InvalidIntervals uint64
	Duration         uint64
}

// AttributionCoverage separates native link coverage by collection,
// process generation, device and activity kind. Caller-provided rank identity
// must be resolved separately; a process or device number does not imply rank.
// Matched means a native link, not a Python frame or operator match.
type AttributionCoverage struct {
	Session, Run, Process                  string
	DeviceID                               uint32
	Kind                                   cupti.ActivityKind
	Total, Matched, NotObserved, Ambiguous ActivityTotals
}

// ExternalCoverage reports native external-correlation coverage for concurrent
// kernels, memcpy and memset. It uses the exact matching rules of MatchExternal.
// API, memory and external-link records do not add device execution duration.
// Results are sorted by session, run, process, device and kind. Overflow returns
// an error without partial results. Inputs are not modified.
//
// Report both record and duration coverage; a high matched record count does
// not establish high duration coverage. Missing intervals and collector losses
// remain separate from attribution. Durations from different rows must not be
// subtracted to infer clock alignment, idle time or rank skew.
func ExternalCoverage(records []ScopedRecord, kind cupti.ExternalCorrelationKind) ([]AttributionCoverage, error) {
	matches, err := MatchExternal(records, kind)
	if err != nil {
		return nil, err
	}
	return attributionCoverage(records, matches)
}

// APICoverage reports device activity coverage by exact native API invocation
// links. kind selects the driver or runtime domain, using MatchAPI's identity
// and ambiguity rules. Results have the same grouping and interval accounting
// as ExternalCoverage. Only device intervals contribute duration: a graph's
// shared API invocation is not counted once per linked kernel. API linkage does
// not imply Python operator attribution, successful execution, or complete
// collection. Missing API records remain not_observed even with valid device
// intervals.
func APICoverage(records []ScopedRecord, kind cupti.ActivityKind) ([]AttributionCoverage, error) {
	apis, err := MatchAPI(records, kind)
	if err != nil {
		return nil, err
	}
	matches := make([]ExternalMatch, len(apis))
	for i, m := range apis {
		matches[i] = ExternalMatch{Index: m.Index, Status: m.Status}
	}
	return attributionCoverage(records, matches)
}

func attributionCoverage(records []ScopedRecord, matches []ExternalMatch) ([]AttributionCoverage, error) {
	type key struct {
		session, run, process string
		device                uint32
		kind                  cupti.ActivityKind
	}
	groups := make(map[key]*AttributionCoverage)
	for _, m := range matches {
		s := records[m.Index]
		r := s.Record
		if r.Kind != cupti.ActivityKindConcurrentKernel && r.Kind != cupti.ActivityKindMemcpy && r.Kind != cupti.ActivityKindMemset {
			continue
		}
		k := key{s.Session, s.Run, s.Process, r.DeviceID, r.Kind}
		g := groups[k]
		if g == nil {
			g = &AttributionCoverage{Session: s.Session, Run: s.Run, Process: s.Process, DeviceID: r.DeviceID, Kind: r.Kind}
			groups[k] = g
		}
		dst := &g.NotObserved
		switch m.Status {
		case "matched":
			dst = &g.Matched
		case "ambiguous":
			dst = &g.Ambiguous
		}
		for _, total := range []*ActivityTotals{&g.Total, dst} {
			total.Records++
			if !r.Complete || r.Start == 0 || r.End < r.Start {
				total.InvalidIntervals++
				continue
			}
			d := r.End - r.Start
			if ^uint64(0)-total.Duration < d {
				return nil, fmt.Errorf("cuptiactivity: attribution duration overflows at record %d", m.Index)
			}
			total.Duration += d
			total.TimedRecords++
		}
	}
	var out []AttributionCoverage
	for _, g := range groups {
		out = append(out, *g)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Session != b.Session {
			return a.Session < b.Session
		}
		if a.Run != b.Run {
			return a.Run < b.Run
		}
		if a.Process != b.Process {
			return a.Process < b.Process
		}
		if a.DeviceID != b.DeviceID {
			return a.DeviceID < b.DeviceID
		}
		return a.Kind < b.Kind
	})
	return out, nil
}
