package timeline

import (
	"errors"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
)

// Correlation describes the retained host provenance of one device activity.
// Indexes refer to Records.Activities, Records.Calls and Records.Ranges.
// APIStatus and RangeStatus are matched, missing, ambiguous or unsupported.
// Call and Range are -1 unless the corresponding status is matched.
// A link identifies submission context; it does not explain kernel duration.
type Correlation struct {
	Metadata       *marker.Metadata `json:"metadata,omitempty"`
	MetadataStatus string           `json:"metadata_status"`

	Activity    int    `json:"activity"`
	Call        int    `json:"call"`
	Range       int    `json:"range"`
	APIStatus   string `json:"api_status"`
	RangeStatus string `json:"range_status"`
	// Provenance is submission, graph_replay, graph_construction, or unresolved.
	Provenance      string     `json:"provenance"`
	Lifecycle       *GraphLink `json:"graph_lifecycle,omitempty"`
	LifecycleStatus string     `json:"graph_lifecycle_status,omitempty"`
}

// Correlations joins device activities to retained calls in the selected API
// domain ("driver" or "runtime"), within a single process-generation journal.
// It returns one result per activity in input order. Repeated correlation IDs
// are ambiguous; zero IDs and absent journal identity never match. A single
// call may submit several activities, so callers must not sum its duration
// once per activity. A graph record matches only an exact retained correlation
// naming a graph launch API. A construction API never identifies replay context;
// graph IDs, node IDs and temporal containment alone never identify a launch.
//
// Range links select the innermost range enclosing the whole API call on its
// process and thread. Crossing ranges and equally narrow candidates remain
// ambiguous. An operation that submits on another thread needs explicit
// instrumentation on that thread; temporal overlap alone never links it.
// Ranges with missing process or thread identity are unsupported.
// Metadata comes only from the selected innermost range, with no inheritance
// from outer ranges. Missing optional fields remain missing. Conflicting
// explicitly supplied fields in enclosing markers remain ambiguous.
func Correlations(records Records, domain string) []Correlation {
	type key struct {
		journal     string
		correlation uint32
	}
	calls := make(map[key][]int)
	for i, c := range records.Calls {
		if c.Journal != "" && c.Correlation != 0 && c.Domain == domain {
			k := key{c.Journal, c.Correlation}
			calls[k] = append(calls[k], i)
		}
	}
	var index *rangeIndex
	var parsed []rangeMetadata
	contexts := make(map[int]rangeContext)
	lifecycle := GraphLifecycle(records.GraphLifecycle)
	out := make([]Correlation, 0, len(records.Activities))
	for i, a := range records.Activities {
		m := Correlation{Activity: i, Call: -1, Range: -1, APIStatus: "missing", RangeStatus: "missing", MetadataStatus: "missing", Provenance: "unresolved"}
		switch {
		case domain != "driver" && domain != "runtime":
			m.APIStatus, m.RangeStatus = "unsupported", "unsupported"
		case a.Subtype == "memcpy" || a.Subtype == "memset":
			// Retained transfer records do not distinguish runtime identity.
			if domain == "runtime" {
				m.APIStatus, m.RangeStatus = "unsupported", "unsupported"
				break
			}
			fallthrough
		default:
			correlation := a.Correlation
			if domain == "runtime" && (a.Subtype == "memcpy" || a.Subtype == "memset") {
				correlation = a.RuntimeCorrelation
			}
			ids := calls[key{a.Journal, correlation}]
			if a.Journal == "" || correlation == 0 || len(ids) == 0 {
				break
			}
			if len(ids) != 1 {
				m.APIStatus, m.RangeStatus = "ambiguous", "unsupported"
				break
			}
			if a.GraphID != 0 || a.Subtype == "graph_trace" {
				name := records.Calls[ids[0]].Name()
				if !strings.HasPrefix(name, "cuGraphLaunch") && !strings.HasPrefix(name, "cudaGraphLaunch") {
					m.APIStatus, m.RangeStatus = "unsupported", "unsupported"
					if strings.Contains(name, "Graph") {
						m.Provenance = "graph_construction"
					}
					break
				}
				m.Provenance = "graph_replay"
			} else {
				m.Provenance = "submission"
			}
			m.Call, m.APIStatus = ids[0], "matched"
			c := records.Calls[m.Call]
			if c.Process == 0 || c.Thread == 0 || c.End < c.Start {
				m.RangeStatus = "unsupported"
				break
			}
			context, ok := contexts[m.Call]
			if !ok {
				if index == nil {
					index = newRangeIndex(records.Ranges)
					parsed = make([]rangeMetadata, len(records.Ranges))
				}
				context = resolveRangeContext(c, records.Ranges, index, parsed)
				contexts[m.Call] = context
			}
			m.Range, m.RangeStatus = context.rangeIndex, context.rangeStatus
		}
		if m.RangeStatus == "matched" {
			m.MetadataStatus = contexts[m.Call].metadataStatus
			if m.MetadataStatus == "matched" {
				// Return independent marker values for each activity. The cached
				// parse is private to this invocation and never escapes to callers.
				meta, _ := marker.ParseRange(records.Ranges[m.Range].Name)
				m.Metadata = &meta
			}
		} else if m.RangeStatus != "missing" {
			m.MetadataStatus = m.RangeStatus
		}
		if a.GraphID != 0 || a.Subtype == "graph_trace" {
			m.LifecycleStatus = "unsupported"
			if len(records.GraphLifecycle) != 0 {
				m.LifecycleStatus = "missing"
				var matches []int
				for j, e := range records.GraphLifecycle {
					if e.Journal == a.Journal && e.Domain == domain && e.Kind == "replay" && e.Correlation != 0 && e.Correlation == a.Correlation {
						matches = append(matches, j)
					}
				}
				if len(matches) > 1 {
					m.LifecycleStatus = "ambiguous"
				} else if len(matches) == 1 {
					link := lifecycle[matches[0]]
					m.LifecycleStatus = link.Status
					m.Lifecycle = &link
				}
			}
		}
		out = append(out, m)
	}
	return out
}

type rangeContext struct {
	rangeIndex     int
	rangeStatus    string
	metadataStatus string
}

type rangeMetadata struct {
	parsed bool
	value  marker.Metadata
	err    error
}

func readRangeMetadata(ranges []Range, cache []rangeMetadata, i int) (marker.Metadata, error) {
	if !cache[i].parsed {
		cache[i].value, cache[i].err = marker.ParseRange(ranges[i].Name)
		cache[i].parsed = true
	}
	return cache[i].value, cache[i].err
}

func resolveRangeContext(call Call, ranges []Range, index *rangeIndex, parsed []rangeMetadata) rangeContext {
	result := rangeContext{rangeIndex: -1, rangeStatus: "missing", metadataStatus: "missing"}
	candidates := index.enclosing(call)
	var start, end int64
	found := false
	for _, i := range candidates {
		r := ranges[i]
		if r.End <= r.Start {
			continue
		}
		if index.crossed(i) {
			result.rangeStatus = "ambiguous"
			return result
		}
		if !found || r.Start > start {
			start = r.Start
		}
		if !found || r.End < end {
			end = r.End
		}
		found = true
	}
	if !found {
		return result
	}
	// A unique innermost range has the latest start and earliest end.
	// Equal intervals stay ambiguous, even when they carry equal metadata.
	for _, i := range candidates {
		r := ranges[i]
		if r.End > r.Start && r.Start == start && r.End == end {
			if result.rangeIndex >= 0 {
				result.rangeIndex, result.rangeStatus = -1, "ambiguous"
				return result
			}
			result.rangeIndex = i
		}
	}
	if result.rangeIndex < 0 {
		result.rangeStatus = "ambiguous"
		return result
	}
	result.rangeStatus = "matched"
	meta, err := readRangeMetadata(ranges, parsed, result.rangeIndex)
	switch {
	case err == nil && meta.Event != "":
		result.metadataStatus = "unsupported"
	case err == nil:
		result.metadataStatus = "matched"
		// The index preserves input order, including zero-duration ranges, so a
		// malformed marker and a conflict retain the original refusal precedence.
		for _, i := range candidates {
			outer, err := readRangeMetadata(ranges, parsed, i)
			if err == nil && conflictingMetadata(meta, outer) {
				result.metadataStatus = "ambiguous"
				break
			}
			if err != nil && !errors.Is(err, marker.ErrNotMarker) {
				result.metadataStatus = "unsupported"
				break
			}
		}
	case errors.Is(err, marker.ErrNotMarker):
	default:
		result.metadataStatus = "unsupported"
	}
	return result
}

// conflictingMetadata compares only fields explicitly supplied by both ranges.
func conflictingMetadata(a, b marker.Metadata) bool {
	if a.Version == 3 || b.Version == 3 {
		if a.Version != b.Version {
			return true
		}
		if a.Component != nil && b.Component != nil && !reflect.DeepEqual(a.Component, b.Component) {
			return true
		}
		if conflictingValue(a.Invocation, b.Invocation) || conflictingValue(a.Batch, b.Batch) || conflictingValue(a.RootCount, b.RootCount) || conflictingValue(a.Root, b.Root) {
			return true
		}
	}
	return (a.ExecutionUnit != "" && b.ExecutionUnit != "" && a.ExecutionUnit != b.ExecutionUnit) || a.Event != b.Event || a.Phase != b.Phase || (a.RequestID != "" && b.RequestID != "" && a.RequestID != b.RequestID) || (a.Role != "" && b.Role != "" && a.Role != b.Role) || (a.Boundary != "" && b.Boundary != "" && a.Boundary != b.Boundary) || conflictingValue(a.Executions, b.Executions) || (a.Operation != "" && b.Operation != "" && a.Operation != b.Operation) ||
		conflictingValue(a.Token, b.Token) || conflictingValue(a.Layer, b.Layer) || conflictingValue(a.DType, b.DType) ||
		conflictingValue(a.CacheLength, b.CacheLength) || conflictingValue(a.Lookahead, b.Lookahead) || conflictingValue(a.Materialization, b.Materialization)
}

func conflictingValue[T comparable](a, b *T) bool { return a != nil && b != nil && *a != *b }
