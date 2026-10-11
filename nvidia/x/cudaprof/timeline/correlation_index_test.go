package timeline

import (
	"errors"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
)

// referenceCorrelations retains the pre-index implementation from fb9e31beb.
// It intentionally scans all ranges and is limited to small oracle inputs.
func referenceCorrelations(records Records, domain string) []Correlation {
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
			var candidates []int
			for j, r := range records.Ranges {
				if r.Journal == c.Journal && r.Process == c.Process && r.Thread == c.Thread && r.Start <= c.Start && r.End >= c.End && r.End > r.Start {
					candidates = append(candidates, j)
				}
			}
			if len(candidates) == 0 {
				break
			}
			// An invalid ancestor makes its nested context ambiguous too.
			// Crossing ranges need not enclose the entire API call.
			crossed := false
			for _, j := range candidates {
				r := records.Ranges[j]
				for _, s := range records.Ranges {
					if s.Journal != r.Journal || s.Process != r.Process || s.Thread != r.Thread || s.End <= s.Start {
						continue
					}
					if r.Start < s.Start && s.Start < r.End && r.End < s.End || s.Start < r.Start && r.Start < s.End && s.End < r.End {
						crossed = true
						break
					}
				}
				if crossed {
					break
				}
			}
			if crossed {
				m.RangeStatus = "ambiguous"
				break
			}
			// A unique minimum must be contained by every other candidate.
			for _, j := range candidates {
				r := records.Ranges[j]
				inner := true
				for _, k := range candidates {
					if j == k {
						continue
					}
					s := records.Ranges[k]
					if r.Start < s.Start || r.End > s.End || (r.Start == s.Start && r.End == s.End) {
						inner = false
						break
					}
				}
				if inner {
					m.Range, m.RangeStatus = j, "matched"
					break
				}
			}
			if m.Range < 0 {
				m.RangeStatus = "ambiguous"
			}
		}
		if m.RangeStatus == "matched" {
			meta, err := marker.ParseRange(records.Ranges[m.Range].Name)
			switch {
			case err == nil && meta.Event != "":
				m.MetadataStatus = "unsupported"
			case err == nil:
				m.Metadata, m.MetadataStatus = &meta, "matched"
				c := records.Calls[m.Call]
				for _, r := range records.Ranges {
					if r.Journal != c.Journal || r.Process != c.Process || r.Thread != c.Thread || r.Start > c.Start || r.End < c.End {
						continue
					}
					outer, err := marker.ParseRange(r.Name)
					if err == nil && conflictingMetadata(meta, outer) {
						m.Metadata, m.MetadataStatus = nil, "ambiguous"
						break
					}
					if err != nil && !errors.Is(err, marker.ErrNotMarker) {
						m.Metadata, m.MetadataStatus = nil, "unsupported"
						break
					}
				}
			case errors.Is(err, marker.ErrNotMarker):
			default:
				m.MetadataStatus = "unsupported"
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

func checkCorrelationOracle(t *testing.T, records Records, domain string) {
	t.Helper()
	got, want := Correlations(records, domain), referenceCorrelations(records, domain)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("domain %q\ngot %+v\nwant %+v", domain, got, want)
	}
}

func TestCorrelationIndexDifferential(t *testing.T) {
	random := rand.New(rand.NewSource(42))
	names := []string{"plain", `cudaprof:v1 {"phase":"decode","token":1}`, `cudaprof:v1 {"phase":"decode","token":2}`, `cudaprof:v1 {"phase":"decode","unknown":1}`, `cudaprof:v1 {"phase":"decode","event":"done"}`}
	for trial := 0; trial < 80; trial++ {
		var records Records
		for i := 0; i < 30; i++ {
			start := int64(random.Intn(45) - 5)
			records.Ranges = append(records.Ranges, Range{Journal: []string{"a", "b", ""}[random.Intn(3)], Process: uint32(random.Intn(3)), Thread: uint32(random.Intn(3)), Start: start, End: start + int64(random.Intn(25)-2), Name: names[random.Intn(len(names))]})
		}
		for i := 0; i < 16; i++ {
			start := int64(random.Intn(50) - 5)
			records.Calls = append(records.Calls, Call{Journal: []string{"a", "b", ""}[random.Intn(3)], Process: uint32(random.Intn(3)), Thread: uint32(random.Intn(3)), Domain: []string{"driver", "runtime"}[random.Intn(2)], CallbackID: []uint32{1, 502, 514, 311}[random.Intn(4)], Correlation: uint32(random.Intn(9)), Start: start, End: start + int64(random.Intn(8)-1)})
		}
		for i := 0; i < 36; i++ {
			records.Activities = append(records.Activities, Activity{Journal: []string{"a", "b", ""}[random.Intn(3)], Correlation: uint32(random.Intn(9)), RuntimeCorrelation: uint32(random.Intn(9)), GraphID: uint32(random.Intn(2)), Subtype: []string{"kernel", "memcpy", "memset", "graph_trace"}[random.Intn(4)]})
		}
		for _, domain := range []string{"driver", "runtime", "unknown"} {
			checkCorrelationOracle(t, records, domain)
		}
	}
}

func TestCorrelationIndexBoundaries(t *testing.T) {
	good := `cudaprof:v1 {"phase":"decode","token":1}`
	bad := `cudaprof:v1 {"phase":"decode","invalid":1}`
	conflict := `cudaprof:v1 {"phase":"decode","token":2}`
	for _, tt := range []struct {
		name       string
		ranges     []Range
		start, end int64
	}{
		{"crossing ancestor", []Range{{Start: 10, End: 40}, {Start: 20, End: 30}, {Start: 30, End: 60}}, 21, 29},
		{"equal bounds", []Range{{Start: 10, End: 30}, {Start: 10, End: 30}}, 20, 25},
		{"touching", []Range{{Start: 0, End: 20}, {Start: 20, End: 40}}, 20, 20},
		{"zero marker malformed first", []Range{{Start: 20, End: 20, Name: bad}, {Start: 10, End: 30, Name: good}}, 20, 20},
		{"zero marker conflict first", []Range{{Start: 20, End: 20, Name: conflict}, {Start: 20, End: 20, Name: bad}, {Start: 10, End: 30, Name: good}}, 20, 20},
		{"negative times", []Range{{Start: -30, End: -5}, {Start: -20, End: -10}}, -18, -12},
		{"maximum endpoints", []Range{{Start: math.MinInt64, End: math.MaxInt64}, {Start: 0, End: math.MaxInt64}}, math.MaxInt64, math.MaxInt64},
	} {
		t.Run(tt.name, func(t *testing.T) {
			records := Records{Ranges: tt.ranges, Calls: []Call{{Journal: "a", Process: 1, Thread: 2, Domain: "driver", Correlation: 1, Start: tt.start, End: tt.end}}, Activities: []Activity{{Journal: "a", Correlation: 1}, {Journal: "a", Correlation: 1}}}
			for i := range records.Ranges {
				records.Ranges[i].Journal = "a"
				records.Ranges[i].Process = 1
				records.Ranges[i].Thread = 2
				if records.Ranges[i].Name == "" {
					records.Ranges[i].Name = good
				}
			}
			checkCorrelationOracle(t, records, "driver")
			// Identical timestamps in another generation/process/thread must not cross.
			foreign := Range{Journal: "b", Process: 1, Thread: 2, Start: tt.start, End: math.MaxInt64, Name: bad}
			records.Ranges = append(records.Ranges, foreign)
			checkCorrelationOracle(t, records, "driver")
		})
	}
}

func TestCorrelationMetadataDoesNotAlias(t *testing.T) {
	records := Records{Ranges: []Range{{Journal: "a", Process: 1, Thread: 2, Start: 0, End: 20, Name: `cudaprof:v1 {"phase":"decode","token":1}`}}, Calls: []Call{{Journal: "a", Process: 1, Thread: 2, Domain: "driver", Correlation: 1, Start: 2, End: 3}}, Activities: []Activity{{Journal: "a", Correlation: 1}, {Journal: "a", Correlation: 1}}}
	got := Correlations(records, "driver")
	if got[0].Metadata == nil || got[1].Metadata == nil {
		t.Fatalf("missing metadata %+v", got)
	}
	*got[0].Metadata.Token = 9
	if *got[1].Metadata.Token != 1 {
		t.Fatal("cached metadata escaped between activities")
	}
}
