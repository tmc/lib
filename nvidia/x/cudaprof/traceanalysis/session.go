package traceanalysis

import (
	"fmt"
	"math"
	"regexp"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/dependency"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Options supplies observed gates separately from analysis. CollectorFailure
// prevents numerical views from qualifying partial records. Empty Domain means
// driver. CollectorComplete requires caller-verified collector evidence; zero
// options leave collector qualification unavailable. IterationMarker labels
// inferred windows only.
type Options struct {
	Domain            string
	CollectorComplete bool
	CollectorFailure  string
	IterationMarker   *regexp.Regexp
}

// Session owns immutable query results. New reads records during construction;
// callers must not mutate returned results. The zero value is not usable.
type Session struct {
	report  Summary
	queries map[string]any
}

// Query describes a versioned named-query contract.
type Query struct {
	Name        string   `json:"name"`
	Version     int      `json:"version"`
	Schema      string   `json:"schema"`
	Backends    []string `json:"backends"`
	Units       string   `json:"units"`
	Required    string   `json:"required_evidence"`
	Aggregation string   `json:"aggregation"`
}

// Catalog returns the supported query schemas and evidence requirements.
func Catalog() []Query {
	return []Query{
		{"inventory", 1, "cudaprof.inventory.v1", []string{"native", "trace_processor_v58.2"}, "records", "decoded raw journals", "count source kinds once; no duration aggregation"},
		{"qualifications", 1, "cudaprof.qualifications.v1", []string{"native", "trace_processor_v58.2"}, "state", "collector receipt; other gates independently supplied", "independent states; missing evidence is unavailable"},
		{"delivery-cadence", 1, "cudaprof.delivery-cadence.v1", []string{"native", "trace_processor_v58.2"}, "ns", "instantaneous delivered/EOS/token_limit metadata; explicit request scope for spacing", "raw endpoints retained; request spacing unavailable without request identity"},
		{"kernel-shapes", 1, "cudaprof.kernel-shapes.v1", []string{"native", "trace_processor_v58.2"}, "ns", "raw kernel symbol and native launch geometry", "native raw launches plus per-symbol/geometry distributions; SQL raw groups; neither is independent engine-run uncertainty"},
		{"recorded-gaps", 1, "cudaprof.recorded-gaps.v1", []string{"native", "trace_processor_v58.2"}, "ns", "valid recorded device activities", "per journal/device interval union and complement within recorded span; not hardware idle"},
		{"unmatched", 1, "cudaprof.unmatched.v1", []string{"native", "trace_processor_v58.2"}, "records", "domain-specific submission/context records; SQL needs explicit versioned receipts, positive raw sequences and coverage", "each unresolved activity once, retain matching reasons"},
		{"correlations", 1, "cudaprof.correlations.v1", []string{"native", "trace_processor_v58.2"}, "records", "retained submission calls in selected domain; SQL needs explicit versioned native receipts, positive raw sequences and coverage", "one result per device activity; provenance not causality"},
		{"language-stacks", 1, "cudaprof.language-stacks.v1", []string{"native", "trace_processor_v58.2"}, "records", "retained language stack journals and exact external correlation identity", "scope-entry and submission provenance; never sampled CPU execution"},
		{"host-stacks", 1, "cudaprof.host-stacks.v1", []string{"native", "trace_processor_v58.2"}, "samples", "retained timestamped stacks and qualified same-process native clock", "Python wall snapshots and Go CPU samples are distinct; no submission causality"},
	}
}

// Source identifies an indexed decoded record and its raw journal sequence.
type Source struct {
	Journal  string `json:"journal"`
	Kind     string `json:"kind"`
	Index    int    `json:"index"`
	Sequence uint64 `json:"raw_record_sequence"`
}

// Endpoint is an explicitly retained delivery or termination observation.
type Endpoint struct {
	Source   Source          `json:"source"`
	Process  uint32          `json:"process"`
	Thread   uint32          `json:"thread"`
	Time     int64           `json:"time_ns"`
	Metadata marker.Metadata `json:"metadata"`
}

// DeviceWindows excludes inferred occupancy and cross-journal clock merging.
type DeviceWindows struct {
	Journal string              `json:"journal"`
	Device  uint32              `json:"device"`
	UnionNS uint64              `json:"recorded_union_ns"`
	Busy    []timeline.Interval `json:"recorded_busy"`
	Gaps    []timeline.Interval `json:"recorded_gaps"`
	Sources []Source            `json:"sources"`
}

// Launch retains raw geometry and duration without application equivalence.
type Launch struct {
	Source   Source `json:"source"`
	Symbol   string `json:"raw_symbol"`
	Grid     [3]int `json:"grid"`
	Block    [3]int `json:"block"`
	Duration int64  `json:"duration_ns"`
	Device   uint32 `json:"device"`
	Stream   uint32 `json:"stream"`
	Graph    uint32 `json:"graph_id"`
}

// Summary separates native observations, inference, and evidence gates.
type Summary struct {
	Version         int               `json:"version"`
	Domain          string            `json:"domain"`
	Gates           map[string]string `json:"qualifications"`
	Counts          map[string]int    `json:"source_counts"`
	Stationarity    *iteration.Report `json:"stationarity_receipt,omitempty"`
	Overview        overview.Report   `json:"overview"`
	Windows         []DeviceWindows   `json:"recorded_device_windows"`
	Requests        []Request         `json:"requests"`
	Endpoints       []Endpoint        `json:"delivery_endpoints"`
	Dependencies    dependency.Report `json:"dependencies"`
	DependencyError string            `json:"dependency_error,omitempty"`
	Sources         []Source          `json:"sources"`
	Limits          []string          `json:"limits"`
}

// New analyzes bounded decoded records once. A failed collector gate preserves
// inventory but suppresses numerical summaries, context and dependency results.
func New(records timeline.Records, opts Options) (*Session, error) {
	if opts.Domain == "" {
		opts.Domain = "driver"
	}
	if opts.Domain != "driver" && opts.Domain != "runtime" {
		return nil, fmt.Errorf("trace analysis: invalid domain %q", opts.Domain)
	}
	counts := map[string]int{"activity": len(records.Activities), "call": len(records.Calls), "range": len(records.Ranges), "marker": len(records.Markers), "unified_memory": len(records.UnifiedMemory), "memory_policy": len(records.MemoryPolicies), "dependency": len(records.Dependencies), "graph": len(records.Graphs), "graph_lifecycle": len(records.GraphLifecycle), "gemm": len(records.GEMMs), "sync": len(records.Syncs), "overhead": len(records.Overheads)}
	total := 0
	for _, n := range counts {
		if n > timeline.MaxRecords-total {
			return nil, fmt.Errorf("trace analysis: record limit exceeded")
		}
		total += n
	}
	gates := map[string]string{"collector": "passed", "import": "unchecked: no external importer used", "workload_coverage": "unavailable", "stationarity": "unassessed", "shape": "unqualified", "correctness": "unverified", "request_scope": "unavailable: marker schema has no request identity", "dependency": "partial"}
	r := Summary{Version: 1, Domain: opts.Domain, Gates: gates, Counts: counts, Windows: []DeviceWindows{}, Endpoints: []Endpoint{}, Sources: []Source{}, Limits: []string{"observed delivery, inferred boundaries, materialization host ranges, GPU unions and kernel sums remain distinct", "legacy v1 markers do not identify requests; v2 request identities do not prove workload completeness", "recorded gaps are complements within each journal/device recorded span, not hardware idle or causes", "launch durations are not independent engine-run uncertainty", "comparison requires two bound arm identities and qualified shape/campaign receipts; no single-capture numerical operation delta"}}
	r.Dependencies = dependency.Report{State: "partial", Reasons: []string{"native coverage and delivery endpoints are not independently qualified"}}
	s := &Session{report: r, queries: map[string]any{}}
	for i, a := range records.Activities {
		s.report.Sources = append(s.report.Sources, Source{a.Journal, "activity", i, a.Sequence})
	}
	for i, a := range records.Calls {
		s.report.Sources = append(s.report.Sources, Source{a.Journal, "call", i, a.Sequence})
	}
	for i, a := range records.Markers {
		s.report.Sources = append(s.report.Sources, Source{a.Journal, "marker", i, a.Sequence})
	}
	for i, a := range records.UnifiedMemory {
		s.report.Sources = append(s.report.Sources, Source{a.Journal, "unified_memory", i, a.Sequence})
	}
	for i, a := range records.MemoryPolicies {
		s.report.Sources = append(s.report.Sources, Source{a.Journal, "memory_policy", i, a.Sequence})
	}
	s.queries["inventory"] = struct {
		Counts             map[string]int `json:"source_counts"`
		Sources            []Source       `json:"sources"`
		ActivityCapability string         `json:"activity_capability"`
	}{counts, s.report.Sources, "retained kind inventory; collection support requires collector capability receipt"}
	if opts.CollectorFailure != "" || !opts.CollectorComplete {
		gates["collector"] = "unavailable: caller did not supply collector verification"
		if opts.CollectorFailure != "" {
			gates["collector"] = "failed: " + opts.CollectorFailure
		}
		s.queries["qualifications"] = gates
		return s, nil
	}
	var totalDuration int64
	for _, a := range records.Activities {
		if a.Start < 0 || a.End < a.Start {
			continue
		}
		duration := a.End - a.Start
		if duration > math.MaxInt64-totalDuration {
			return nil, fmt.Errorf("trace analysis: activity duration sum overflows")
		}
		totalDuration += duration
	}
	for _, a := range records.Ranges {
		if a.Start < 0 || a.End < a.Start {
			continue
		}
		duration := a.End - a.Start
		if duration > math.MaxInt64-totalDuration {
			return nil, fmt.Errorf("trace analysis: scoped duration sum overflows")
		}
		totalDuration += duration
	}
	s.report.Overview = overview.Analyze(records, opts.Domain, opts.IterationMarker)
	if opts.IterationMarker != nil {
		journals := map[string]bool{}
		devices := map[uint32]bool{}
		for _, a := range records.Activities {
			journals[a.Journal] = true
			devices[a.Device] = true
		}
		if len(journals) != 1 || len(devices) != 1 {
			gates["stationarity"] = "unavailable: single journal/device iteration scope required"
		} else {
			report, err := iteration.Analyze(records, iteration.Options{Marker: opts.IterationMarker})
			if err != nil {
				gates["stationarity"] = "unavailable: " + err.Error()
			} else {
				s.report.Stationarity = report
				gates["stationarity"] = "passed"
				if !report.Gate.Pass {
					gates["stationarity"] = "failed: " + fmt.Sprint(report.Gate.Failures)
				}
			}
		}
	}
	links := timeline.Correlations(records, opts.Domain)
	s.queries["correlations"] = links
	unresolved := []timeline.Correlation{}
	for _, link := range links {
		if link.APIStatus != "matched" || link.RangeStatus != "matched" || link.MetadataStatus != "matched" {
			unresolved = append(unresolved, link)
		}
	}
	s.queries["unmatched"] = unresolved
	launches := []Launch{}
	type key struct {
		journal string
		device  uint32
	}
	windows := map[key]*DeviceWindows{}
	for i, a := range records.Activities {
		source := Source{a.Journal, "activity", i, a.Sequence}
		if a.Start < 0 || a.End < a.Start || a.Journal == "" {
			continue
		}
		if a.Subtype == "kernel" {
			launches = append(launches, Launch{source, a.Name, a.Grid, a.Block, a.End - a.Start, a.Device, a.Stream, a.GraphID})
		}
		if a.Subtype != "kernel" && a.Subtype != "memcpy" && a.Subtype != "memset" {
			continue
		}
		k := key{a.Journal, a.Device}
		if windows[k] == nil {
			windows[k] = &DeviceWindows{Journal: a.Journal, Device: a.Device, Busy: []timeline.Interval{}, Gaps: []timeline.Interval{}, Sources: []Source{}}
		}
		w := windows[k]
		w.Busy = append(w.Busy, timeline.Interval{Start: uint64(a.Start), End: uint64(a.End)})
		w.Sources = append(w.Sources, source)
	}
	for _, w := range windows {
		sort.Slice(w.Busy, func(i, j int) bool { return w.Busy[i].Start < w.Busy[j].Start })
		merged := []timeline.Interval{}
		for _, v := range w.Busy {
			if len(merged) == 0 || v.Start > merged[len(merged)-1].End {
				merged = append(merged, v)
			} else if v.End > merged[len(merged)-1].End {
				merged[len(merged)-1].End = v.End
			}
		}
		w.Busy = merged
		w.UnionNS = timeline.Union(merged)
		for i := 1; i < len(merged); i++ {
			w.Gaps = append(w.Gaps, timeline.Interval{Start: merged[i-1].End, End: merged[i].Start})
		}
		s.report.Windows = append(s.report.Windows, *w)
	}
	sort.Slice(s.report.Windows, func(i, j int) bool {
		a, b := s.report.Windows[i], s.report.Windows[j]
		if a.Journal != b.Journal {
			return a.Journal < b.Journal
		}
		return a.Device < b.Device
	})
	for i, m := range records.Markers {
		if m.Flags&timeline.MarkerInstantaneous == 0 || m.Flags&(timeline.MarkerStart|timeline.MarkerEnd) != 0 || m.Time < 0 {
			continue
		}
		metadata, err := marker.Parse(m.Name)
		if err != nil || metadata.Event == "" {
			continue
		}
		s.report.Endpoints = append(s.report.Endpoints, Endpoint{Source{m.Journal, "marker", i, m.Sequence}, m.Process, m.Thread, m.Time, metadata})
	}
	s.report.Requests = requests(s.report.Endpoints)
	if len(s.report.Requests) > 0 {
		gates["request_scope"] = "observed: process/journal/request/role; lifecycle and execution qualifications retained per request"
	}
	graph := dependency.Observe(records)
	var dependencyError error
	s.report.Dependencies, dependencyError = dependency.Analyze(graph)
	if err := dependencyError; err != nil {
		s.report.DependencyError = err.Error()
	}
	s.queries["kernel-shapes"] = kernelShapes(launches)
	s.queries["recorded-gaps"] = s.report.Windows
	s.queries["qualifications"] = gates
	s.queries["delivery-cadence"] = struct {
		Requests      []Request  `json:"requests"`
		Endpoints     []Endpoint `json:"endpoints"`
		Spacing       *int64     `json:"spacing_ns"`
		State         string     `json:"state"`
		FirstEndpoint string     `json:"first_endpoint"`
	}{s.report.Requests, s.report.Endpoints, nil, "legacy spacing unavailable; explicit v2 request qualifications retained separately", "request start endpoints are qualified per v2 request"}
	return s, nil
}

// Summary returns the native summary, with all independent missing states.
func (s *Session) Summary() Summary { return s.report }

// Query returns a catalogued native view. Missing qualification is returned as
// an explicit unavailable payload, not as a numerical zero.
func (s *Session) Query(name string) (any, error) {
	for _, q := range Catalog() {
		if q.Name == name {
			if value, ok := s.queries[name]; ok {
				return value, nil
			}
			return struct {
				State  string `json:"state"`
				Reason string `json:"reason"`
			}{"unavailable", s.report.Gates["collector"]}, nil
		}
	}
	return nil, fmt.Errorf("trace query: unknown name %q", name)
}
