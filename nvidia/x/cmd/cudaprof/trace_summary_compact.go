package main

import (
	"crypto/sha256"
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
)

// Compact rows retain references to raw queries, not copies of their payloads.
type compactLabel struct {
	Text         string `json:"text"`
	SHA256       string `json:"sha256,omitempty"`
	OmittedBytes int    `json:"omitted_bytes"`
}

func compactText(s string) compactLabel {
	const limit = 160
	if len(s) <= limit {
		return compactLabel{Text: s}
	}
	// Cut at a UTF-8 boundary; the hash identifies the complete original string.
	n := limit
	for n > 0 && s[n]&0xc0 == 0x80 {
		n--
	}
	return compactLabel{Text: s[:n], SHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(s))), OmittedBytes: len(s) - n}
}

type compactStats struct {
	Count  int   `json:"count"`
	Min    int64 `json:"min_ns"`
	Median int64 `json:"median_ns"`
	Max    int64 `json:"max_ns"`
}

func compactSpacing(values []int64) *compactStats {
	if len(values) == 0 {
		return nil
	}
	xs := append([]int64(nil), values...)
	sort.Slice(xs, func(i, j int) bool { return xs[i] < xs[j] })
	return &compactStats{len(xs), xs[0], xs[len(xs)/2], xs[len(xs)-1]}
}

type compactDelta struct {
	Delta     int           `json:"successful_execution_delta"`
	Intervals int           `json:"interval_count"`
	Spacing   *compactStats `json:"consumer_spacing"`
}
type compactSource struct {
	Journal  compactLabel `json:"journal"`
	Kind     string       `json:"kind"`
	Index    int          `json:"index"`
	Sequence uint64       `json:"raw_record_sequence"`
}

func compactSources(ss []traceanalysis.Source) []compactSource {
	out := make([]compactSource, 0, min(4, len(ss)))
	for _, s := range ss[:min(4, len(ss))] {
		out = append(out, compactSource{compactText(s.Journal), s.Kind, s.Index, s.Sequence})
	}
	return out
}

type compactReference struct {
	Journal compactLabel `json:"journal"`
	Kind    string       `json:"kind"`
	Index   int          `json:"index"`
}

func compactReferences(ss []overview.Reference) []compactReference {
	out := make([]compactReference, 0, min(4, len(ss)))
	for _, s := range ss[:min(4, len(ss))] {
		out = append(out, compactReference{compactText(s.Journal), s.Kind, s.Index})
	}
	return out
}

type compactRequest struct {
	ClockDomain        string `json:"clock_domain"`
	ClockQualification string `json:"clock_qualification"`

	Journal            compactLabel    `json:"journal"`
	Process            uint32          `json:"process"`
	ID                 compactLabel    `json:"request_id"`
	Role               string          `json:"role"`
	Endpoints          int             `json:"lifecycle_endpoint_count"`
	Starts             int             `json:"request_start_count"`
	Ends               int             `json:"request_end_count"`
	Delivered          int             `json:"delivered_count"`
	ValidIntervals     *int            `json:"valid_interval_count"`
	Wall               *int64          `json:"request_wall_ns"`
	FirstDelivery      *int64          `json:"first_delivery_latency_ns"`
	AllSpacing         *compactStats   `json:"all_consumer_spacing"`
	Deltas             []compactDelta  `json:"execution_delta_histogram"`
	OmittedDeltas      int             `json:"omitted_delta_groups"`
	TrailingZero       *int            `json:"trailing_zero_delta_intervals"`
	Unavailable        []compactLabel  `json:"unavailable"`
	Diagnostics        []compactLabel  `json:"diagnostics"`
	OmittedUnavailable int             `json:"omitted_unavailable_reasons"`
	OmittedDiagnostics int             `json:"omitted_diagnostics"`
	Sources            []compactSource `json:"sources"`
	OmittedSources     int             `json:"omitted_sources"`
}

func compactLabels(ss []string) []compactLabel {
	out := make([]compactLabel, 0, min(8, len(ss)))
	for _, s := range ss[:min(8, len(ss))] {
		out = append(out, compactText(s))
	}
	return out
}
func compactRequestRow(r traceanalysis.Request, top int) compactRequest {
	q := compactRequest{ClockDomain: "native CUPTI journal marker timestamps", ClockQualification: "same journal and process only; cross-journal clock equivalence is unqualified", Journal: compactText(r.Journal), Process: r.Process, ID: compactText(r.ID), Role: r.Role, Endpoints: len(r.Endpoints), Unavailable: compactLabels(r.Unavailable), Diagnostics: compactLabels(r.Diagnostics)}
	for _, e := range r.Endpoints {
		if e.Metadata.Event == "request_start" {
			q.Starts++
		}
		if e.Metadata.Event == "request_end" {
			q.Ends++
		}
		if e.Metadata.Event == "delivered" {
			q.Delivered++
		}
		if len(q.Sources) < 4 {
			q.Sources = append(q.Sources, compactSource{compactText(e.Source.Journal), e.Source.Kind, e.Source.Index, e.Source.Sequence})
		}
	}
	q.OmittedSources = len(r.Endpoints) - len(q.Sources)
	q.OmittedUnavailable = max(0, len(r.Unavailable)-8)
	q.OmittedDiagnostics = max(0, len(r.Diagnostics)-8)
	// Never reconstruct metrics cleared by the canonical request validator.
	if len(r.Unavailable) != 0 || len(r.Spacing) == 0 || len(r.Spacing) != len(r.ExecutionDeltas) {
		return q
	}
	n := len(r.Spacing)
	q.ValidIntervals = &n
	q.AllSpacing = compactSpacing(r.Spacing)
	byDelta := map[int][]int64{}
	for i, d := range r.ExecutionDeltas {
		byDelta[d] = append(byDelta[d], r.Spacing[i])
	}
	keys := make([]int, 0, len(byDelta))
	for d := range byDelta {
		keys = append(keys, d)
	}
	sort.Ints(keys)
	q.OmittedDeltas = max(0, len(keys)-top)
	for _, d := range keys[:min(top, len(keys))] {
		q.Deltas = append(q.Deltas, compactDelta{d, len(byDelta[d]), compactSpacing(byDelta[d])})
	}
	z := 0
	for i := len(r.ExecutionDeltas) - 1; i >= 0 && r.ExecutionDeltas[i] == 0; i-- {
		z++
	}
	q.TrailingZero = &z
	if len(r.Endpoints) >= 2 {
		first, last := r.Endpoints[0], r.Endpoints[len(r.Endpoints)-1]
		if first.Metadata.Event == "request_start" && last.Metadata.Event == "request_end" && last.Time >= first.Time {
			wall := last.Time - first.Time
			q.Wall = &wall
			for _, e := range r.Endpoints {
				if e.Metadata.Event == "delivered" && e.Time >= first.Time && e.Time <= last.Time {
					v := e.Time - first.Time
					q.FirstDelivery = &v
					break
				}
			}
		}
	}
	return q
}

type compactOperation struct {
	ID                compactLabel       `json:"id"`
	Label             compactLabel       `json:"label"`
	Provenance        string             `json:"provenance"`
	KernelCount       int                `json:"kernel_count"`
	KernelSum         int64              `json:"cumulative_kernel_ns"`
	References        []compactReference `json:"references"`
	OmittedReferences int                `json:"omitted_references"`
}
type compactShape struct {
	Journal        compactLabel    `json:"journal"`
	Device         uint32          `json:"device"`
	Symbol         compactLabel    `json:"raw_symbol"`
	Grid           [3]int          `json:"grid"`
	Block          [3]int          `json:"block"`
	Count          int             `json:"count"`
	Min            int64           `json:"min_ns"`
	Median         int64           `json:"median_ns"`
	Max            int64           `json:"max_ns"`
	Sum            int64           `json:"cumulative_kernel_ns"`
	Sources        []compactSource `json:"sources"`
	OmittedSources int             `json:"omitted_sources"`
}
type compactWindow struct {
	Journal        compactLabel    `json:"journal"`
	Device         uint32          `json:"device"`
	UnionNS        uint64          `json:"recorded_union_ns"`
	BusyIntervals  int             `json:"recorded_busy_interval_count"`
	GapIntervals   int             `json:"recorded_gap_interval_count"`
	Sources        []compactSource `json:"sources"`
	OmittedSources int             `json:"omitted_sources"`
}
type compactSummary struct {
	NumericEvidence string `json:"numeric_evidence"`

	Windows        []compactWindow `json:"device_windows"`
	OmittedWindows int             `json:"omitted_device_windows"`

	Counts            map[string]int     `json:"record_counts"`
	MarkerVersions    map[int]int        `json:"observed_endpoint_versions"`
	Requests          []compactRequest   `json:"requests"`
	OmittedRequests   int                `json:"omitted_requests"`
	Operations        []compactOperation `json:"operations"`
	OmittedOperations int                `json:"omitted_operations"`
	Shapes            []compactShape     `json:"kernel_shapes"`
	OmittedShapes     int                `json:"omitted_kernel_shapes"`
	RawQueries        map[string]string  `json:"raw_queries"`
	Limits            []string           `json:"limits"`
}

func compactTraceSummary(s traceanalysis.Summary, shapes traceanalysis.KernelShapes, top int) compactSummary {
	out := compactSummary{Counts: s.Counts, MarkerVersions: map[int]int{}, RawQueries: map[string]string{"requests": "trace query -json delivery-cadence INPUT", "kernel_shapes": "trace query -json kernel-shapes INPUT", "operations": "trace summary -json -view operations INPUT", "all_evidence": "trace summary -json -view detailed INPUT", "device_windows": "trace summary -json -view gaps INPUT"}, Limits: []string{"Each request is scoped to one journal, process, request ID and role; cross-journal clocks are not aligned.", "Spacing measures consumer delivery cadence. Execution counters are application-reported logical forwards, not GPU launches; zero delta does not imply zero GPU work.", "Kernel shapes preserve the native kernel-shapes query grouping key exactly: journal, device, raw symbol, grid and block; stream/register/shared-memory dimensions are not part of that query key.", "Median is the upper middle observed interval. Kernel durations are cumulative activity, not critical-path wall time.", "Completion and isolation gates are unchanged; external application drain evidence cannot promote unknown native completion.", "Native v1 endpoints cannot establish request identity. Omitted rows and references remain available through raw queries."}}
	out.NumericEvidence = "observed: empty collections mean no retained observations"
	if s.Gates["collector"] != "passed" {
		out.NumericEvidence = "unavailable: " + s.Gates["collector"]
	}
	for _, w := range s.Windows[:min(top, len(s.Windows))] {
		out.Windows = append(out.Windows, compactWindow{compactText(w.Journal), w.Device, w.UnionNS, len(w.Busy), len(w.Gaps), compactSources(w.Sources), max(0, len(w.Sources)-4)})
	}
	out.OmittedWindows = max(0, len(s.Windows)-top)
	for _, e := range s.Endpoints {
		out.MarkerVersions[e.Metadata.Version]++
	}
	for _, r := range s.Requests[:min(top, len(s.Requests))] {
		out.Requests = append(out.Requests, compactRequestRow(r, top))
	}
	out.OmittedRequests = max(0, len(s.Requests)-top)
	var nodes []*overview.Node
	var visit func(*overview.Node)
	visit = func(n *overview.Node) {
		if n == nil {
			return
		}
		if n.Kind == "operation" {
			nodes = append(nodes, n)
		}
		for _, c := range n.Children {
			visit(c)
		}
	}
	visit(s.Overview.Hierarchy)
	sort.SliceStable(nodes, func(i, j int) bool {
		if nodes[i].Metrics.KernelSumNanos != nodes[j].Metrics.KernelSumNanos {
			return nodes[i].Metrics.KernelSumNanos > nodes[j].Metrics.KernelSumNanos
		}
		return nodes[i].ID < nodes[j].ID
	})
	for _, n := range nodes[:min(top, len(nodes))] {
		out.Operations = append(out.Operations, compactOperation{compactText(n.ID), compactText(n.Label), n.Provenance, n.Metrics.KernelCount, n.Metrics.KernelSumNanos, compactReferences(n.References), max(0, len(n.References)-4)})
	}
	out.OmittedOperations = max(0, len(nodes)-top)
	ds := append([]traceanalysis.LaunchDistribution(nil), shapes.Distributions...)
	sort.SliceStable(ds, func(i, j int) bool {
		if ds[i].Sum != ds[j].Sum {
			return ds[i].Sum > ds[j].Sum
		}
		a, b := ds[i], ds[j]
		if a.Journal != b.Journal {
			return a.Journal < b.Journal
		}
		if a.Device != b.Device {
			return a.Device < b.Device
		}
		if a.Symbol != b.Symbol {
			return a.Symbol < b.Symbol
		}
		for k := range a.Grid {
			if a.Grid[k] != b.Grid[k] {
				return a.Grid[k] < b.Grid[k]
			}
		}
		for k := range a.Block {
			if a.Block[k] != b.Block[k] {
				return a.Block[k] < b.Block[k]
			}
		}
		return false
	})
	for _, d := range ds[:min(top, len(ds))] {
		out.Shapes = append(out.Shapes, compactShape{compactText(d.Journal), d.Device, compactText(d.Symbol), d.Grid, d.Block, d.Count, d.Min, d.Median, d.Max, d.Sum, compactSources(d.Sources), max(0, len(d.Sources)-4)})
	}
	out.OmittedShapes = max(0, len(ds)-top)
	return out
}
