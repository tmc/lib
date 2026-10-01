package shape

import (
	"encoding/json"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Identity is an explicitly observed operation and launch configuration.
// Layer remains optional, but an absent layer does not match a supplied layer.
// SharedMemory remains optional with the same rule. Raw symbols are retained
// on Launch and are never normalized to discard template parameters.
type Identity struct {
	RequestPair     *int    `json:"request_pair,omitempty"`
	Role            string  `json:"role,omitempty"`
	Boundary        string  `json:"boundary,omitempty"`
	Phase           string  `json:"phase"`
	Token           int     `json:"token"`
	Layer           *int    `json:"layer,omitempty"`
	Operation       string  `json:"operation"`
	DType           string  `json:"dtype"`
	CacheLength     int     `json:"cache_length"`
	Grid            [3]int  `json:"grid"`
	Block           [3]int  `json:"block"`
	SharedMemory    *uint32 `json:"shared_memory,omitempty"`
	Lookahead       *bool   `json:"lookahead,omitempty"`
	Materialization *bool   `json:"materialization,omitempty"`
}

// Launch retains one kernel's capture-local identity and qualification evidence.
// Missing lists unavailable required fields. No inferred field qualifies work.
type Launch struct {
	Activity       int              `json:"activity"`
	Journal        string           `json:"journal"`
	RawSymbol      string           `json:"raw_symbol"`
	Start          int64            `json:"start_ns"`
	Duration       int64            `json:"duration_ns"`
	Identity       *Identity        `json:"identity,omitempty"`
	Metadata       *marker.Metadata `json:"metadata,omitempty"`
	APIStatus      string           `json:"api_status"`
	MetadataStatus string           `json:"metadata_status"`
	Missing        []string         `json:"unavailable,omitempty"`
	Reason         string           `json:"reason,omitempty"`
}

// Distribution summarizes retained launch durations, not independent run samples.
type Distribution struct {
	Count  int   `json:"count"`
	Min    int64 `json:"min_ns"`
	Median int64 `json:"median_ns"`
	Max    int64 `json:"max_ns"`
	Sum    int64 `json:"sum_ns"`
}

// Group contains uniquely matched work and its launch-duration distributions.
// Repeated identities remain unmatched because their occurrence order alone
// does not establish equivalence.
type Group struct {
	Identity        Identity     `json:"identity"`
	Base            []Launch     `json:"base"`
	Target          []Launch     `json:"target"`
	BaseDurations   Distribution `json:"base_durations"`
	TargetDurations Distribution `json:"target_durations"`
}

// Report is a shape qualification receipt, separate from capture completeness
// and iteration stationarity. Pass requires nonempty matched work, no unsupported graph replay, and no
// unmatched kernels. Even a pass does not establish causal speedup.
type Report struct {
	RequestPairs    []RequestPair     `json:"request_pairs,omitempty"`
	WorkScope       string            `json:"work_scope"`
	Diagnostics     []string          `json:"diagnostics,omitempty"`
	Version         int               `json:"version"`
	Pass            bool              `json:"pass"`
	Failures        []string          `json:"failures,omitempty"`
	Sources         map[string]string `json:"field_sources"`
	Groups          []Group           `json:"matched"`
	BaseUnmatched   []Launch          `json:"base_unmatched"`
	TargetUnmatched []Launch          `json:"target_unmatched"`
}

// Compare joins two decoded captures by observed application and launch fields.
// domain selects driver or runtime submission correlation. Missing metadata,
// ambiguous correlation, invalid durations, and unread geometry never match.
// The caller must retain independent completeness and stationarity receipts.
func Compare(base, target timeline.Records, domain string) Report {
	return CompareScoped(base, target, domain, nil)
}

// RequestPair explicitly declares equivalent requests across two captures.
// Journals and processes bind each capture-local ID; Role must agree in both.
// This declaration does not prove equal work, outputs or capture completeness.
type RequestPair struct {
	BaseJournal   string `json:"base_journal"`
	BaseProcess   uint32 `json:"base_process"`
	BaseID        string `json:"base_id"`
	TargetJournal string `json:"target_journal"`
	TargetProcess uint32 `json:"target_process"`
	TargetID      string `json:"target_id"`
	Role          string `json:"role"`
}

type requestKey struct {
	journal  string
	process  uint32
	id, role string
}

// CompareScoped compares shapes with explicitly paired v2 request identities.
// Cumulative execution counts are retained as separate work-scope diagnostics,
// not shape identity. A shape pass never qualifies an equal-work comparison.
func CompareScoped(base, target timeline.Records, domain string, pairs []RequestPair) Report {
	r := Report{Version: 1, RequestPairs: append([]RequestPair(nil), pairs...), WorkScope: "unavailable: legacy metadata or successful execution counters missing", Sources: map[string]string{
		"phase": "observed NVTX application marker", "token": "observed NVTX application marker", "layer": "observed when supplied; otherwise unavailable", "operation": "observed NVTX application marker", "dtype": "observed NVTX application marker", "cache_length": "observed NVTX application marker", "grid": "observed CUPTI activity", "block": "observed CUPTI activity", "raw_symbol": "observed CUPTI activity; templates preserved", "shared_memory": "observed when supplied; otherwise unavailable", "lookahead": "observed when supplied; otherwise unavailable", "materialization": "observed when supplied; otherwise unavailable",
	}, Groups: []Group{}, BaseUnmatched: []Launch{}, TargetUnmatched: []Launch{}}
	bm, tm := map[requestKey]int{}, map[requestKey]int{}
	for i, p := range pairs {
		bk, tk := requestKey{p.BaseJournal, p.BaseProcess, p.BaseID, p.Role}, requestKey{p.TargetJournal, p.TargetProcess, p.TargetID, p.Role}
		_, bd := bm[bk]
		_, td := tm[tk]
		if p.BaseJournal == "" || p.TargetJournal == "" || p.BaseProcess == 0 || p.TargetProcess == 0 || p.BaseID == "" || p.TargetID == "" || (p.Role != "warmup" && p.Role != "measured" && p.Role != "control") || bd || td {
			r.Failures = append(r.Failures, "invalid or duplicate request mapping")
			return r
		}
		bm[bk], tm[tk] = i, i
	}
	b, bu, bscope := collect(base, domain, bm)
	t, tu, tscope := collect(target, domain, tm)
	workKnown, workMismatch := len(pairs) > 0, false
	for i := range pairs {
		if !bscope[i] || !tscope[i] {
			r.Failures = append(r.Failures, "declared request mapping lacks observed qualifying work in both arms")
			workKnown = false
		}
	}
	r.BaseUnmatched, r.TargetUnmatched = bu, tu
	keys := make(map[string]bool)
	for k := range b {
		keys[k] = true
	}
	for k := range t {
		keys[k] = true
	}
	var sorted []string
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Strings(sorted)
	for _, k := range sorted {
		bs, ts := b[k], t[k]
		reason := ""
		switch {
		case len(bs) == 0 || len(ts) == 0:
			reason = "no equivalent observed identity"
		case len(bs) != len(ts):
			reason = "launch counts differ"
		case len(bs) != 1:
			reason = "ambiguous repeated identity"
		}
		if reason != "" {
			for _, l := range bs {
				l.Reason = reason
				r.BaseUnmatched = append(r.BaseUnmatched, l)
			}
			for _, l := range ts {
				l.Reason = reason
				r.TargetUnmatched = append(r.TargetUnmatched, l)
			}
			continue
		}
		mb, mt := bs[0].Metadata, ts[0].Metadata
		if mb == nil || mt == nil || mb.Executions == nil || mt.Executions == nil || mb.ExecutionUnit != "logical_forward" || mt.ExecutionUnit != "logical_forward" {
			workKnown = false
		} else if *mb.Executions != *mt.Executions {
			workMismatch = true
			r.Diagnostics = append(r.Diagnostics, "successful logical-forward counts differ for otherwise matched shape; prefill and delivery scheduling may differ")
		}
		r.Groups = append(r.Groups, Group{Identity: *bs[0].Identity, Base: bs, Target: ts, BaseDurations: distribution(bs), TargetDurations: distribution(ts)})
	}
	sort.Slice(r.BaseUnmatched, func(i, j int) bool { return r.BaseUnmatched[i].Activity < r.BaseUnmatched[j].Activity })
	sort.Slice(r.TargetUnmatched, func(i, j int) bool { return r.TargetUnmatched[i].Activity < r.TargetUnmatched[j].Activity })
	if len(r.Groups) == 0 {
		r.Failures = append(r.Failures, "no uniquely matched identities")
	}
	if len(r.BaseUnmatched) != 0 || len(r.TargetUnmatched) != 0 {
		r.Failures = append(r.Failures, "unmatched kernels remain")
	}
	if base.UnpairedMarkers != 0 || target.UnpairedMarkers != 0 {
		r.Failures = append(r.Failures, "unpaired NVTX marker halves")
	}
	if unsupportedGraphs(base) || unsupportedGraphs(target) {
		r.Failures = append(r.Failures, "graph replay has no qualified kernel identity")
	}
	if workKnown && len(r.Groups) > 0 && len(r.BaseUnmatched) == 0 && len(r.TargetUnmatched) == 0 {
		r.WorkScope = "observed cumulative successful logical-forward counts agree at matched scopes; execution-delta and completeness gates remain independent"
	}
	if workMismatch {
		r.WorkScope = "mismatch: cumulative successful logical-forward counts differ; shapes alone do not establish equal work"
	}
	r.Pass = len(r.Failures) == 0 && !unsupportedGraphs(base) && !unsupportedGraphs(target) && len(r.Groups) > 0 && len(r.BaseUnmatched) == 0 && len(r.TargetUnmatched) == 0 && base.UnpairedMarkers == 0 && target.UnpairedMarkers == 0
	return r
}

func collect(records timeline.Records, domain string, mappings map[requestKey]int) (map[string][]Launch, []Launch, map[int]bool) {
	groups := make(map[string][]Launch)
	seen := map[int]bool{}
	unmatched := []Launch{}
	links := timeline.Correlations(records, domain)
	for i, a := range records.Activities {
		if a.Subtype != "" && a.Subtype != "kernel" {
			continue
		}
		link := links[i]
		l := Launch{Activity: i, Journal: a.Journal, RawSymbol: a.Name, Start: a.Start, Duration: a.End - a.Start, APIStatus: link.APIStatus, MetadataStatus: link.MetadataStatus, Metadata: link.Metadata}
		m := link.Metadata
		var process uint32
		if link.Call >= 0 && link.Call < len(records.Calls) {
			process = records.Calls[link.Call].Process
		}
		if link.APIStatus != "matched" {
			l.Missing = append(l.Missing, "submission correlation")
		}
		if link.MetadataStatus != "matched" || m == nil {
			l.Missing = append(l.Missing, "application metadata")
		}
		if m == nil {
			l.Missing = append(l.Missing, "phase", "token", "operation", "dtype", "cache_length")
		}
		if m != nil {
			if m.Version == 2 {
				if _, ok := mappings[requestKey{a.Journal, process, m.RequestID, m.Role}]; !ok {
					l.Missing = append(l.Missing, "explicit request equivalence mapping")
				}
			}
			if m.Version == 2 && m.Boundary == "" {
				l.Missing = append(l.Missing, "explicit endpoint boundary")
			}
			if m.Token == nil {
				l.Missing = append(l.Missing, "token")
			}
			if m.Operation == "" {
				l.Missing = append(l.Missing, "operation")
			}
			if m.DType == nil {
				l.Missing = append(l.Missing, "dtype")
			}
			if m.CacheLength == nil {
				l.Missing = append(l.Missing, "cache_length")
			}
		}
		for _, n := range a.Grid {
			if n <= 0 {
				l.Missing = append(l.Missing, "grid")
				break
			}
		}
		for _, n := range a.Block {
			if n <= 0 {
				l.Missing = append(l.Missing, "block")
				break
			}
		}
		if a.End <= a.Start || l.Duration <= 0 {
			l.Missing = append(l.Missing, "valid duration")
		}
		if len(l.Missing) != 0 {
			l.Reason = "required evidence unavailable"
			unmatched = append(unmatched, l)
			continue
		}
		var pair *int
		if m.Version == 2 {
			v := mappings[requestKey{a.Journal, process, m.RequestID, m.Role}]
			pair = &v
			seen[v] = true
		}
		id := Identity{RequestPair: pair, Role: m.Role, Boundary: m.Boundary, Phase: m.Phase, Token: *m.Token, Layer: m.Layer, Operation: m.Operation, DType: *m.DType, CacheLength: *m.CacheLength, Grid: a.Grid, Block: a.Block, SharedMemory: a.SharedMemory, Lookahead: m.Lookahead, Materialization: m.Materialization}
		l.Identity = &id
		data, _ := json.Marshal(id) // Identity contains no unsupported JSON types.
		k := string(data)
		groups[k] = append(groups[k], l)
	}
	return groups, unmatched, seen
}

func distribution(launches []Launch) Distribution {
	d := Distribution{Count: len(launches)}
	if len(launches) == 0 {
		return d
	}
	values := make([]int64, len(launches))
	for i, l := range launches {
		values[i] = l.Duration
		d.Sum += l.Duration
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	d.Min, d.Max = values[0], values[len(values)-1]
	d.Median = values[len(values)/2]
	return d
}

func unsupportedGraphs(records timeline.Records) bool {
	// A global graph-node count does not prove every envelope is resolved.
	return len(records.Graphs) != 0
}
