package overview

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Reference identifies a source record in the decoded capture.
type Reference struct {
	Journal string `json:"journal"`
	Kind    string `json:"kind"`
	Index   int    `json:"index"`
}

// Interval is a retained time interval, not an inference about device occupancy.
type Interval struct {
	Journal string `json:"journal"`
	Start   int64  `json:"start_ns"`
	End     int64  `json:"end_ns"`
}

// Metrics keeps elapsed span, activity union, and summed kernel time distinct.
// GPUUnion holds one union per device; devices are never merged into one clock.
type Metrics struct {
	HostRanges     []Interval            `json:"host_ranges"`
	HostUnionNanos int64                 `json:"host_union_ns"`
	GPUSpan        *Interval             `json:"gpu_span,omitempty"`
	GPUUnion       map[uint32][]Interval `json:"gpu_union"`
	KernelSumNanos int64                 `json:"kernel_sum_ns"`
	KernelCount    int                   `json:"kernel_count"`
}

// Node is one hierarchy group or activity leaf. Provenance follows retained
// metadata resolution: observed, missing, ambiguous or unsupported.
// References identify contributing records; metrics deduplicate references.
type Node struct {
	ID         string      `json:"id"`
	Kind       string      `json:"kind"`
	Label      string      `json:"label"`
	Provenance string      `json:"provenance"`
	References []Reference `json:"references"`
	Metrics    Metrics     `json:"metrics"`
	Children   []*Node     `json:"children"`
}

// Token is an observed application token or a separately inferred interval.
// DeliveryLatency measures earliest observed host scope to explicit delivered
// event, not engine request latency. Nil is unavailable, never zero.
type Token struct {
	Process               uint32      `json:"process,omitempty"`
	ExecutionUnit         string      `json:"execution_unit,omitempty"`
	RequestID             string      `json:"request_id,omitempty"`
	Role                  string      `json:"role,omitempty"`
	Boundary              string      `json:"boundary,omitempty"`
	Executions            *int        `json:"executions,omitempty"`
	InferredBoundary      *Interval   `json:"inferred_boundary,omitempty"`
	ID                    string      `json:"id"`
	Journal               string      `json:"journal"`
	Phase                 string      `json:"phase"`
	Index                 int         `json:"token"`
	Provenance            string      `json:"provenance"`
	Metrics               Metrics     `json:"metrics"`
	References            []Reference `json:"references"`
	Crossing              []Reference `json:"crossing_records"`
	MaterializationRanges []Interval  `json:"materialization_host_ranges"`
	Materialization       []Reference `json:"materialization"`
	Delivery              *int64      `json:"delivered_at_ns,omitempty"`
	DeliveryLatency       *int64      `json:"host_scope_to_delivery_ns,omitempty"`
	EndReason             string      `json:"end_reason,omitempty"`
	Availability          []string    `json:"unavailable"`
}

// Report is a versioned token and hierarchy view. It retains unattributed work
// and unpaired range evidence. Capture qualification remains caller-owned.
type Report struct {
	Version         int         `json:"version"`
	TimestampDomain string      `json:"timestamp_domain"`
	Hierarchy       *Node       `json:"hierarchy"`
	Tokens          []Token     `json:"tokens"`
	Inferred        []Token     `json:"inferred_intervals"`
	Unmatched       []Reference `json:"unmatched"`
	UnpairedMarkers int         `json:"unpaired_markers"`
	Limits          []string    `json:"limits"`
}

// Analyze builds observed groups using exact supported submission links.
// inferred, when nonnil, matches raw kernel symbols to create visibly inferred
// intervals. It supplies neither delivered events nor materialization evidence.
func Analyze(records timeline.Records, domain string, inferred *regexp.Regexp) Report {
	root := node("capture", "capture", "observed", "capture")
	r := Report{Version: 1, TimestampDomain: "CUPTI journal timestamps; cross-journal clock equivalence must be qualified by the caller", Hierarchy: root, Tokens: []Token{}, Inferred: []Token{}, Unmatched: []Reference{}, UnpairedMarkers: records.UnpairedMarkers, Limits: []string{"submission context does not prove kernel duration causes", "host scopes, GPU union, kernel sums and delivery observations are distinct", "metadata does not inherit; cross-thread overlap is not submission evidence", "inferred kernel boundaries are not delivered or materialized events", "missing journal identity or invalid timestamps are excluded from time metrics; activity references remain visible", "kernel-marker inference is unavailable for journals containing multiple device IDs", "legacy v1 phase/token numbers do not identify requests; v2 retains process/request/role identities but lifecycle completeness is independently qualified"}}
	links := timeline.Correlations(records, domain)
	groups := map[string]*Node{}
	tokens := map[string]*Token{}
	ensureToken := func(journal string, process uint32, m marker.Metadata) *Token {
		if m.Token == nil {
			return nil
		}
		key := tokenKey(journal, m.Phase, *m.Token)
		if m.Version == 2 {
			key = fmt.Sprintf("%s/process:%d/request:%q/role:%q", key, process, m.RequestID, m.Role)
		}
		if tokens[key] == nil {
			tokens[key] = &Token{Process: process, ExecutionUnit: m.ExecutionUnit, RequestID: m.RequestID, Role: m.Role, Boundary: m.Boundary, Executions: m.Executions, ID: stableID(key), Journal: journal, Phase: m.Phase, Index: *m.Token, Provenance: "observed", References: []Reference{}, Crossing: []Reference{}, Materialization: []Reference{}, Availability: []string{}}
		}
		return tokens[key]
	}
	ensureHierarchy := func(journal string, process uint32, m *marker.Metadata) []*Node {
		parent, prefix := root, ""
		if m.Version == 2 {
			prefix = fmt.Sprintf("journal:%s/process:%d/request:%q/role:%q/", journal, process, m.RequestID, m.Role)
			parent = child(root, groups, "request", m.RequestID+" ("+m.Role+")", "observed", prefix)
		}
		phase := child(parent, groups, "phase", m.Phase, "observed", prefix+"phase:"+m.Phase)
		tokenLabel, tokenProv := "unknown token", "missing"
		tokenValue := "unknown"
		if m.Token != nil {
			tokenLabel = fmt.Sprintf("token %d", *m.Token)
			tokenValue = strconv.Itoa(*m.Token)
			tokenProv = "observed"
		}
		key := prefix + "phase:" + m.Phase + "/journal:" + journal + "/token:" + tokenValue
		tok := child(phase, groups, "token", tokenLabel, tokenProv, key)
		layerLabel, layerProv, layerValue := "unknown layer", "missing", "unknown"
		if m.Layer != nil {
			layerLabel = fmt.Sprintf("layer %d", *m.Layer)
			layerValue = strconv.Itoa(*m.Layer)
			layerProv = "observed"
		}
		layer := child(tok, groups, "layer", layerLabel, layerProv, key+"/layer:"+layerValue)
		opLabel, opProv := m.Operation, "observed"
		if opLabel == "" {
			opLabel = "unknown operation"
			opProv = "missing"
		}
		op := child(layer, groups, "operation", opLabel, opProv, key+"/layer:"+layerValue+"/op:"+opLabel)
		return []*Node{phase, tok, layer, op}
	}
	// A range by itself is host scope, not evidence of kernel membership.
	for i, scope := range records.Ranges {
		m, err := marker.ParseRange(scope.Name)
		if err != nil || m.Event != "" || invalidScope(records.Ranges, i) || scope.Journal == "" || scope.Start < 0 || scope.End <= scope.Start || scope.Process == 0 || scope.Thread == 0 {
			continue
		}
		ref := Reference{scope.Journal, "range", i}
		root.References = appendUnique(root.References, ref)
		for _, n := range ensureHierarchy(scope.Journal, scope.Process, &m) {
			n.References = appendUnique(n.References, ref)
		}
		if t := ensureToken(scope.Journal, scope.Process, m); t != nil {
			t.References = appendUnique(t.References, ref)
			if m.Materialization != nil && *m.Materialization {
				t.Materialization = append(t.Materialization, ref)
			}
		}
	}
	for i, a := range records.Activities {
		ref := Reference{a.Journal, "activity", i}
		root.References = append(root.References, ref)
		link := links[i]
		m := link.Metadata
		if m != nil && m.Event != "" {
			link.MetadataStatus = "unsupported"
		}
		if link.MetadataStatus != "matched" || m == nil {
			r.Unmatched = append(r.Unmatched, ref)
			unknown := child(root, groups, "unknown", "unattributed", "missing", "unknown")
			unknown.References = append(unknown.References, ref)
			provenance := link.MetadataStatus
			if link.APIStatus == "ambiguous" || link.RangeStatus == "ambiguous" {
				provenance = "ambiguous"
			}
			leaf := node(activityKind(a), a.Name, provenance, fmt.Sprintf("activity:%s:%d", a.Journal, i))
			leaf.References = []Reference{ref}
			unknown.Children = append(unknown.Children, leaf)
			continue
		}
		nodes := ensureHierarchy(a.Journal, records.Calls[link.Call].Process, m)
		op := nodes[len(nodes)-1]
		for _, n := range nodes {
			n.References = appendUnique(n.References, ref)
			if link.Call >= 0 {
				n.References = appendUnique(n.References, Reference{records.Calls[link.Call].Journal, "call", link.Call})
			}
			if link.Range >= 0 {
				n.References = appendUnique(n.References, Reference{records.Ranges[link.Range].Journal, "range", link.Range})
			}
		}
		leaf := node(activityKind(a), a.Name, "observed", fmt.Sprintf("activity:%s:%d", a.Journal, i))
		leaf.References = []Reference{ref}
		op.Children = append(op.Children, leaf)
		if t := ensureToken(a.Journal, records.Calls[link.Call].Process, *m); t != nil {
			t.References = appendUnique(t.References, ref)
			if link.Call >= 0 {
				t.References = appendUnique(t.References, Reference{records.Calls[link.Call].Journal, "call", link.Call})
			}
			if link.Range >= 0 {
				t.References = appendUnique(t.References, Reference{a.Journal, "range", link.Range})
			}
		}
	}
	// Instant delivery boundaries require explicit per-token metadata.
	for i, event := range records.Markers {
		if event.Flags != timeline.MarkerInstantaneous || event.Journal == "" || event.Time < 0 || event.Process == 0 || event.Thread == 0 {
			continue
		}
		m, err := marker.ParseRange(event.Name)
		if err != nil || m.Event == "" || m.Token == nil {
			continue
		}
		t := ensureToken(event.Journal, event.Process, m)
		if m.Version == 2 {
			t.Boundary, t.Executions = m.Boundary, m.Executions
		}
		ref := Reference{event.Journal, "marker", i}
		t.References = appendUnique(t.References, ref)
		switch m.Event {
		case "delivered":
			if m.Version == 2 && m.Boundary != "consumer_delivery" {
				t.Availability = append(t.Availability, "delivered event lacks consumer_delivery boundary")
				continue
			}
			if t.Delivery != nil {
				t.Delivery = nil
				t.Availability = append(t.Availability, "ambiguous delivered events")
			} else if !contains(t.Availability, "ambiguous delivered events") {
				when := event.Time
				t.Delivery = &when
			}
		case "eos", "token_limit":
			if t.EndReason != "" && t.EndReason != m.Event {
				t.EndReason = "ambiguous"
			} else {
				t.EndReason = m.Event
			}
		}
	}
	for _, t := range tokens {
		t.Metrics = measure(records, t.References)
		t.MaterializationRanges = measure(records, t.Materialization).HostRanges
		if t.Delivery == nil {
			t.Availability = append(t.Availability, "delivery event")
		}
		if len(t.Materialization) == 0 {
			t.Availability = append(t.Availability, "materialization scope")
		}
		if t.Delivery != nil && len(t.Metrics.HostRanges) > 0 {
			start := t.Metrics.HostRanges[0].Start
			if *t.Delivery >= start {
				latency := *t.Delivery - start
				t.DeliveryLatency = &latency
			} else {
				t.Availability = append(t.Availability, "delivery precedes host scope")
			}
		}
		if len(t.Metrics.HostRanges) > 0 {
			end := t.Metrics.HostRanges[len(t.Metrics.HostRanges)-1].End
			for _, ref := range t.References {
				if ref.Kind == "activity" && records.Activities[ref.Index].End > end {
					t.Crossing = appendUnique(t.Crossing, ref)
				}
			}
		}
		r.Tokens = append(r.Tokens, *t)
	}
	sort.Slice(r.Tokens, func(i, j int) bool {
		a, b := r.Tokens[i], r.Tokens[j]
		if a.Journal != b.Journal {
			return a.Journal < b.Journal
		}
		if a.RequestID != b.RequestID {
			return a.RequestID < b.RequestID
		}
		if a.Role != b.Role {
			return a.Role < b.Role
		}
		if a.ID != b.ID && a.Index == b.Index && a.Phase == b.Phase {
			return a.ID < b.ID
		}
		if a.Phase != b.Phase {
			return a.Phase < b.Phase
		}
		return a.Index < b.Index
	})
	populate(records, root)
	if inferred != nil {
		r.Inferred = infer(records, inferred)
	}
	return r
}

func node(kind, label, provenance, key string) *Node {
	return &Node{ID: stableID(key), Kind: kind, Label: label, Provenance: provenance, References: []Reference{}, Children: []*Node{}}
}
func child(parent *Node, groups map[string]*Node, kind, label, provenance, key string) *Node {
	if groups[key] == nil {
		groups[key] = node(kind, label, provenance, key)
		parent.Children = append(parent.Children, groups[key])
	}
	return groups[key]
}
func stableID(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:12]) }
func tokenKey(journal, phase string, index int) string {
	return fmt.Sprintf("%s\x00%s\x00%d", journal, phase, index)
}
func appendUnique(refs []Reference, ref Reference) []Reference {
	for _, r := range refs {
		if r == ref {
			return refs
		}
	}
	return append(refs, ref)
}
func contains(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}
func populate(records timeline.Records, n *Node) {
	n.Metrics = measure(records, n.References)
	sort.Slice(n.Children, func(i, j int) bool {
		a, b := n.Children[i], n.Children[j]
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		if a.Label != b.Label {
			return a.Label < b.Label
		}
		return a.ID < b.ID
	})
	for _, c := range n.Children {
		populate(records, c)
	}
}

func measure(records timeline.Records, refs []Reference) Metrics {
	m := Metrics{HostRanges: []Interval{}, GPUUnion: map[uint32][]Interval{}}
	seen := map[Reference]bool{}
	for _, ref := range refs {
		if seen[ref] {
			continue
		}
		seen[ref] = true
		switch ref.Kind {
		case "activity":
			a := records.Activities[ref.Index]
			if a.Journal == "" || a.Start < 0 || a.End <= a.Start {
				continue
			}
			m.GPUUnion[a.Device] = append(m.GPUUnion[a.Device], Interval{a.Journal, a.Start, a.End})
			if m.GPUSpan == nil {
				m.GPUSpan = &Interval{a.Journal, a.Start, a.End}
			} else if m.GPUSpan.Journal == a.Journal {
				m.GPUSpan.Start = min(m.GPUSpan.Start, a.Start)
				m.GPUSpan.End = max(m.GPUSpan.End, a.End)
			}
			if a.Subtype == "kernel" {
				m.KernelCount++
				m.KernelSumNanos += a.End - a.Start
			}
		case "range":
			r := records.Ranges[ref.Index]
			if r.Journal != "" && r.Start >= 0 && r.End > r.Start {
				m.HostRanges = append(m.HostRanges, Interval{r.Journal, r.Start, r.End})
			}
		}
	}
	journals := map[string]bool{}
	devices := map[uint32]bool{}
	for _, ref := range refs {
		if ref.Kind == "activity" {
			journals[ref.Journal] = true
			devices[records.Activities[ref.Index].Device] = true
		}
	}
	if len(journals) > 1 || len(devices) > 1 {
		m.GPUSpan = nil
	}
	m.HostRanges = merge(m.HostRanges)
	for _, span := range m.HostRanges {
		m.HostUnionNanos += span.End - span.Start
	}
	for device, spans := range m.GPUUnion {
		m.GPUUnion[device] = merge(spans)
	}
	return m
}
func merge(spans []Interval) []Interval {
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].Journal != spans[j].Journal {
			return spans[i].Journal < spans[j].Journal
		}
		if spans[i].Start != spans[j].Start {
			return spans[i].Start < spans[j].Start
		}
		return spans[i].End < spans[j].End
	})
	out := []Interval{}
	for _, s := range spans {
		if len(out) == 0 || out[len(out)-1].Journal != s.Journal || out[len(out)-1].End < s.Start {
			out = append(out, s)
		} else {
			out[len(out)-1].End = max(out[len(out)-1].End, s.End)
		}
	}
	return out
}
func infer(records timeline.Records, re *regexp.Regexp) []Token {
	byJournal := map[string][]int64{}
	devices := map[string]map[uint32]bool{}
	for _, a := range records.Activities {
		if devices[a.Journal] == nil {
			devices[a.Journal] = map[uint32]bool{}
		}
		devices[a.Journal][a.Device] = true
	}
	for _, a := range records.Activities {
		if a.Journal != "" && a.Start >= 0 && a.End > a.Start && a.Subtype == "kernel" && re.MatchString(a.Name) {
			byJournal[a.Journal] = append(byJournal[a.Journal], a.Start)
		}
	}
	out := []Token{}
	for journal, times := range byJournal {
		if len(devices[journal]) != 1 {
			continue
		}
		sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
		for i := 0; i+1 < len(times); i++ {
			start, end := times[i], times[i+1]
			if end <= start {
				continue
			}
			t := Token{ID: stableID(tokenKey(journal, "inferred", i)), Journal: journal, Phase: "unknown", Index: i, Provenance: "inferred kernel-marker interval", References: []Reference{}, Crossing: []Reference{}, Materialization: []Reference{}, Availability: []string{"observed token identity", "delivery event", "materialization scope"}}
			for j, a := range records.Activities {
				if a.Journal != journal || a.End <= start || a.Start >= end {
					continue
				}
				ref := Reference{journal, "activity", j}
				if a.Start < start || a.End > end {
					t.Crossing = append(t.Crossing, ref)
				} else {
					t.References = append(t.References, ref)
				}
			}
			t.InferredBoundary = &Interval{Journal: journal, Start: start, End: end}
			t.Metrics = measure(records, t.References)
			out = append(out, t)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Journal != out[j].Journal {
			return strings.Compare(out[i].Journal, out[j].Journal) < 0
		}
		return out[i].Index < out[j].Index
	})
	return out
}

func invalidScope(ranges []timeline.Range, index int) bool {
	r := ranges[index]
	for i, s := range ranges {
		if i == index || s.Journal != r.Journal || s.Process != r.Process || s.Thread != r.Thread {
			continue
		}
		if r.Start < s.Start && s.Start < r.End && r.End < s.End || s.Start < r.Start && r.Start < s.End && s.End < r.End {
			return true
		}
	}
	return false
}

func activityKind(a timeline.Activity) string {
	if a.Subtype == "" {
		return "activity"
	}
	return a.Subtype
}
