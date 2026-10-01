package dependency

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Selection identifies one controlled request. ExpectedActivities must come
// from an independent workload invariant, not the observed trace count. The
// capture must contain only this request's device work. Its submitting thread
// must enclose submissions in a phase/token range named operation "token" and
// a successful synchronization in a materialization range, followed by one
// instantaneous delivered marker with the same token. Unsupported native APIs
// and disconnected work prevent qualification.
type Selection struct {
	RequestID          string `json:"request_id,omitempty"`
	Role               string `json:"role,omitempty"`
	Journal            string `json:"journal"`
	Process            uint32 `json:"process"`
	Thread             uint32 `json:"thread"`
	Token              int    `json:"token"`
	ExpectedActivities int    `json:"expected_activities"`
}

// Delivery selects a controlled delivery graph from explicit native evidence
// and independently supplied workload counts. A complete result qualifies
// only this bounded request and the collector's supported dependencies. It
// does not establish application-wide coverage or attribute engine differences.
func Delivery(records timeline.Records, selection Selection) Graph {
	g := Observe(records)
	reject := func(reason string) Graph { g.Missing = append(g.Missing, reason); return g }
	if selection.Journal == "" || selection.Process == 0 || selection.Thread == 0 || selection.Token < 0 || selection.ExpectedActivities <= 0 {
		return reject("delivery selection lacks scoped identities or an independent activity invariant")
	}
	if len(records.Activities) != selection.ExpectedActivities {
		return reject("independent request activity invariant does not match retained device records")
	}
	scoped := func(journal string, process, thread uint32) bool {
		return journal == selection.Journal && process == selection.Process && thread == selection.Thread
	}
	token := func(name string) (marker.Metadata, bool) {
		m, err := marker.ParseRange(name)
		return m, err == nil && (m.Version != 2 || (m.RequestID == selection.RequestID && m.Role == selection.Role)) && m.Token != nil && *m.Token == selection.Token && m.Event == ""
	}
	var boundary *timeline.Range
	var materializations []timeline.Range
	for _, r := range records.Ranges {
		if !scoped(r.Journal, r.Process, r.Thread) {
			continue
		}
		m, ok := token(r.Name)
		if !ok {
			continue
		}
		if m.Operation == "token" {
			if boundary != nil {
				return reject("selected token start range is ambiguous")
			}
			copy := r
			boundary = &copy
		}
		if m.Materialization != nil && *m.Materialization {
			materializations = append(materializations, r)
		}
	}
	if boundary == nil || boundary.Start < 0 || boundary.End < boundary.Start {
		return reject("selected token start range is missing or invalid")
	}
	var delivered *timeline.Marker
	for _, m := range records.Markers {
		metadata, err := marker.ParseRange(m.Name)
		if err != nil || (metadata.Version == 2 && (metadata.RequestID != selection.RequestID || metadata.Role != selection.Role || metadata.Boundary != "consumer_delivery")) || metadata.Event != "delivered" || metadata.Token == nil || *metadata.Token != selection.Token || !scoped(m.Journal, m.Process, m.Thread) || m.Flags != timeline.MarkerInstantaneous {
			continue
		}
		if delivered != nil {
			return reject("selected delivered marker is ambiguous")
		}
		copy := m
		delivered = &copy
	}
	if delivered == nil || delivered.Time < boundary.End {
		return reject("selected delivered endpoint is missing or precedes the token boundary")
	}
	calls := map[submissionKey][]timeline.Call{}
	for _, c := range records.Calls {
		calls[submissionKey{c.Journal, c.Domain, c.Correlation}] = append(calls[submissionKey{c.Journal, c.Domain, c.Correlation}], c)
	}
	var completions []int
	byID := map[string]int{}
	for i, n := range g.Nodes {
		byID[n.ID] = i
	}
	for _, e := range records.Dependencies {
		if e.Journal != selection.Journal || e.Site != "exit" || e.Success == nil || !*e.Success || (e.Kind != "stream_sync" && e.Kind != "context_sync" && e.Kind != "event_sync") {
			continue
		}
		api := calls[submissionKey{e.Journal, e.Domain, e.Correlation}]
		if len(api) != 1 || !scoped(api[0].Journal, api[0].Process, api[0].Thread) {
			continue
		}
		for _, r := range materializations {
			if r.Start <= api[0].Start && api[0].End <= r.End && r.Start <= e.End && e.End <= r.End && r.End <= delivered.Time {
				if i, ok := byID[fmt.Sprintf("sync:%s:%d", e.Journal, e.Ordinal)]; ok {
					completions = append(completions, i)
				}
			}
		}
	}
	if len(completions) != 1 {
		return reject("selected materialization lacks one uniquely observed successful synchronization completion")
	}
	// An observed range on the submitting thread establishes program order,
	// not a dependency guessed from neighboring activity timestamps.
	for _, a := range records.Activities {
		if a.Journal != selection.Journal {
			return reject("request includes work from another process generation")
		}
		api := calls[submissionKey{a.Journal, "driver", a.Correlation}]
		if len(api) != 1 || !scoped(api[0].Journal, api[0].Process, api[0].Thread) || api[0].Start < boundary.Start || api[0].End > boundary.End {
			return reject("device work lacks one submitting API inside the selected token range")
		}
	}
	startRecord := ""
	for _, m := range records.Markers {
		if scoped(m.Journal, m.Process, m.Thread) && m.Name == boundary.Name && m.Time == boundary.Start && m.Flags&timeline.MarkerStart != 0 {
			if startRecord != "" {
				return reject("token start marker source is ambiguous")
			}
			startRecord = source(m.Journal, m.Sequence, "marker", 0)
		}
	}
	if startRecord == "" {
		return reject("token start lacks a retained raw marker source")
	}
	start := len(g.Nodes)
	g.Start = "delivery:start"
	g.Nodes = append(g.Nodes, Node{g.Start, startRecord, selection.Journal, boundary.Start, boundary.Start})
	end := len(g.Nodes)
	g.Delivery = "delivery:delivered"
	g.Nodes = append(g.Nodes, Node{g.Delivery, source(delivered.Journal, delivered.Sequence, "marker", 0), selection.Journal, delivered.Time, delivered.Time})
	incoming := map[string]bool{}
	for _, e := range g.Edges {
		incoming[e.To] = true
	}
	for i, n := range g.Nodes[:start] {
		if strings.HasPrefix(n.ID, "activity:") && !incoming[n.ID] {
			if n.Start < boundary.Start {
				return reject("device execution precedes the observed token start")
			}
			g.Edges = append(g.Edges, Edge{g.Start, g.Nodes[i].ID, "application_order", startRecord})
		}
	}
	sync := g.Nodes[completions[0]]
	g.Edges = append(g.Edges, Edge{sync.ID, g.Delivery, "host_materialization", sync.Record})
	// Every retained work node must lie between the selected boundaries. An
	// independent branch not joined by materialization is a coverage failure.
	next := map[string][]string{}
	previous := map[string][]string{}
	for _, e := range g.Edges {
		next[e.From] = append(next[e.From], e.To)
		previous[e.To] = append(previous[e.To], e.From)
	}
	reachable := func(root string, edges map[string][]string) map[string]bool {
		seen := map[string]bool{}
		queue := []string{root}
		for len(queue) > 0 {
			id := queue[0]
			queue = queue[1:]
			if seen[id] {
				continue
			}
			seen[id] = true
			queue = append(queue, edges[id]...)
		}
		return seen
	}
	forward, backward := reachable(g.Nodes[start].ID, next), reachable(g.Nodes[end].ID, previous)
	for _, n := range g.Nodes[:start] {
		if !forward[n.ID] || !backward[n.ID] {
			return reject("retained work is disconnected from the selected materialization endpoint")
		}
	}
	kept := g.Missing[:0]
	for _, reason := range g.Missing {
		if reason != "a selected delivery start and host materialization endpoint are unverified" && reason != "native dependency coverage is partial" {
			kept = append(kept, reason)
		}
	}
	g.Missing = kept
	g.Complete = len(g.Missing) == 0
	return g
}
