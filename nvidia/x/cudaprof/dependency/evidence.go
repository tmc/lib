package dependency

import (
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Observe builds explicit ordering edges from retained native receipts. It
// preserves all device nodes, including unmatched ones. Successful callbacks
// must have unique paired ordinals and a unified clock. Only observed live
// nonblocking streams qualify stream ordering; default streams, overlapping
// submissions, unsupported APIs and graph internals remain coverage gaps.
// Observe does not select delivery endpoints or assert complete coverage.
func Observe(records timeline.Records) Graph {
	g := Graph{Nodes: []Node{}, Edges: []Edge{}}
	missing := map[string]bool{}
	note := func(s string) {
		if !missing[s] {
			missing[s] = true
			g.Missing = append(g.Missing, s)
		}
	}
	type key struct {
		journal, domain string
		correlation     uint32
	}
	activities := map[key][]int{}
	activityData := map[int]timeline.Activity{}
	graphs := observedGraphs(records)
	for i, a := range records.Activities {
		if a.Journal == "" || a.Start < 0 || a.End < a.Start {
			note("activity lacks a valid journal clock or interval")
			continue
		}
		id := fmt.Sprintf("activity:%d", i)
		g.Nodes = append(g.Nodes, Node{id, source(a.Journal, a.Sequence, "activity", i), a.Journal, a.Start, a.End})
		activityData[len(g.Nodes)-1] = a
		activities[key{a.Journal, "driver", a.Correlation}] = append(activities[key{a.Journal, "driver", a.Correlation}], len(g.Nodes)-1)
	}
	events := append([]timeline.DependencyEvent(nil), records.Dependencies...)
	sort.SliceStable(events, func(i, j int) bool {
		if events[i].Journal != events[j].Journal {
			return events[i].Journal < events[j].Journal
		}
		return events[i].Ordinal < events[j].Ordinal
	})
	type pair struct {
		enter, exit timeline.DependencyEvent
		valid       bool
	}
	pairs := map[key]*pair{}
	ordinals := map[string]map[uint64]bool{}
	for _, e := range events {
		if ordinals[e.Journal] == nil {
			ordinals[e.Journal] = map[uint64]bool{}
		}
		if e.Ordinal == 0 || ordinals[e.Journal][e.Ordinal] {
			note("native callback ordinals are missing or duplicated")
			continue
		}
		ordinals[e.Journal][e.Ordinal] = true
		k := key{e.Journal, e.Domain, e.Correlation}
		if e.Correlation == 0 || e.Journal == "" || e.TimestampDomain != "monotonic" {
			note("native dependency receipt lacks a correlation or unified clock")
			continue
		}
		p := pairs[k]
		if p == nil {
			p = &pair{valid: true}
			pairs[k] = p
		}
		switch e.Site {
		case "enter":
			if p.enter.Ordinal != 0 {
				p.valid = false
			}
			p.enter = e
		case "exit":
			if p.exit.Ordinal != 0 {
				p.valid = false
			}
			p.exit = e
		default:
			p.valid = false
		}
	}
	ordered := []*pair{}
	for _, p := range pairs {
		if !p.valid || p.enter.Ordinal == 0 || p.exit.Ordinal <= p.enter.Ordinal || p.enter.Kind != p.exit.Kind || p.enter.CallbackID != p.exit.CallbackID {
			note("native callback entry or exit is missing or ambiguous")
			continue
		}
		if p.exit.Success == nil {
			note("native API result is unavailable")
			continue
		}
		if !*p.exit.Success {
			continue
		}
		ordered = append(ordered, p)
	}
	sort.Slice(ordered, func(i, j int) bool {
		a, b := ordered[i].exit, ordered[j].exit
		if a.Journal != b.Journal {
			return a.Journal < b.Journal
		}
		return a.Ordinal < b.Ordinal
	})
	type handle struct {
		journal string
		context uint32
		value   uint64
	}
	type state struct {
		waits             map[int]timeline.DependencyEvent
		nodes             []int
		live, nonblocking bool
		boundary          uint64
	}
	streams := map[handle]*state{}
	type eventState struct {
		nodes  []int
		record *pair
		live   bool
	}
	handles := map[handle]*eventState{}
	linked := map[[2]int]bool{}
	edge := func(a, b int, kind string, e timeline.DependencyEvent) {
		if a == b || linked[[2]int{a, b}] {
			return
		}
		if g.Nodes[a].Clock != g.Nodes[b].Clock || g.Nodes[a].End > g.Nodes[b].Start {
			note("observed dependency endpoints lack ordered common-clock execution")
			return
		}
		linked[[2]int{a, b}] = true
		g.Edges = append(g.Edges, Edge{g.Nodes[a].ID, g.Nodes[b].ID, kind, source(e.Journal, e.Sequence, "dependency", int(e.Ordinal))})
	}
	matched := map[int]bool{}
	for _, p := range ordered {
		e := p.exit
		sk := handle{e.Journal, e.Context, e.Stream}
		ek := handle{e.Journal, e.Context, e.Event}
		s := streams[sk]
		switch e.Kind {
		case "graph_node", "graph_metadata":
			// Structural receipts are resolved through lifecycle versions.
		case "create_stream":
			if e.Stream == 0 || (s != nil && s.live) {
				note("stream creation identity is unavailable or reused while live")
				continue
			}
			streams[sk] = &state{live: true, nonblocking: e.Flags&1 != 0, boundary: e.Ordinal}
		case "destroy_stream":
			if s != nil {
				s.live = false
			}
		case "create_event":
			if e.Event == 0 || (handles[ek] != nil && handles[ek].live) {
				note("event creation identity is unavailable or reused while live")
				continue
			}
			handles[ek] = &eventState{live: true}
		case "destroy_event":
			if handles[ek] != nil {
				handles[ek].live = false
			}
		case "launch", "copy_to_host", "graph_launch", "event_record", "event_wait", "stream_sync":
			if s == nil || !s.live || !s.nonblocking || e.Stream == 0 {
				note("default, blocking or unobserved stream dependencies are unsupported")
				continue
			}
			if s.boundary >= p.enter.Ordinal {
				note("concurrent host submissions have ambiguous stream order")
				s.nodes = nil
				s.waits = nil
				continue
			}
			current := activities[key{e.Journal, e.Domain, e.Correlation}]
			switch e.Kind {
			case "launch", "copy_to_host", "graph_launch":
				if len(current) == 0 {
					note("successful submission has no retained device activity")
					continue
				}
				roots, leaves := current, current
				if e.Kind == "graph_launch" {
					r, l, edges, receipts, ok := graphs.graphExecution(e, current, activityData)
					if ok {
						roots, leaves = r, l
						for i, pair := range edges {
							edge(pair[0], pair[1], "graph_edge", receipts[i])
						}
					} else {
						note("graph replay lacks a complete explicit node and dependency snapshot")
					}
				} else if len(current) != 1 {
					note("multi-node submission lacks explicit internal dependencies")
				}
				for _, b := range current {
					matched[b] = true
				}
				for _, b := range roots {
					for _, a := range s.nodes {
						if wait, ok := s.waits[a]; ok {
							edge(a, b, "event_wait", wait)
						} else {
							edge(a, b, "stream_order", e)
						}
					}
				}
				s.nodes = leaves
				s.waits = nil
			case "event_record":
				v := handles[ek]
				if v == nil || !v.live {
					note("event record lacks an observed event lifetime")
					continue
				}
				if v.record != nil && v.record.exit.Ordinal >= p.enter.Ordinal {
					note("concurrent event records are ambiguous")
					v.record = nil
					v.nodes = nil
					continue
				}
				v.nodes = append([]int(nil), s.nodes...)
				v.record = p
			case "event_wait":
				v := handles[ek]
				if v == nil || !v.live || v.record == nil || v.record.exit.Ordinal >= p.enter.Ordinal {
					note("event wait lacks a unique completed record submission")
					continue
				}
				// Retain both predecessors. The next stream operation cannot execute
				// until the preceding stream work and the recorded event complete.
				s.nodes = append(append([]int(nil), s.nodes...), v.nodes...)
				// Mark event predecessors distinctly when the next submission arrives.
				if s.waits == nil {
					s.waits = map[int]timeline.DependencyEvent{}
				}
				for _, a := range v.nodes {
					s.waits[a] = e
				}
			case "stream_sync":
				if e.Start < 0 {
					note("synchronization completion timestamp is invalid")
					continue
				}
				n := len(g.Nodes)
				g.Nodes = append(g.Nodes, Node{fmt.Sprintf("sync:%s:%d", e.Journal, e.Ordinal), source(e.Journal, e.Sequence, "dependency", int(e.Ordinal)), e.Journal, e.End, e.End})
				for _, a := range s.nodes {
					edge(a, n, "host_materialization", e)
				}
				s.nodes = []int{n}
				s.waits = nil
			}
			s.boundary = e.Ordinal
		case "event_sync", "context_sync":
			predecessors := []int{}
			if e.Kind == "event_sync" {
				v := handles[ek]
				if v == nil || !v.live || v.record == nil || v.record.exit.Ordinal >= p.enter.Ordinal {
					note("event synchronization lacks a unique recorded event lifetime")
					continue
				}
				predecessors = append(predecessors, v.nodes...)
			} else {
				if e.Context == 0 {
					note("context synchronization lacks context identity")
					continue
				}
				for k, stream := range streams {
					if k.journal == e.Journal && k.context == e.Context && stream.live {
						if stream.boundary >= p.enter.Ordinal {
							note("concurrent context synchronization is ambiguous")
							continue
						}
						predecessors = append(predecessors, stream.nodes...)
					}
				}
			}
			if e.End < 0 {
				note("synchronization completion timestamp is invalid")
				continue
			}
			n := len(g.Nodes)
			g.Nodes = append(g.Nodes, Node{fmt.Sprintf("sync:%s:%d", e.Journal, e.Ordinal), source(e.Journal, e.Sequence, "dependency", int(e.Ordinal)), e.Journal, e.End, e.End})
			sort.Ints(predecessors)
			for _, a := range predecessors {
				edge(a, n, "host_materialization", e)
			}
		case "unsupported":
			note("unsupported work-affecting native API is retained")
		default:
			note("unknown native dependency event is retained")
		}
	}
	for i := range g.Nodes {
		if len(g.Nodes[i].ID) >= 9 && g.Nodes[i].ID[:9] == "activity:" && !matched[i] {
			note("device activity has no uniquely matched supported submission")
		}
	}
	if len(events) == 0 {
		note("native stream and event dependency receipts are not retained")
	}
	note("a selected delivery start and host materialization endpoint are unverified")
	note("native dependency coverage is partial")
	return g
}

func source(journal string, sequence uint64, kind string, index int) string {
	if sequence != 0 {
		return fmt.Sprintf("%s:record:%d", journal, sequence)
	}
	return fmt.Sprintf("%s:%s:%d", journal, kind, index)
}
