package dependency

import (
	"fmt"
	"math"
)

// Node describes work in one clock domain. Record names its raw evidence.
// Durations exclude waiting outside the node. IDs must be unique.
type Node struct {
	ID     string `json:"id"`
	Record string `json:"record"`
	Clock  string `json:"clock"`
	Start  int64  `json:"start_ns"`
	End    int64  `json:"end_ns"`
}

// Edge identifies an explicit completion-before-start dependency. Kind is
// stream_order, event_wait, graph_edge or host_materialization. Record must
// identify retained ordering evidence, not a timestamp or submission link.
type Edge struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Kind   string `json:"kind"`
	Record string `json:"record"`
}

// Graph retains a caller-verified dependency inventory. Complete may be true
// only when all dependencies and work between Start and Delivery are retained.
// An analysis cannot independently establish that a native collector did so.
type Graph struct {
	Nodes    []Node   `json:"nodes"`
	Edges    []Edge   `json:"edges"`
	Start    string   `json:"start"`
	Delivery string   `json:"delivery"`
	Complete bool     `json:"complete"`
	Missing  []string `json:"missing"`
}

// Report preserves graph evidence and qualification. CriticalPath lists only
// a complete delivery graph's longest work path; WorkNS excludes waiting gaps
// and is not delivery latency or causal attribution of a performance change.
type Report struct {
	Graph        Graph    `json:"graph"`
	State        string   `json:"state"`
	Reasons      []string `json:"reasons"`
	CriticalPath []string `json:"critical_path,omitempty"`
	WorkNS       int64    `json:"work_ns,omitempty"`
}

// Analyze validates explicit edges and finds a longest path only for complete
// evidence with known endpoints. Independent concurrent branches are not summed.
// Nodes and edges remain available in partial reports. It rejects cycles,
// overlapping dependency endpoints, invalid clocks and missing source records.
func Analyze(g Graph) (Report, error) {
	r := Report{Graph: g, State: "partial", Reasons: append([]string(nil), g.Missing...)}
	ids := make(map[string]int, len(g.Nodes))
	in := make([]int, len(g.Nodes))
	next := make([][]int, len(g.Nodes))
	for i, n := range g.Nodes {
		if n.ID == "" || n.Record == "" || n.Clock == "" || n.Start < 0 || n.End < n.Start {
			return r, fmt.Errorf("dependency: invalid node %d", i)
		}
		if _, ok := ids[n.ID]; ok {
			return r, fmt.Errorf("dependency: duplicate node %q", n.ID)
		}
		ids[n.ID] = i
	}
	seen := map[[2]int]bool{}
	for i, e := range g.Edges {
		a, okA := ids[e.From]
		b, okB := ids[e.To]
		if !okA || !okB || a == b || e.Record == "" {
			return r, fmt.Errorf("dependency: invalid edge %d", i)
		}
		switch e.Kind {
		case "stream_order", "event_wait", "graph_edge", "host_materialization":
		default:
			return r, fmt.Errorf("dependency: unsupported edge kind %q", e.Kind)
		}
		if g.Nodes[a].Clock != g.Nodes[b].Clock || g.Nodes[a].End > g.Nodes[b].Start {
			return r, fmt.Errorf("dependency: edge %d lacks ordered common-clock endpoints", i)
		}
		key := [2]int{a, b}
		if seen[key] {
			return r, fmt.Errorf("dependency: duplicate edge %d", i)
		}
		seen[key] = true
		next[a] = append(next[a], b)
		in[b]++
	}
	queue := []int{}
	for i, n := range in {
		if n == 0 {
			queue = append(queue, i)
		}
	}
	order := []int{}
	for len(queue) > 0 {
		a := queue[0]
		queue = queue[1:]
		order = append(order, a)
		for _, b := range next[a] {
			in[b]--
			if in[b] == 0 {
				queue = append(queue, b)
			}
		}
	}
	if len(order) != len(g.Nodes) {
		return r, fmt.Errorf("dependency: cycle")
	}
	start, haveStart := ids[g.Start]
	end, haveEnd := ids[g.Delivery]
	if !g.Complete {
		r.Reasons = append(r.Reasons, "dependency inventory is incomplete")
	}
	if !haveStart {
		r.Reasons = append(r.Reasons, "delivery start endpoint is missing")
	}
	if !haveEnd {
		r.Reasons = append(r.Reasons, "delivery completion endpoint is missing")
	}
	if len(r.Reasons) != 0 {
		return r, nil
	}
	distance := make([]int64, len(g.Nodes))
	previous := make([]int, len(g.Nodes))
	for i := range distance {
		distance[i] = -1
		previous[i] = -1
	}
	distance[start] = g.Nodes[start].End - g.Nodes[start].Start
	for _, a := range order {
		if distance[a] < 0 {
			continue
		}
		for _, b := range next[a] {
			duration := g.Nodes[b].End - g.Nodes[b].Start
			if duration > math.MaxInt64-distance[a] {
				return r, fmt.Errorf("dependency: path duration overflow")
			}
			d := distance[a] + duration
			if d > distance[b] {
				distance[b] = d
				previous[b] = a
			}
		}
	}
	if distance[end] < 0 {
		r.Reasons = append(r.Reasons, "delivery endpoint is disconnected")
		return r, nil
	}
	r.State = "complete"
	r.WorkNS = distance[end]
	for i := end; i >= 0; i = previous[i] {
		r.CriticalPath = append(r.CriticalPath, g.Nodes[i].ID)
	}
	for i, j := 0, len(r.CriticalPath)-1; i < j; i, j = i+1, j-1 {
		r.CriticalPath[i], r.CriticalPath[j] = r.CriticalPath[j], r.CriticalPath[i]
	}
	return r, nil
}
