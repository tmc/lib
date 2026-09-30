package dependency

import "testing"

func TestAnalyze(t *testing.T) {
	nodes := []Node{{"start", "marker:0", "cupti", 0, 0}, {"left", "activity:0", "cupti", 0, 10}, {"right", "activity:1", "cupti", 0, 6}, {"wait", "event:0", "cupti", 10, 12}, {"deliver", "marker:1", "cupti", 12, 12}}
	edges := []Edge{{"start", "left", "stream_order", "stream:0"}, {"start", "right", "graph_edge", "graph:0"}, {"left", "wait", "event_wait", "event:0"}, {"right", "wait", "graph_edge", "graph:1"}, {"wait", "deliver", "host_materialization", "sync:0"}}
	for _, tt := range []struct {
		name    string
		graph   Graph
		state   string
		work    int64
		wantErr bool
	}{
		{"parallel event graph and materialization", Graph{nodes, edges, "start", "deliver", true, nil}, "complete", 12, false},
		{"missing records", Graph{nodes, edges, "start", "deliver", true, []string{"event activity disabled"}}, "partial", 0, false},
		{"unverified inventory", Graph{nodes, edges, "start", "deliver", false, nil}, "partial", 0, false},
		{"missing delivery", Graph{nodes, edges, "start", "missing", true, nil}, "partial", 0, false},
		{"timestamp proximity rejected", Graph{nodes, []Edge{{"left", "wait", "timestamp_adjacent", "activity:0"}}, "start", "deliver", true, nil}, "", 0, true},
		{"overlapping serial endpoints", Graph{nodes, []Edge{{"left", "right", "stream_order", "stream:0"}}, "left", "right", true, nil}, "", 0, true},
		{"no raw provenance", Graph{nodes, []Edge{{"start", "left", "stream_order", ""}}, "start", "left", true, nil}, "", 0, true},
		{"disconnected", Graph{nodes, nil, "start", "deliver", true, nil}, "partial", 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, e := Analyze(tt.graph)
			if (e != nil) != tt.wantErr {
				t.Fatalf("error=%v", e)
			}
			if e == nil && (r.State != tt.state || r.WorkNS != tt.work) {
				t.Fatalf("report=%+v", r)
			}
		})
	}
}

func TestAnalyzeCycle(t *testing.T) {
	_, err := Analyze(Graph{Nodes: []Node{{"a", "a", "clock", 0, 0}, {"b", "b", "clock", 0, 0}}, Edges: []Edge{{"a", "b", "graph_edge", "edge:0"}, {"b", "a", "graph_edge", "edge:1"}}})
	if err == nil {
		t.Fatal("accepted cycle")
	}
}
