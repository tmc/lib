package dependency

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func dependencyFixture() timeline.Records {
	r := timeline.Records{Activities: []timeline.Activity{{Journal: "j", Sequence: 100, Subtype: "kernel", Correlation: 4, Start: 30, End: 40}, {Journal: "j", Sequence: 101, Subtype: "kernel", Correlation: 7, Start: 50, End: 60}}}
	success := true
	for i, e := range []timeline.DependencyEvent{{Kind: "create_stream", Stream: 1, Flags: 1}, {Kind: "create_stream", Stream: 2, Flags: 1}, {Kind: "create_event", Event: 3}, {Kind: "launch", Stream: 1}, {Kind: "event_record", Stream: 1, Event: 3}, {Kind: "event_wait", Stream: 2, Event: 3}, {Kind: "launch", Stream: 2}, {Kind: "stream_sync", Stream: 2, End: 70, Start: 70}} {
		e.Journal = "j"
		e.Domain = "driver"
		e.Correlation = uint32(i + 1)
		e.TimestampDomain = "monotonic"
		e.CallbackID = uint32(i + 1)
		e.Context = 1
		e.Site = "enter"
		e.Ordinal = uint64(i*2 + 1)
		e.Sequence = e.Ordinal
		r.Dependencies = append(r.Dependencies, e)
		e.Site = "exit"
		e.Ordinal++
		e.Sequence = e.Ordinal
		e.Success = &success
		r.Dependencies = append(r.Dependencies, e)
	}
	return r
}

func ExampleObserve() {
	graph := Observe(timeline.Records{Activities: []timeline.Activity{{Journal: "j", Start: 1, End: 2}}})
	report, _ := Analyze(graph)
	fmt.Println(report.State, len(graph.Nodes), len(graph.Edges))
	// Output: partial 1 0
}

func TestObserveExplicitEventWait(t *testing.T) {
	g := Observe(dependencyFixture())
	if len(g.Edges) != 2 || g.Edges[0].Kind != "event_wait" || g.Edges[0].Record != "j:record:12" || g.Edges[1].Kind != "host_materialization" {
		t.Fatalf("edges=%+v", g.Edges)
	}
	if _, err := Analyze(g); err != nil {
		t.Fatal(err)
	}
	if g.Complete {
		t.Fatal("partial collector claimed completeness")
	}
}

func TestObserveRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*timeline.Records)
		reason string
	}{
		{"missing exit", func(r *timeline.Records) { r.Dependencies = r.Dependencies[:len(r.Dependencies)-1] }, "entry or exit"},
		{"clock", func(r *timeline.Records) { r.Dependencies[0].TimestampDomain = "unknown" }, "unified clock"},
		{"blocking stream", func(r *timeline.Records) { r.Dependencies[1].Flags = 0 }, "blocking"},
		{"unknown event", func(r *timeline.Records) { r.Dependencies[10].Event = 4; r.Dependencies[11].Event = 4 }, "unique completed record"},
		{"concurrent submit", func(r *timeline.Records) {
			r.Dependencies[7].Ordinal = 10
			r.Dependencies[8].Ordinal = 8
			r.Dependencies[9].Ordinal = 9
		}, "ambiguous stream order"},
		{"missing activity", func(r *timeline.Records) { r.Activities = r.Activities[:1] }, "no retained device activity"},
		{"overlap", func(r *timeline.Records) { r.Activities[1].Start = 35 }, "ordered common-clock"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := dependencyFixture()
			tc.change(&r)
			g := Observe(r)
			if !strings.Contains(strings.Join(g.Missing, " "), tc.reason) {
				t.Fatalf("missing=%v, want %q", g.Missing, tc.reason)
			}
			if _, err := Analyze(g); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestObserveExplicitGraphReplay(t *testing.T) {
	success := true
	r := timeline.Records{Activities: []timeline.Activity{{Journal: "j", Subtype: "kernel", Correlation: 5, GraphNodeID: 110, Start: 10, End: 20}, {Journal: "j", Subtype: "kernel", Correlation: 5, GraphNodeID: 120, Start: 30, End: 40}}, GraphLifecycle: []timeline.GraphEvent{
		{Journal: "j", Domain: "driver", Kind: "create", Graph: 1, Correlation: 1, Success: &success},
		{Journal: "j", Domain: "driver", Kind: "update_graph", Graph: 1, Correlation: 2, Success: &success},
		{Journal: "j", Domain: "driver", Kind: "update_graph", Graph: 1, Correlation: 3, Success: &success},
		{Journal: "j", Domain: "driver", Kind: "instantiate", Graph: 1, Executable: 2, Correlation: 4, Success: &success},
		{Journal: "j", Domain: "resource", Kind: "resource", NodeID: 110, SourceNodeID: 11},
		{Journal: "j", Domain: "resource", Kind: "resource", NodeID: 120, SourceNodeID: 12},
		{Journal: "j", Domain: "driver", Kind: "replay", Executable: 2, Correlation: 5, Success: &success},
	}}
	for i, e := range []timeline.DependencyEvent{{Kind: "create_stream", Stream: 1, Flags: 1, Correlation: 10}, {Kind: "graph_node", Graph: 1, Node: 11, Correlation: 2}, {Kind: "graph_node", Graph: 1, Node: 12, Dependencies: []uint64{11}, Correlation: 3}, {Kind: "graph_launch", Stream: 1, Executable: 2, Correlation: 5}, {Kind: "stream_sync", Stream: 1, Correlation: 6, Start: 50, End: 50}} {
		e.Journal = "j"
		e.Domain = "driver"
		e.TimestampDomain = "monotonic"
		e.CallbackID = uint32(i + 1)
		e.Site = "enter"
		e.Ordinal = uint64(2*i + 1)
		e.Sequence = e.Ordinal
		r.Dependencies = append(r.Dependencies, e)
		e.Site = "exit"
		e.Ordinal++
		e.Sequence = e.Ordinal
		e.Success = &success
		r.Dependencies = append(r.Dependencies, e)
	}
	g := Observe(r)
	if len(g.Edges) != 2 || g.Edges[0].Kind != "graph_edge" || g.Edges[0].Record != "j:record:6" {
		t.Fatalf("edges=%+v missing=%v", g.Edges, g.Missing)
	}
	if _, err := Analyze(g); err != nil {
		t.Fatal(err)
	}
	r.GraphLifecycle = r.GraphLifecycle[:4]
	g = Observe(r)
	if !strings.Contains(strings.Join(g.Missing, " "), "complete explicit node") {
		t.Fatalf("missing=%v", g.Missing)
	}
}

func controlledDeliveryFixture() timeline.Records {
	r := dependencyFixture()
	r.Ranges = []timeline.Range{{Journal: "j", Process: 1, Thread: 2, Name: `cudaprof:v1 {"phase":"decode","token":0,"operation":"token"}`, Start: 1, End: 72}, {Journal: "j", Process: 1, Thread: 2, Name: `cudaprof:v1 {"phase":"decode","token":0,"materialization":true}`, Start: 65, End: 71}}
	r.Markers = []timeline.Marker{{Journal: "j", Sequence: 200, Process: 1, Thread: 2, Time: 1, Flags: timeline.MarkerStart, Name: r.Ranges[0].Name}, {Journal: "j", Sequence: 202, Process: 1, Thread: 2, Time: 73, Flags: timeline.MarkerInstantaneous, Name: `cudaprof:v1 {"phase":"decode","token":0,"event":"delivered"}`}}
	r.Calls = []timeline.Call{{Journal: "j", Domain: "driver", Process: 1, Thread: 2, Correlation: 4, Start: 5, End: 6}, {Journal: "j", Domain: "driver", Process: 1, Thread: 2, Correlation: 7, Start: 7, End: 8}, {Journal: "j", Domain: "driver", Process: 1, Thread: 2, Correlation: 8, Start: 66, End: 70}}
	return r
}

func ExampleDelivery() {
	selection := Selection{Journal: "j", Process: 1, Thread: 2, Token: 0, ExpectedActivities: 2}
	graph := Delivery(timeline.Records{}, selection)
	fmt.Println(graph.Complete)
	// Output: false
}

func ExampleSelection() {
	selection := Selection{Journal: "process.journal", Process: 1, Thread: 2, Token: 0, ExpectedActivities: 3}
	fmt.Println(selection.Token, selection.ExpectedActivities)
	// Output: 0 3
}

func TestControlledDelivery(t *testing.T) {
	selection := Selection{Journal: "j", Process: 1, Thread: 2, Token: 0, ExpectedActivities: 2}
	g := Delivery(controlledDeliveryFixture(), selection)
	report, err := Analyze(g)
	if err != nil || report.State != "complete" || report.WorkNS != 20 {
		t.Fatalf("report=%+v err=%v", report, err)
	}
	for _, tc := range []struct {
		name   string
		change func(*timeline.Records, *Selection)
	}{
		{"wrong independent count", func(_ *timeline.Records, s *Selection) { s.ExpectedActivities = 3 }},
		{"different delivery thread", func(r *timeline.Records, _ *Selection) { r.Markers[1].Thread = 3 }},
		{"no successful materialization", func(r *timeline.Records, _ *Selection) { r.Dependencies = r.Dependencies[:14] }},
		{"unjoined independent work", func(r *timeline.Records, _ *Selection) { r.Dependencies[10].Event = 4; r.Dependencies[11].Event = 4 }},
		{"no API thread evidence", func(r *timeline.Records, _ *Selection) { r.Calls = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := controlledDeliveryFixture()
			s := selection
			tc.change(&r, &s)
			g := Delivery(r, s)
			if g.Complete {
				t.Fatalf("incorrectly complete: %+v", g)
			}
			report, err := Analyze(g)
			if err != nil || report.State != "partial" {
				t.Fatalf("report=%+v error=%v", report, err)
			}
		})
	}
}
