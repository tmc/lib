package overview

import (
	"regexp"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func markedRecords() timeline.Records {
	return timeline.Records{
		Activities: []timeline.Activity{
			{Journal: "j", Subtype: "kernel", Name: "raw<float>", Device: 0, Correlation: 1, Start: 20, End: 50},
			{Journal: "j", Subtype: "kernel", Name: "raw<float>", Device: 0, Stream: 2, Correlation: 2, Start: 30, End: 60},
		},
		Calls: []timeline.Call{
			{Journal: "j", Process: 1, Thread: 2, Domain: "driver", Correlation: 1, Start: 12, End: 15},
			{Journal: "j", Process: 1, Thread: 3, Domain: "driver", Correlation: 2, Start: 15, End: 18},
		},
		Ranges: []timeline.Range{
			{Journal: "j", Process: 1, Thread: 2, Start: 10, End: 25, Name: `cudaprof:v1 {"phase":"decode","token":0,"layer":0,"operation":"gemv","materialization":true}`},
			{Journal: "j", Process: 1, Thread: 3, Start: 12, End: 30, Name: `cudaprof:v1 {"phase":"decode","token":1,"layer":0,"operation":"gemv","lookahead":true}`},
		},
		Markers: []timeline.Marker{{Journal: "j", Process: 1, Thread: 2, Time: 55, Flags: timeline.MarkerInstantaneous, Name: `cudaprof:v1 {"phase":"decode","token":0,"event":"delivered"}`}},
	}
}
func TestObservedTokensAndOverlap(t *testing.T) {
	r := Analyze(markedRecords(), "driver", nil)
	if len(r.Tokens) != 2 || len(r.Unmatched) != 0 {
		t.Fatalf("report=%+v", r)
	}
	if r.Hierarchy.Metrics.KernelSumNanos != 60 || len(r.Hierarchy.Metrics.GPUUnion[0]) != 1 || r.Hierarchy.Metrics.GPUUnion[0][0].End-r.Hierarchy.Metrics.GPUUnion[0][0].Start != 40 {
		t.Fatalf("metrics=%+v", r.Hierarchy.Metrics)
	}
	token := r.Tokens[0]
	if token.Delivery == nil || *token.Delivery != 55 || token.DeliveryLatency == nil || *token.DeliveryLatency != 45 || len(token.Crossing) != 1 || len(token.Materialization) != 1 {
		t.Fatalf("token=%+v", token)
	}
	if r.Tokens[1].Delivery != nil {
		t.Fatal("lookahead invented delivery")
	}
}
func TestMissingAndInferred(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{
		{Journal: "j", Subtype: "kernel", Name: "arg_reduce", Start: 10, End: 11},
		{Journal: "j", Subtype: "kernel", Name: "crossing", Start: 15, End: 25},
		{Journal: "j", Subtype: "kernel", Name: "arg_reduce", Start: 20, End: 21},
	}}
	r := Analyze(records, "driver", regexp.MustCompile("arg_reduce"))
	if len(r.Tokens) != 0 || len(r.Unmatched) != 3 || len(r.Inferred) != 1 || len(r.Inferred[0].Crossing) != 1 || r.Inferred[0].Delivery != nil {
		t.Fatalf("report=%+v", r)
	}
}
func TestDeliveryConflictsAndEndReason(t *testing.T) {
	records := markedRecords()
	event := records.Markers[0]
	event.Time = 56
	records.Markers = append(records.Markers, event)
	event.Name = `cudaprof:v1 {"phase":"decode","token":0,"event":"eos"}`
	records.Markers = append(records.Markers, event)
	r := Analyze(records, "driver", nil)
	if r.Tokens[0].Delivery != nil || r.Tokens[0].EndReason != "eos" {
		t.Fatalf("token=%+v", r.Tokens[0])
	}
	event.Name = `cudaprof:v1 {"phase":"decode","token":0,"event":"token_limit"}`
	records.Markers = append(records.Markers, event)
	r = Analyze(records, "driver", nil)
	if r.Tokens[0].EndReason != "ambiguous" {
		t.Fatalf("token=%+v", r.Tokens[0])
	}
}
func TestCrossingRangesDoNotSupplyTokenContext(t *testing.T) {
	records := markedRecords()
	records.Ranges = append(records.Ranges, timeline.Range{Journal: "j", Process: 1, Thread: 2, Start: 20, End: 40, Name: `cudaprof:v1 {"phase":"decode","token":2}`})
	records.Markers = nil
	r := Analyze(records, "driver", nil)
	if len(r.Tokens) != 1 || r.Tokens[0].Index != 1 || len(r.Unmatched) != 1 {
		t.Fatalf("report=%+v", r)
	}
}
func TestSeparateJournalUnions(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "a", Subtype: "kernel", Start: 10, End: 20}, {Journal: "b", Subtype: "kernel", Start: 10, End: 20}}}
	r := Analyze(records, "driver", nil)
	if r.Hierarchy.Metrics.GPUSpan != nil || len(r.Hierarchy.Metrics.GPUUnion[0]) != 2 {
		t.Fatalf("cross-journal clock invented: %+v", r.Hierarchy.Metrics)
	}
}

func TestMultipleDeviceSpanIsUnavailable(t *testing.T) {
	records := markedRecords()
	records.Activities[1].Device = 1
	r := Analyze(records, "driver", nil)
	if r.Hierarchy.Metrics.GPUSpan != nil || len(r.Hierarchy.Metrics.GPUUnion) != 2 {
		t.Fatalf("merged device span: %+v", r.Hierarchy.Metrics)
	}
}
func TestHostOnlyHierarchyAndUniqueMetrics(t *testing.T) {
	records := markedRecords()
	records.Activities = nil
	records.Calls = nil
	r := Analyze(records, "driver", nil)
	if len(r.Tokens) != 2 || len(r.Hierarchy.Children) == 0 || r.Hierarchy.Metrics.HostUnionNanos != 20 {
		t.Fatalf("missing host-only scope: %+v", r)
	}
	records = markedRecords()
	r = Analyze(records, "driver", nil)
	if r.Hierarchy.Metrics.HostUnionNanos != 20 || r.Tokens[0].Metrics.HostUnionNanos != 15 {
		t.Fatalf("repeated range references multiplied host duration: %+v", r.Hierarchy.Metrics)
	}
}
func TestInvalidDeliveryScopeDoesNotSupplyBoundary(t *testing.T) {
	for _, bad := range []timeline.Marker{
		{Process: 1, Thread: 2, Time: 55, Flags: timeline.MarkerInstantaneous},
		{Journal: "j", Process: 1, Thread: 2, Time: -1, Flags: timeline.MarkerInstantaneous},
		{Journal: "j", Process: 1, Thread: 2, Time: 55, Flags: timeline.MarkerInstantaneous | timeline.MarkerStart},
	} {
		records := markedRecords()
		bad.Name = records.Markers[0].Name
		records.Markers = []timeline.Marker{bad}
		r := Analyze(records, "driver", nil)
		for _, token := range r.Tokens {
			if token.Delivery != nil {
				t.Fatalf("accepted invalid boundary: %+v", bad)
			}
		}
	}
}
func TestActivityKindsAndAmbiguityPreserved(t *testing.T) {
	records := markedRecords()
	records.Activities[0].Subtype = "memcpy"
	records.Calls = append(records.Calls, records.Calls[0])
	r := Analyze(records, "driver", nil)
	var found bool
	var walk func(*Node)
	walk = func(n *Node) {
		if n.Kind == "memcpy" && n.Provenance == "ambiguous" {
			found = true
		}
		for _, c := range n.Children {
			walk(c)
		}
	}
	walk(r.Hierarchy)
	if !found {
		t.Fatal("transfer kind or ambiguous provenance lost")
	}
}
func TestInferenceDoesNotMixDevices(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "j", Device: 0, Subtype: "kernel", Name: "marker", Start: 1, End: 2}, {Journal: "j", Device: 1, Subtype: "kernel", Name: "marker", Start: 10, End: 11}}}
	if got := Analyze(records, "driver", regexp.MustCompile("marker")); len(got.Inferred) != 0 {
		t.Fatal("inferred interval spans unrelated device timestamps")
	}
}

func TestDeliveryRangeDoesNotSupplyActivityContext(t *testing.T) {
	records := markedRecords()
	records.Ranges[0].Name = `cudaprof:v1 {"phase":"decode","token":0,"event":"delivered"}`
	records.Markers = nil
	r := Analyze(records, "driver", nil)
	if len(r.Unmatched) != 1 || len(r.Tokens) != 1 || r.Tokens[0].Index != 1 {
		t.Fatalf("delivery range supplied operation context: %+v", r)
	}
}
