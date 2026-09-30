package shape

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func capture(token int, symbol string, grid int) timeline.Records {
	dtype := "float32"
	cache := 32
	text, _ := marker.Encode(marker.Metadata{Phase: "decode", Token: &token, Operation: "attention", DType: &dtype, CacheLength: &cache})
	return timeline.Records{
		Activities: []timeline.Activity{{Journal: "capture", Name: symbol, Start: 10, End: 20, Correlation: 1, Grid: [3]int{grid, 1, 1}, Block: [3]int{32, 1, 1}}},
		Calls:      []timeline.Call{{Journal: "capture", Process: 1, Thread: 2, Domain: "driver", Correlation: 1, Start: 2, End: 3}},
		Ranges:     []timeline.Range{{Journal: "capture", Process: 1, Thread: 2, Name: text, Start: 1, End: 4}},
	}
}

func TestCompare(t *testing.T) {
	for _, tt := range []struct {
		name   string
		alter  func(*timeline.Records)
		pass   bool
		groups int
		reason string
	}{
		{"observed", func(r *timeline.Records) { r.Activities[0].Name = "other<float, 32>" }, true, 1, ""},
		{"missing metadata", func(r *timeline.Records) { r.Ranges = nil }, false, 0, "required evidence unavailable"},
		{"unread geometry", func(r *timeline.Records) { r.Activities[0].Grid = [3]int{} }, false, 0, "required evidence unavailable"},
		{"different geometry", func(r *timeline.Records) { r.Activities[0].Grid[0] = 3 }, false, 0, "no equivalent observed identity"},
		{"count mismatch", func(r *timeline.Records) { r.Activities = append(r.Activities, r.Activities[0]) }, false, 0, "launch counts differ"},
		{"ambiguous API", func(r *timeline.Records) { r.Calls = append(r.Calls, r.Calls[0]) }, false, 0, "required evidence unavailable"},
		{"opaque graph", func(r *timeline.Records) {
			r.Graphs = []timeline.Activity{{Journal: "capture", Subtype: "graph_trace", GraphID: 1, Start: 25, End: 50}}
		}, false, 1, ""},
		{"unpaired marker", func(r *timeline.Records) { r.UnpairedMarkers = 1 }, false, 1, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			base, target := capture(0, "base<float, 32>", 2), capture(0, "target<float, 32>", 2)
			tt.alter(&target)
			got := Compare(base, target, "driver")
			if got.Pass != tt.pass || len(got.Groups) != tt.groups {
				t.Fatalf("Compare = %+v", got)
			}
			if tt.reason != "" && (len(got.TargetUnmatched) == 0 || got.TargetUnmatched[0].Reason != tt.reason) {
				t.Fatalf("unmatched = %+v", got.TargetUnmatched)
			}
			if tt.pass && got.Groups[0].Base[0].RawSymbol != "base<float, 32>" {
				t.Fatal("raw symbol discarded")
			}
		})
	}
}

func TestAmbiguousRepeatedIdentity(t *testing.T) {
	base, target := capture(0, "a", 2), capture(0, "b", 2)
	base.Activities = append(base.Activities, base.Activities[0])
	target.Activities = append(target.Activities, target.Activities[0])
	got := Compare(base, target, "driver")
	if got.Pass || len(got.Groups) != 0 || len(got.BaseUnmatched) != 2 || got.BaseUnmatched[0].Reason != "ambiguous repeated identity" {
		t.Fatalf("Compare = %+v", got)
	}
}

func TestAlternatingDispatch(t *testing.T) {
	base, target := capture(0, "attention<float, 32>", 2), capture(0, "attention<float, 64>", 2)
	b2, t2 := capture(1, "attention<float, 64>", 4), capture(1, "attention<float, 32>", 4)
	for _, r := range []*timeline.Records{&b2, &t2} {
		r.Activities[0].Correlation = 2
		r.Calls[0].Correlation = 2
		r.Calls[0].Start = 6
		r.Calls[0].End = 7
		r.Ranges[0].Start = 5
		r.Ranges[0].End = 8
	}
	base.Activities = append(base.Activities, b2.Activities...)
	base.Calls = append(base.Calls, b2.Calls...)
	base.Ranges = append(base.Ranges, b2.Ranges...)
	target.Activities = append(target.Activities, t2.Activities...)
	target.Calls = append(target.Calls, t2.Calls...)
	target.Ranges = append(target.Ranges, t2.Ranges...)
	got := Compare(base, target, "driver")
	if !got.Pass || len(got.Groups) != 2 {
		t.Fatalf("Compare = %+v", got)
	}
	target.Activities[0].Grid, target.Activities[1].Grid = target.Activities[1].Grid, target.Activities[0].Grid
	if got := Compare(base, target, "driver"); got.Pass || len(got.Groups) != 0 {
		t.Fatalf("dispatch shapes conflated: %+v", got)
	}
}

func TestQualificationFailures(t *testing.T) {
	base, target := capture(0, "a", 2), capture(0, "b", 2)
	target.UnpairedMarkers = 1
	target.Graphs = []timeline.Activity{{Subtype: "graph_trace", GraphID: 1, Start: 25, End: 50}}
	got := Compare(base, target, "driver")
	if got.Pass || len(got.Failures) != 2 {
		t.Fatalf("receipt = %+v", got)
	}
	target.UnpairedMarkers = 0
	target.GraphNodes = 1
	if got := Compare(base, target, "driver"); got.Pass {
		t.Fatalf("global node count qualified unsupported graph identity: %+v", got)
	}
}
