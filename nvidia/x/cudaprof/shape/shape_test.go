package shape

import (
	"strings"
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

func TestRequestComparison(t *testing.T) {
	for _, tt := range []struct {
		name, request, role, boundary string
		executions                    *int
		pass                          bool
	}{
		{"same", "r", "measured", "submission", intPointer(3), true},
		{"mapped request differs", "s", "measured", "submission", intPointer(3), true},
		{"role differs", "r", "warmup", "submission", intPointer(3), false},
		{"boundary differs", "r", "measured", "completion", intPointer(3), false},
		{"work differs", "r", "measured", "submission", intPointer(4), true},
		{"work missing", "r", "measured", "submission", nil, true},
		{"boundary missing", "r", "measured", "", intPointer(3), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			base, target := capture(0, "base", 2), capture(0, "target", 2)
			for i, r := range []*timeline.Records{&base, &target} {
				m, err := marker.ParseRange(r.Ranges[0].Name)
				if err != nil {
					t.Fatal(err)
				}
				m.Version = 2
				m.RequestID = "r"
				m.Role = "measured"
				m.Boundary = "submission"
				m.ExecutionUnit = "logical_forward"
				m.Executions = intPointer(3)
				if i == 1 {
					m.RequestID = tt.request
					m.Role = tt.role
					m.Boundary = tt.boundary
					m.Executions = tt.executions
				}
				r.Ranges[0].Name, err = marker.Encode(m)
				if err != nil {
					t.Fatal(err)
				}
			}
			got := CompareScoped(base, target, "driver", []RequestPair{{BaseJournal: "capture", BaseProcess: 1, BaseID: "r", TargetJournal: "capture", TargetProcess: 1, TargetID: tt.request, Role: "measured"}})
			if tt.name == "work differs" && !strings.Contains(got.WorkScope, "mismatch") {
				t.Fatalf("missing work diagnostic: %+v", got)
			}
			if Compare(base, target, "driver").Pass {
				t.Fatal("v2 matched without explicit mapping")
			}
			if got.Pass != tt.pass {
				t.Fatalf("Compare=%+v", got)
			}
		})
	}
}

func intPointer(v int) *int { return &v }

func TestRequestMappingAdmission(t *testing.T) {
	captureV2 := func(process uint32, correlation uint32) timeline.Records {
		r := capture(0, "kernel", 2)
		m, err := marker.ParseRange(r.Ranges[0].Name)
		if err != nil {
			t.Fatal(err)
		}
		m.Version = 2
		m.RequestID = "r"
		m.Role = "measured"
		m.Boundary = "submission"
		m.ExecutionUnit = "logical_forward"
		m.Executions = intPointer(2)
		r.Ranges[0].Name, err = marker.Encode(m)
		if err != nil {
			t.Fatal(err)
		}
		r.Calls[0].Process = process
		r.Ranges[0].Process = process
		r.Calls[0].Correlation = correlation
		r.Activities[0].Correlation = correlation
		return r
	}
	base, target := captureV2(1, 1), captureV2(3, 1)
	pair := RequestPair{BaseJournal: "capture", BaseProcess: 1, BaseID: "r", TargetJournal: "capture", TargetProcess: 3, TargetID: "r", Role: "measured"}
	t.Run("unused target", func(t *testing.T) {
		extra := pair
		extra.BaseProcess = 2
		extra.TargetProcess = 9
		b2 := captureV2(2, 2)
		b := base
		b.Calls = append(append([]timeline.Call(nil), b.Calls...), b2.Calls...)
		b.Ranges = append(append([]timeline.Range(nil), b.Ranges...), b2.Ranges...)
		b.Activities = append(append([]timeline.Activity(nil), b.Activities...), b2.Activities...)
		got := CompareScoped(b, target, "driver", []RequestPair{pair, extra})
		if got.Pass || len(got.Groups) != 1 || !strings.Contains(strings.Join(got.Failures, ";"), "mapping lacks observed") {
			t.Fatalf("unused mapping accepted: %+v", got)
		}
	})
	t.Run("same ids separate processes", func(t *testing.T) {
		b2, t2 := captureV2(2, 2), captureV2(4, 2)
		b, q := base, target
		b.Calls = append(append([]timeline.Call(nil), b.Calls...), b2.Calls...)
		b.Ranges = append(append([]timeline.Range(nil), b.Ranges...), b2.Ranges...)
		b.Activities = append(append([]timeline.Activity(nil), b.Activities...), b2.Activities...)
		q.Calls = append(append([]timeline.Call(nil), q.Calls...), t2.Calls...)
		q.Ranges = append(append([]timeline.Range(nil), q.Ranges...), t2.Ranges...)
		q.Activities = append(append([]timeline.Activity(nil), q.Activities...), t2.Activities...)
		other := pair
		other.BaseProcess = 2
		other.TargetProcess = 4
		got := CompareScoped(b, q, "driver", []RequestPair{pair, other})
		if !got.Pass || len(got.Groups) != 2 {
			t.Fatalf("process scope lost: %+v", got)
		}
	})
	t.Run("no matched scopes", func(t *testing.T) {
		q := target
		q.Activities = append([]timeline.Activity(nil), q.Activities...)
		q.Activities[0].Grid[0] = 99
		got := CompareScoped(base, q, "driver", []RequestPair{pair})
		if len(got.Groups) != 0 || strings.Contains(got.WorkScope, "counts agree") {
			t.Fatalf("empty work qualified: %+v", got)
		}
	})
}
