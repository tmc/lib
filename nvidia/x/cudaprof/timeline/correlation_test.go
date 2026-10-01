package timeline

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
)

func TestCorrelations(t *testing.T) {
	for _, tt := range []struct {
		name      string
		edit      func(*Records)
		api, span string
	}{
		{"nested", func(r *Records) {}, "matched", "matched"},
		{"other generation", func(r *Records) { r.Calls[0].Journal = "other" }, "missing", "missing"},
		{"reused ID", func(r *Records) { r.Calls = append(r.Calls, r.Calls[0]) }, "ambiguous", "unsupported"},
		{"zero ID", func(r *Records) { r.Activities[0].Correlation = 0 }, "missing", "missing"},
		{"missing identity", func(r *Records) { r.Activities[0].Journal = "" }, "missing", "missing"},
		{"graph", func(r *Records) { r.Activities[0].GraphID = 1 }, "unsupported", "unsupported"},
		{"missing thread", func(r *Records) { r.Calls[0].Thread = 0; r.Ranges[0].Thread = 0; r.Ranges[1].Thread = 0 }, "matched", "unsupported"},
		{"missing process", func(r *Records) { r.Calls[0].Process = 0; r.Ranges[0].Process = 0; r.Ranges[1].Process = 0 }, "matched", "unsupported"},
		{"partial crossing", func(r *Records) { r.Ranges[0].Start = 26; r.Ranges[0].End = 40 }, "matched", "ambiguous"},
		{"other thread", func(r *Records) { r.Calls[0].Thread = 9 }, "matched", "missing"},
		{"other process", func(r *Records) { r.Calls[0].Process = 9 }, "matched", "missing"},
		{"crossing ranges", func(r *Records) { r.Ranges[0].Start = 15; r.Ranges[0].End = 40; r.Ranges[1].Start = 10 }, "matched", "ambiguous"},
		{"equal ranges", func(r *Records) { r.Ranges[1] = r.Ranges[0] }, "matched", "ambiguous"},
		{"async execution", func(r *Records) { r.Activities[0].Start = 100; r.Activities[0].End = 200 }, "matched", "matched"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := Records{Activities: []Activity{{Journal: "j", Correlation: 7, Subtype: "kernel"}}, Calls: []Call{{Journal: "j", Domain: "driver", Process: 1, Thread: 2, Correlation: 7, Start: 20, End: 25}}, Ranges: []Range{{Journal: "j", Process: 1, Thread: 2, Start: 0, End: 50}, {Journal: "j", Process: 1, Thread: 2, Start: 10, End: 30}}}
			tt.edit(&r)
			got := Correlations(r, "driver")[0]
			if got.APIStatus != tt.api || got.RangeStatus != tt.span {
				t.Fatalf("got %+v, want %s/%s", got, tt.api, tt.span)
			}
			if tt.name == "nested" && got.Range != 1 {
				t.Fatalf("range=%d", got.Range)
			}
		})
	}
}

func ExampleCorrelations() {
	r := Records{Activities: []Activity{{Journal: "capture.journal", Correlation: 7}}, Calls: []Call{{Journal: "capture.journal", Process: 1, Thread: 2, Domain: "driver", Correlation: 7}}}
	m := Correlations(r, "driver")[0]
	fmt.Println(m.APIStatus, m.Call, m.RangeStatus)
	// Output: matched 0 missing
}

func TestCorrelationMetadataConflict(t *testing.T) {
	r := Records{
		Activities: []Activity{{Journal: "j", Correlation: 7}},
		Calls:      []Call{{Journal: "j", Process: 1, Thread: 2, Domain: "driver", Correlation: 7, Start: 20, End: 25}},
		Ranges: []Range{
			{Journal: "j", Process: 1, Thread: 2, Name: `cudaprof:v1 {"phase":"decode","token":1}`, Start: 1, End: 50},
			{Journal: "j", Process: 1, Thread: 2, Name: `cudaprof:v1 {"phase":"decode","token":2}`, Start: 10, End: 30},
		},
	}
	got := Correlations(r, "driver")[0]
	if got.MetadataStatus != "ambiguous" || got.Metadata != nil {
		t.Fatalf("conflicting metadata: %+v", got)
	}
	r.Ranges[1].Name = `cudaprof:v1 {"phase":"decode","operation":"gemv"}`
	got = Correlations(r, "driver")[0]
	if got.MetadataStatus != "matched" || got.Metadata.Token != nil {
		t.Fatalf("metadata inherited optional token: %+v", got)
	}
}

func TestMetadataLargeIntegerConflict(t *testing.T) {
	a, b := int64(1<<53), int64(1<<53)+1
	x, y := int(a), int(b)
	if int64(x) != a || int64(y) != b {
		t.Skip("requires 64-bit ints")
	}
	if !conflictingMetadata(marker.Metadata{Token: &x}, marker.Metadata{Token: &y}) {
		t.Fatal("lost integer precision")
	}
}

func TestCorrelationCrossingAncestor(t *testing.T) {
	name := `cudaprof:v1 {"phase":"decode","token":0}`
	r := Records{
		Activities: []Activity{{Journal: "j", Correlation: 1}},
		Calls:      []Call{{Journal: "j", Domain: "driver", Process: 1, Thread: 2, Correlation: 1, Start: 21, End: 29}},
		Ranges: []Range{
			{Journal: "j", Process: 1, Thread: 2, Name: name, Start: 20, End: 30},
			{Journal: "j", Process: 1, Thread: 2, Name: name, Start: 10, End: 40},
			{Journal: "j", Process: 1, Thread: 2, Name: name, Start: 30, End: 60},
		},
	}
	got := Correlations(r, "driver")[0]
	if got.APIStatus != "matched" || got.RangeStatus != "ambiguous" || got.MetadataStatus != "ambiguous" || got.Range != -1 || got.Metadata != nil {
		t.Fatalf("crossing ancestor qualified nested context: %+v", got)
	}
}

func TestGraphReplayCorrelation(t *testing.T) {
	for _, tt := range []struct {
		name, domain       string
		callback           uint32
		duplicate          bool
		correlation        uint32
		status, provenance string
	}{
		{"driver replay", "driver", 514, false, 7, "matched", "graph_replay"},
		{"runtime replay", "runtime", 311, false, 7, "matched", "graph_replay"},
		{"construction", "driver", 502, false, 7, "unsupported", "graph_construction"},
		{"absent launch", "driver", 514, false, 9, "missing", "unresolved"},
		{"reused launch correlation", "driver", 514, true, 7, "ambiguous", "unresolved"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := Records{
				Activities: []Activity{{Journal: "j", Subtype: "kernel", GraphID: 1, GraphNodeID: 1<<54 + 1, Context: 2, Correlation: tt.correlation, Start: 100, End: 200}},
				Calls:      []Call{{Journal: "j", Domain: tt.domain, CallbackID: tt.callback, Process: 1, Thread: 2, Correlation: 7, Start: 20, End: 25}},
				Ranges:     []Range{{Journal: "j", Process: 1, Thread: 2, Start: 10, End: 30, Name: `cudaprof:v1 {"phase":"decode","token":3,"operation":"gemv","dtype":"bf16","cache_length":128}`}},
				Graphs:     []Activity{{Journal: "j", Subtype: "graph_trace", GraphID: 1, Context: 2, Correlation: 7, Start: 90, End: 210}},
			}
			if tt.duplicate {
				r.Calls = append(r.Calls, r.Calls[0])
			}
			got := Correlations(r, tt.domain)[0]
			if got.APIStatus != tt.status || got.Provenance != tt.provenance {
				t.Fatalf("got %+v", got)
			}
			if tt.status == "matched" && (got.MetadataStatus != "matched" || *got.Metadata.Token != 3) {
				t.Fatalf("replay context missing: %+v", got)
			}
		})
	}
}

func TestDecodeGraphIdentity(t *testing.T) {
	const node = uint64(1<<54 + 1)
	b := journaltest.New().Append(wire.KindActivity, map[string]any{"subtype": "kernel", "valid": true, "start": 10, "end": 20, "graph_id": 5, "graph_node_id": node, "context": 3, "runtime_correlation": 9}).Finalize()
	r, err := Decode(map[string][]byte{"j": b}, MaxRecords, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	a := r.Activities[0]
	if a.GraphID != 5 || a.GraphNodeID != node || a.Context != 3 || a.RuntimeCorrelation != 9 {
		t.Fatalf("lost graph identity: %+v", a)
	}
}

func TestRequestMetadataConflict(t *testing.T) {
	for _, tt := range []struct {
		name string
		a, b marker.Metadata
	}{
		{"request", marker.Metadata{RequestID: "a"}, marker.Metadata{RequestID: "b"}},
		{"role", marker.Metadata{Role: "warmup"}, marker.Metadata{Role: "measured"}},
		{"boundary", marker.Metadata{Boundary: "completion"}, marker.Metadata{Boundary: "consumer_delivery"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if !conflictingMetadata(tt.a, tt.b) {
				t.Fatal("conflict ignored")
			}
		})
	}
}
