package timeline

import (
	"fmt"
	"testing"

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
	r := Records{Activities: []Activity{{Journal: "capture.journal", Correlation: 7}}, Calls: []Call{{Journal: "capture.journal", Domain: "driver", Correlation: 7}}}
	m := Correlations(r, "driver")[0]
	fmt.Println(m.APIStatus, m.Call, m.RangeStatus)
	// Output: matched 0 missing
}

func TestCorrelationMetadataConflict(t *testing.T) {
	r := Records{
		Activities: []Activity{{Journal: "j", Correlation: 7}},
		Calls:      []Call{{Journal: "j", Domain: "driver", Correlation: 7, Start: 20, End: 25}},
		Ranges: []Range{
			{Journal: "j", Name: `cudaprof:v1 {"phase":"decode","token":1}`, Start: 1, End: 50},
			{Journal: "j", Name: `cudaprof:v1 {"phase":"decode","token":2}`, Start: 10, End: 30},
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
