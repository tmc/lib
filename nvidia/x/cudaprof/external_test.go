package cudaprof_test

import (
	"fmt"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestLabelExternal(t *testing.T) {
	a := cuptiactivity.ScopedRecord{Session: "session", Run: "run", Process: "generation", Record: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7}}
	l := a
	l.Record = cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalID: 1<<63 + 1}
	for _, tt := range []struct {
		name      string
		indices   []int
		change    func(*profile.Profile, []cuptiactivity.ScopedRecord)
		wantError bool
	}{
		{"valid", []int{0}, func(*profile.Profile, []cuptiactivity.ScopedRecord) {}, false},
		{"negative index", []int{-1}, func(*profile.Profile, []cuptiactivity.ScopedRecord) {}, true},
		{"link not activity", []int{1}, func(*profile.Profile, []cuptiactivity.ScopedRecord) {}, true},
		{"missing mapping", nil, func(*profile.Profile, []cuptiactivity.ScopedRecord) {}, true},
		{"missing sample", nil, func(p *profile.Profile, _ []cuptiactivity.ScopedRecord) { p.Sample = nil }, true},
		{"duplicate index", []int{0, 0}, func(p *profile.Profile, _ []cuptiactivity.ScopedRecord) { p.Sample = append(p.Sample, p.Sample[0]) }, true},
		{"existing label", []int{0}, func(p *profile.Profile, _ []cuptiactivity.ScopedRecord) {
			p.Sample[0].Label["cupti.external.id"] = []string{"9"}
		}, true},
		{"existing numeric label", []int{0}, func(p *profile.Profile, _ []cuptiactivity.ScopedRecord) {
			p.Sample[0].NumLabel = map[string][]int64{"cupti.external.id": {9}}
		}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "duration", Unit: "nanoseconds"}}, Sample: []*profile.Sample{{Value: []int64{17}, Label: map[string][]string{"device": {"0"}}}}}
			records := []cuptiactivity.ScopedRecord{a, l}
			tt.change(p, records)
			got, err := cudaprof.LabelExternal(p, records, tt.indices, cupti.ExternalCorrelationKindCustom0)
			if (err != nil) != tt.wantError {
				t.Fatalf("err=%v", err)
			}
			if tt.wantError {
				if got != nil {
					t.Fatal("returned partial profile")
				}
				return
			}
			if got.Sample[0].Value[0] != 17 || got.Sample[0].Label["cupti.external.id"][0] != "9223372036854775809" || got.Sample[0].Label["device"][0] != "0" {
				t.Fatal("lost values/labels")
			}
			if len(p.Sample[0].Label) != 1 {
				t.Fatal("mutated input")
			}
			got.Sample[0].Value[0] = 1
			if p.Sample[0].Value[0] != 17 {
				t.Fatal("copied profile aliases input")
			}
		})
	}
}

func ExampleLabelExternal() {
	p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "duration", Unit: "nanoseconds"}}, Sample: []*profile.Sample{{Value: []int64{10}}}}
	r := cuptiactivity.ScopedRecord{Session: "capture", Run: "live", Process: "generation-1", Record: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel}}
	labeled, err := cudaprof.LabelExternal(p, []cuptiactivity.ScopedRecord{r}, []int{0}, cupti.ExternalCorrelationKindCustom0)
	fmt.Println(labeled.Sample[0].Label["cupti.external.status"][0], err)
	// Output: not_observed <nil>
}

func TestLabelExternalUnresolved(t *testing.T) {
	for _, ambiguous := range []bool{false, true} {
		p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "duration", Unit: "nanoseconds"}}, Sample: []*profile.Sample{{Value: []int64{10}}}}
		a := cuptiactivity.ScopedRecord{Session: "s", Run: "run", Process: "generation", Record: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7}}
		records := []cuptiactivity.ScopedRecord{a}
		want := "not_observed"
		if ambiguous {
			want = "ambiguous"
			for _, id := range []uint64{0, 9} {
				l := a
				l.Record = cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalID: id}
				records = append(records, l)
			}
		}
		got, err := cudaprof.LabelExternal(p, records, []int{0}, cupti.ExternalCorrelationKindCustom0)
		if err != nil || got.Sample[0].Label["cupti.external.status"][0] != want || len(got.Sample[0].Label["cupti.external.id"]) != 0 {
			t.Fatalf("profile=%v err=%v", got, err)
		}
	}
}
