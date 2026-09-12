package cuptiactivity_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func coverageRecord(kind cupti.ActivityKind, correlation uint32, duration uint64) cuptiactivity.ScopedRecord {
	return cuptiactivity.ScopedRecord{Session: "s", Run: "run", Process: "generation", Record: cuptiactivity.Record{Kind: kind, CorrelationID: correlation, Start: 1, End: 1 + duration, Complete: true}}
}

func TestExternalCoverage(t *testing.T) {
	k := cupti.ActivityKindConcurrentKernel
	link := coverageRecord(cupti.ActivityKindExternalCorrelation, 1, 0)
	link.Record.ExternalKind = cupti.ExternalCorrelationKindCustom0
	link.Record.ExternalID = 7
	records := []cuptiactivity.ScopedRecord{coverageRecord(k, 1, 1), coverageRecord(k, 2, 99), link}
	got, err := cuptiactivity.ExternalCoverage(records, cupti.ExternalCorrelationKindCustom0)
	if err != nil || len(got) != 1 {
		t.Fatalf("coverage=%v err=%v", got, err)
	}
	g := got[0]
	if g.Total.Records != 2 || g.Total.Duration != 100 || g.Matched.Records != 1 || g.Matched.Duration != 1 || g.NotObserved.Duration != 99 {
		t.Fatalf("coverage=%+v", g)
	}
	// Equal counts do not imply equal duration; the unmatched launch dominates.
	other := link
	other.Record.ExternalID = 8
	got, err = cuptiactivity.ExternalCoverage(append(records, other), cupti.ExternalCorrelationKindCustom0)
	if err != nil || got[0].Ambiguous.Duration != 1 || got[0].Matched.Records != 0 {
		t.Fatalf("ambiguous=%v err=%v", got, err)
	}
	for _, change := range []func(*cuptiactivity.ScopedRecord){
		func(r *cuptiactivity.ScopedRecord) { r.Session = "other" },
		func(r *cuptiactivity.ScopedRecord) { r.Run = "other" },
		func(r *cuptiactivity.ScopedRecord) { r.Process = "other-generation" },
		func(r *cuptiactivity.ScopedRecord) { r.Record.DeviceID = 1 },
		func(r *cuptiactivity.ScopedRecord) { r.Record.Kind = cupti.ActivityKindMemcpy },
	} {
		a, b := records[0], records[1]
		change(&b)
		forward, err := cuptiactivity.ExternalCoverage([]cuptiactivity.ScopedRecord{a, b, link}, cupti.ExternalCorrelationKindCustom0)
		if err != nil || len(forward) != 2 {
			t.Fatalf("scopes=%v err=%v", forward, err)
		}
		reverse, err := cuptiactivity.ExternalCoverage([]cuptiactivity.ScopedRecord{link, b, a}, cupti.ExternalCorrelationKindCustom0)
		if err != nil || !reflect.DeepEqual(forward, reverse) {
			t.Fatal("nondeterministic coverage")
		}
	}
}

func TestExternalCoverageIntervals(t *testing.T) {
	for _, tt := range []struct {
		name       string
		start, end uint64
		complete   bool
		valid      bool
	}{
		{"zero duration", 10, 10, true, true},
		{"explicit invalid", 10, 20, false, false},
		{"missing start", 0, 20, true, false},
		{"reversed", 20, 10, true, false},
		{"uint64", 1, 1<<63 + 1, true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := coverageRecord(cupti.ActivityKindConcurrentKernel, 1, 0)
			r.Record.Start, r.Record.End, r.Record.Complete = tt.start, tt.end, tt.complete
			got, err := cuptiactivity.ExternalCoverage([]cuptiactivity.ScopedRecord{r}, cupti.ExternalCorrelationKindCustom0)
			if err != nil {
				t.Fatal(err)
			}
			x := got[0].Total
			if (x.TimedRecords == 1) != tt.valid || (x.InvalidIntervals == 1) == tt.valid {
				t.Fatalf("totals=%+v", x)
			}
			if tt.valid && x.Duration != tt.end-tt.start {
				t.Fatal("lost native duration")
			}
		})
	}
	r := coverageRecord(cupti.ActivityKindConcurrentKernel, 1, 1<<63)
	if got, err := cuptiactivity.ExternalCoverage([]cuptiactivity.ScopedRecord{r, r}, cupti.ExternalCorrelationKindCustom0); err == nil || got != nil {
		t.Fatal("overflow accepted")
	}
	api := coverageRecord(cupti.ActivityKindRuntime, 1, 100)
	if got, err := cuptiactivity.ExternalCoverage([]cuptiactivity.ScopedRecord{api}, cupti.ExternalCorrelationKindCustom0); err != nil || len(got) != 0 {
		t.Fatal("API added device duration")
	}
}

func ExampleExternalCoverage() {
	r := cuptiactivity.ScopedRecord{Session: "capture", Run: "live", Process: "generation-1", Record: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Start: 1, End: 11, Complete: true}}
	coverage, err := cuptiactivity.ExternalCoverage([]cuptiactivity.ScopedRecord{r}, cupti.ExternalCorrelationKindCustom0)
	fmt.Println(coverage[0].NotObserved.Duration, err)
	// Output: 10 <nil>
}

func ExampleActivityTotals() {
	t := cuptiactivity.ActivityTotals{Records: 3, TimedRecords: 2, InvalidIntervals: 1, Duration: 10}
	fmt.Println(t.Records, t.InvalidIntervals)
	// Output: 3 1
}

func ExampleAttributionCoverage() {
	c := cuptiactivity.AttributionCoverage{Process: "generation-1", DeviceID: 0}
	fmt.Println(c.Process, c.DeviceID)
	// Output: generation-1 0
}

func TestAPICoverage(t *testing.T) {
	api := coverageRecord(cupti.ActivityKindDriver, 7, 1000)
	kernel := coverageRecord(cupti.ActivityKindConcurrentKernel, 7, 10)
	copyRecord := coverageRecord(cupti.ActivityKindMemcpy, 8, 20)
	copyRecord.Record.RuntimeCorrelationID = 7
	otherGeneration := kernel
	otherGeneration.Process = "other-generation"
	records := []cuptiactivity.ScopedRecord{api, kernel, kernel, copyRecord, otherGeneration}
	for _, tt := range []struct {
		name    string
		kind    cupti.ActivityKind
		matched uint64
	}{
		{"driver", cupti.ActivityKindDriver, 20},
		{"runtime absent", cupti.ActivityKindRuntime, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cuptiactivity.APICoverage(records, tt.kind)
			if err != nil {
				t.Fatal(err)
			}
			var total, matched uint64
			for _, g := range got {
				total += g.Total.Duration
				matched += g.Matched.Duration
			}
			if total != 50 || matched != tt.matched {
				t.Fatalf("total=%d matched=%d", total, matched)
			}
		})
	}
	api.Record.Kind = cupti.ActivityKindRuntime
	got, err := cuptiactivity.APICoverage([]cuptiactivity.ScopedRecord{api, copyRecord}, cupti.ActivityKindRuntime)
	if err != nil || len(got) != 1 || got[0].Matched.Duration != 20 {
		t.Fatalf("runtime copy=%+v err=%v", got, err)
	}
	conflict := api
	conflict.Record.End++
	got, err = cuptiactivity.APICoverage([]cuptiactivity.ScopedRecord{api, conflict, copyRecord}, cupti.ActivityKindRuntime)
	if err != nil || got[0].Ambiguous.Duration != 20 || got[0].Matched.Records != 0 {
		t.Fatalf("conflict=%+v err=%v", got, err)
	}
	if got, err := cuptiactivity.APICoverage(records, cupti.ActivityKindMemcpy); err == nil || got != nil {
		t.Fatal("invalid domain accepted")
	}
}

func ExampleAPICoverage() {
	r := cuptiactivity.ScopedRecord{Session: "capture", Run: "live", Process: "generation-1", Record: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Start: 1, End: 11, Complete: true}}
	coverage, err := cuptiactivity.APICoverage([]cuptiactivity.ScopedRecord{r}, cupti.ActivityKindDriver)
	fmt.Println(coverage[0].NotObserved.Duration, err)
	// Output: 10 <nil>
}
