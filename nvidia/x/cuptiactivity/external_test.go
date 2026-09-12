package cuptiactivity_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestMatchExternal(t *testing.T) {
	base := cuptiactivity.ScopedRecord{Session: "session", Run: "run", Process: "pid-generation"}
	link, activity := base, base
	link.Record = cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalID: 1<<63 + 5, CorrelationID: 42}
	activity.Record = cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 42}
	for _, tt := range []struct {
		name   string
		change func(*cuptiactivity.ScopedRecord)
		status string
	}{
		{"native link", func(*cuptiactivity.ScopedRecord) {}, "matched"},
		{"synchronization", func(r *cuptiactivity.ScopedRecord) { r.Record.Kind = cupti.ActivityKindSynchronization }, "matched"},
		{"memory", func(r *cuptiactivity.ScopedRecord) { r.Record.Kind = cupti.ActivityKindMemory2 }, "matched"},
		{"different session", func(r *cuptiactivity.ScopedRecord) { r.Session = "other" }, "not_observed"},
		{"different run", func(r *cuptiactivity.ScopedRecord) { r.Run = "other" }, "not_observed"},
		{"reused pid", func(r *cuptiactivity.ScopedRecord) { r.Process = "next-generation" }, "not_observed"},
		{"different correlation", func(r *cuptiactivity.ScopedRecord) { r.Record.CorrelationID = 43 }, "not_observed"},
		{"missing correlation", func(r *cuptiactivity.ScopedRecord) { r.Record.CorrelationID = 0 }, "not_observed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := activity
			tt.change(&r)
			// Link follows the activity: delivery order cannot determine causality.
			got, err := cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{r, link}, cupti.ExternalCorrelationKindCustom0)
			if err != nil || len(got) != 1 || got[0].Status != tt.status || got[0].Index != 0 {
				t.Fatalf("matches=%v err=%v", got, err)
			}
			if tt.status == "matched" && got[0].ExternalID != 1<<63+5 {
				t.Fatal("external ID lost precision")
			}
		})
	}
	other := link
	other.Record.ExternalID = 0
	got, err := cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{link, other, activity}, cupti.ExternalCorrelationKindCustom0)
	if err != nil || got[0].Status != "ambiguous" {
		t.Fatalf("ambiguous=%v err=%v", got, err)
	}
	got, err = cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{other, other, activity}, cupti.ExternalCorrelationKindCustom0)
	if err != nil || got[0].Status != "matched" || got[0].ExternalID != 0 {
		t.Fatalf("zero/duplicate=%v err=%v", got, err)
	}
	got, err = cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{link, activity}, cupti.ExternalCorrelationKindCustom1)
	if err != nil || got[0].Status != "not_observed" {
		t.Fatalf("namespace=%v err=%v", got, err)
	}
	activity.Process = ""
	if _, err := cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{activity}, cupti.ExternalCorrelationKindCustom0); err == nil {
		t.Fatal("missing scope accepted")
	}
}

func ExampleMatchExternal() {
	r := cuptiactivity.ScopedRecord{Session: "capture", Run: "live", Process: "pid-17-generation-1", Record: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7}}
	matches, err := cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{r}, cupti.ExternalCorrelationKindCustom0)
	fmt.Println(matches[0].Status, err)
	// Output: not_observed <nil>
}

func TestMemcpyExternalPaths(t *testing.T) {
	base := cuptiactivity.ScopedRecord{Session: "s", Run: "r", Process: "generation"}
	copyRecord, driver, runtime := base, base, base
	copyRecord.Record = cuptiactivity.Record{Kind: cupti.ActivityKindMemcpy, CorrelationID: 7, RuntimeCorrelationID: 19, Start: 10, End: 30, Complete: true}
	driver.Record = cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalID: 1<<63 + 5}
	runtime.Record = driver.Record
	runtime.Record.CorrelationID = 19
	for _, tt := range []struct {
		name   string
		links  []cuptiactivity.ScopedRecord
		status string
	}{
		{"driver only", []cuptiactivity.ScopedRecord{driver}, "matched"},
		{"runtime only", []cuptiactivity.ScopedRecord{runtime}, "matched"},
		{"both equal", []cuptiactivity.ScopedRecord{driver, runtime}, "matched"},
		{"absent", nil, "not_observed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			records := append([]cuptiactivity.ScopedRecord{copyRecord}, tt.links...)
			got, err := cuptiactivity.MatchExternal(records, cupti.ExternalCorrelationKindCustom0)
			if err != nil || len(got) != 1 || got[0].Status != tt.status {
				t.Fatalf("matches=%v err=%v", got, err)
			}
			if tt.status == "matched" && got[0].ExternalID != driver.Record.ExternalID {
				t.Fatal("lost uint64 external identity")
			}
			coverage, err := cuptiactivity.ExternalCoverage(records, cupti.ExternalCorrelationKindCustom0)
			if err != nil || len(coverage) != 1 || coverage[0].Total.Duration != 20 {
				t.Fatalf("coverage=%v err=%v", coverage, err)
			}
			if tt.status == "matched" && coverage[0].Matched.Duration != 20 {
				t.Fatal("runtime path not reflected in duration coverage")
			}
		})
	}
	runtime.Record.ExternalID = 42
	for _, records := range [][]cuptiactivity.ScopedRecord{{copyRecord, driver, runtime}, {copyRecord, runtime, driver}} {
		got, err := cuptiactivity.MatchExternal(records, cupti.ExternalCorrelationKindCustom0)
		if err != nil || got[0].Status != "ambiguous" || got[0].ExternalID != 0 {
			t.Fatalf("conflicting paths=%v err=%v", got, err)
		}
	}
	runtime.Process = "other-generation"
	got, err := cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{copyRecord, runtime}, cupti.ExternalCorrelationKindCustom0)
	if err != nil || got[0].Status != "not_observed" {
		t.Fatalf("cross-generation runtime link=%v err=%v", got, err)
	}
	runtime.Process = base.Process
	copyRecord.Record.Kind = cupti.ActivityKindConcurrentKernel
	got, err = cuptiactivity.MatchExternal([]cuptiactivity.ScopedRecord{copyRecord, runtime}, cupti.ExternalCorrelationKindCustom0)
	if err != nil || got[0].Status != "not_observed" {
		t.Fatalf("kernel used memcpy-only identity=%v err=%v", got, err)
	}
}

func ExampleScopedRecord() {
	r := cuptiactivity.ScopedRecord{Session: "capture", Run: "live", Process: "pid-17-generation-1"}
	fmt.Println(r.Run, r.Process)
	// Output: live pid-17-generation-1
}

func ExampleExternalMatch() {
	m := cuptiactivity.ExternalMatch{Index: 0, Status: "not_observed"}
	fmt.Println(m.Index, m.Status)
	// Output: 0 not_observed
}
