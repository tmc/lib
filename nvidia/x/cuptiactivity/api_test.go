package cuptiactivity_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestMatchAPI(t *testing.T) {
	base := cuptiactivity.ScopedRecord{Session: "s", Run: "r", Process: "generation"}
	api, device := base, base
	api.Record = cuptiactivity.Record{Kind: cupti.ActivityKindDriver, CorrelationID: 7, ThreadID: 18, Start: 100, End: 110, Complete: true}
	device.Record = cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7, Name: "kernel"}
	for _, tt := range []struct {
		name   string
		change func(*cuptiactivity.ScopedRecord, *cuptiactivity.ScopedRecord)
		kind   cupti.ActivityKind
		status string
	}{
		{"driver kernel", func(a, d *cuptiactivity.ScopedRecord) {}, cupti.ActivityKindDriver, "matched"},
		{"runtime kernel", func(a, d *cuptiactivity.ScopedRecord) { a.Record.Kind = cupti.ActivityKindRuntime }, cupti.ActivityKindRuntime, "matched"},
		{"wrong domain", func(a, d *cuptiactivity.ScopedRecord) {}, cupti.ActivityKindRuntime, "not_observed"},
		{"session", func(a, d *cuptiactivity.ScopedRecord) { d.Session = "other" }, cupti.ActivityKindDriver, "not_observed"},
		{"run", func(a, d *cuptiactivity.ScopedRecord) { d.Run = "other" }, cupti.ActivityKindDriver, "not_observed"},
		{"process generation", func(a, d *cuptiactivity.ScopedRecord) { d.Process = "next" }, cupti.ActivityKindDriver, "not_observed"},
		{"missing id", func(a, d *cuptiactivity.ScopedRecord) { a.Record.CorrelationID = 0; d.Record.CorrelationID = 0 }, cupti.ActivityKindDriver, "not_observed"},
		{"memcpy driver", func(a, d *cuptiactivity.ScopedRecord) {
			d.Record.Kind = cupti.ActivityKindMemcpy
			d.Record.RuntimeCorrelationID = 99
		}, cupti.ActivityKindDriver, "matched"},
		{"memcpy runtime", func(a, d *cuptiactivity.ScopedRecord) {
			a.Record.Kind = cupti.ActivityKindRuntime
			a.Record.CorrelationID = 99
			d.Record.Kind = cupti.ActivityKindMemcpy
			d.Record.RuntimeCorrelationID = 99
		}, cupti.ActivityKindRuntime, "matched"},
		{"memcpy no runtime fallback", func(a, d *cuptiactivity.ScopedRecord) {
			a.Record.Kind = cupti.ActivityKindRuntime
			d.Record.Kind = cupti.ActivityKindMemcpy
		}, cupti.ActivityKindRuntime, "not_observed"},
		{"memset driver", func(a, d *cuptiactivity.ScopedRecord) { d.Record.Kind = cupti.ActivityKindMemset }, cupti.ActivityKindDriver, "matched"},
		{"memset no runtime link", func(a, d *cuptiactivity.ScopedRecord) {
			a.Record.Kind = cupti.ActivityKindRuntime
			d.Record.Kind = cupti.ActivityKindMemset
		}, cupti.ActivityKindRuntime, "not_observed"},
		{"incomplete failed API", func(a, d *cuptiactivity.ScopedRecord) {
			a.Record.Complete = false
			a.Record.End = 0
			a.Record.ReturnValue = 600
		}, cupti.ActivityKindDriver, "matched"},
		{"same name and time wrong id", func(a, d *cuptiactivity.ScopedRecord) {
			d.Record = a.Record
			d.Record.Kind = cupti.ActivityKindConcurrentKernel
			d.Record.CorrelationID = 8
		}, cupti.ActivityKindDriver, "not_observed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a, d := api, device
			tt.change(&a, &d)
			records := []cuptiactivity.ScopedRecord{d, a}
			before := append([]cuptiactivity.ScopedRecord(nil), records...)
			got, err := cuptiactivity.MatchAPI(records, tt.kind)
			wantIndex := -1
			if tt.status == "matched" {
				wantIndex = 1
			}
			want := []cuptiactivity.APIMatch{{Index: 0, APIIndex: wantIndex, Status: tt.status}}
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("got %v, %v; want %v", got, err, want)
			}
			if !reflect.DeepEqual(records, before) {
				t.Fatal("input modified")
			}
		})
	}
	for _, field := range []string{"duplicate", "thread", "time", "return"} {
		t.Run(field, func(t *testing.T) {
			other := api
			switch field {
			case "thread":
				other.Record.ThreadID++
			case "time":
				other.Record.Start++
			case "return":
				other.Record.ReturnValue++
			}
			got, err := cuptiactivity.MatchAPI([]cuptiactivity.ScopedRecord{device, api, other, device}, cupti.ActivityKindDriver)
			status, index := "ambiguous", -1
			if field == "duplicate" {
				status, index = "matched", 1
			}
			want := []cuptiactivity.APIMatch{{Index: 0, APIIndex: index, Status: status}, {Index: 3, APIIndex: index, Status: status}}
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("got %v, %v; want %v", got, err, want)
			}
		})
	}
	if _, err := cuptiactivity.MatchAPI(nil, cupti.ActivityKindMemcpy); err == nil {
		t.Fatal("invalid kind accepted")
	}
	base.Process = ""
	if out, err := cuptiactivity.MatchAPI([]cuptiactivity.ScopedRecord{device, base}, cupti.ActivityKindDriver); err == nil || out != nil {
		t.Fatal("missing scope accepted")
	}
}

func ExampleMatchAPI() {
	r := cuptiactivity.ScopedRecord{Session: "capture", Run: "live", Process: "generation-1", Record: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7}}
	a := r
	a.Record = cuptiactivity.Record{Kind: cupti.ActivityKindDriver, CorrelationID: 7, ThreadID: 12}
	records := []cuptiactivity.ScopedRecord{r, a}
	links, err := cuptiactivity.MatchAPI(records, cupti.ActivityKindDriver)
	fmt.Println(links[0].Status, records[links[0].APIIndex].Record.ThreadID, err)
	// Output: matched 12 <nil>
}

func ExampleAPIMatch() {
	m := cuptiactivity.APIMatch{Index: 0, APIIndex: -1, Status: "not_observed"}
	fmt.Println(m.Index, m.APIIndex, m.Status)
	// Output: 0 -1 not_observed
}
