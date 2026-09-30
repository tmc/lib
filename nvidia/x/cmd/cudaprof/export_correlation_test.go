package main

import (
	"bytes"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"testing"
)

func TestTraceCorrelationAnnotations(t *testing.T) {
	records := []traceRecord{
		{journal: "j", kind: trackHost, rec: cuptiactivity.Record{Kind: cupti.ActivityKindDriver, CorrelationID: 7, Start: 10, End: 15, Complete: true, ProcessID: 1, ThreadID: 2}},
		{journal: "j", kind: trackNVTX, rec: cuptiactivity.Record{Kind: cupti.ActivityKindMarker, Name: "operation", Start: 5, End: 20, Complete: true, ProcessID: 1, ThreadID: 2}},
		{journal: "j", kind: trackDevice, rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "kernel", CorrelationID: 7, Start: 30, End: 40, Complete: true}},
		{journal: "other", kind: trackDevice, rec: cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "other", CorrelationID: 7, Start: 30, End: 40, Complete: true}},
	}
	annotateTraceCorrelations(records)
	if len(records[2].annotations) <= len(records[3].annotations) {
		t.Fatal("matched link has no additional provenance")
	}
	var out bytes.Buffer
	if _, err := writeTrace(&out, records); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"submission_status", "submission_api", "operation_range", "operation"} {
		if !bytes.Contains(out.Bytes(), []byte(want)) {
			t.Errorf("trace lacks %q", want)
		}
	}
}
