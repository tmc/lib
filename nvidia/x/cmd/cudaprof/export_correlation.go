package main

import (
	"encoding/json"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func annotateTraceCorrelations(trace []traceRecord) {
	var records timeline.Records
	var indexes []int
	for i, t := range trace {
		r := t.rec
		switch t.kind {
		case trackHost:
			domain := "driver"
			if r.Kind == cupti.ActivityKindRuntime {
				domain = "runtime"
			}
			records.Calls = append(records.Calls, timeline.Call{Journal: t.journal, Process: r.ProcessID, Domain: domain, CallbackID: r.CallbackID, Thread: r.ThreadID, Correlation: r.CorrelationID, Start: int64(r.Start), End: int64(r.End)})
		case trackNVTX:
			if r.End > r.Start {
				records.Ranges = append(records.Ranges, timeline.Range{Journal: t.journal, Process: r.ProcessID, Thread: r.ThreadID, Name: r.Name, Start: int64(r.Start), End: int64(r.End)})
			}
		case trackDevice:
			if r.Kind != cupti.ActivityKindConcurrentKernel {
				continue
			}
			records.Activities = append(records.Activities, timeline.Activity{Journal: t.journal, Correlation: r.CorrelationID, Subtype: "kernel", GraphID: r.GraphID})
			indexes = append(indexes, i)
		}
	}
	for _, m := range timeline.Correlations(records, "driver") {
		t := &trace[indexes[m.Activity]]
		t.annotations = append(t.annotations, perfetto.NewStringAnnotation("capture_journal", t.journal), perfetto.NewStringAnnotation("submission_status", m.APIStatus), perfetto.NewStringAnnotation("operation_range_status", m.RangeStatus), perfetto.NewStringAnnotation("application_metadata_status", m.MetadataStatus))
		if m.Call >= 0 {
			c := records.Calls[m.Call]
			t.annotations = append(t.annotations, perfetto.NewStringAnnotation("submission_api", c.Name()), perfetto.NewUintAnnotation("submission_start_ns", uint64(c.Start)), perfetto.NewUintAnnotation("submission_end_ns", uint64(c.End)), perfetto.NewUintAnnotation("submission_thread_id", uint64(c.Thread)))
		}
		if m.Range >= 0 {
			t.annotations = append(t.annotations, perfetto.NewStringAnnotation("operation_range", records.Ranges[m.Range].Name))
		}
		if m.Metadata != nil {
			b, _ := json.Marshal(m.Metadata)
			t.annotations = append(t.annotations, perfetto.NewStringAnnotation("application_metadata", string(b)))
		}
	}
}
