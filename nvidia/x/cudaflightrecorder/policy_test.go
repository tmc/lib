package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabpf"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

var _ cudabpf.RecordSink = (*cudaflightrecorder.Recorder)(nil)

func TestPolicyDoesNotExtendGPUSpan(t *testing.T) {
	records := []cudaflightrecorder.Record{
		{TimestampNs: 10, Labels: map[string]string{"record_kind": "driver_policy"}},
		{TimestampNs: 900, Labels: map[string]string{"record_kind": "driver_policy"}},
	}
	_, span, overlap, reliable := cudaflightrecorder.AlignSpans(1000, 1000, 1, true, records, 1)
	if span != 0 || overlap != 0 || reliable {
		t.Fatalf("policy-only span=%d overlap=%d reliable=%t", span, overlap, reliable)
	}
	records = append(records, cudaflightrecorder.Record{TimestampNs: 100, DurationNs: 50})
	_, span, overlap, reliable = cudaflightrecorder.AlignSpans(1000, 1000, 1, true, records, 1)
	if span != 50 || overlap != 50 || !reliable {
		t.Fatalf("mixed span=%d overlap=%d reliable=%t", span, overlap, reliable)
	}
}

func TestPolicySnapshot(t *testing.T) {
	for _, fault := range []bool{false, true} {
		name := "normal"
		if fault {
			name = "fault"
		}
		t.Run(name, func(t *testing.T) {
			r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1})
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Start(); err != nil {
				t.Fatal(err)
			}
			defer r.Stop()
			observer := cudabpf.SinkObserver{Sink: r, Clock: func() int64 { return time.Now().UnixNano() }}
			observer.Observe(cudabpf.Outcome{Hook: "on_bind", Submitted: true})
			observer.Observe(cudabpf.Outcome{
				Hook: "gpu_block_access", Consulted: true, HadOpinion: true,
				Submitted: true, DriverAttempted: true, DriverConflict: true,
				Fallback: true, Reason: "driver retained placement",
			})
			var buf bytes.Buffer
			var m *cudaflightrecorder.Manifest
			if fault {
				m, err = r.FaultSnapshot(context.Background(), &buf, cudaflightrecorder.FaultInfo{})
			} else {
				m, err = r.Snapshot(context.Background(), &buf, "policy")
			}
			if err != nil {
				t.Fatal(err)
			}
			if m.GPURecords != 1 || m.GPUDropped != 1 {
				t.Fatalf("records=%d dropped=%d", m.GPURecords, m.GPUDropped)
			}
			b, err := cudaflightrecorder.ReadBundle(&buf)
			if err != nil {
				t.Fatal(err)
			}
			var records []cudaflightrecorder.Record
			if err := json.Unmarshal(b.Files["gpu_records.json"], &records); err != nil {
				t.Fatal(err)
			}
			if len(records) != 1 {
				t.Fatalf("records=%d", len(records))
			}
			got := records[0]
			for key, want := range map[string]string{
				"hook": "gpu_block_access", "record_kind": "driver_policy",
				"submitted": "true", "driver_attempted": "true", "driver_conflict": "true",
				"fallback": "true", "reason": "driver retained placement",
				"duration_status": "unavailable", "timestamp_source": "host_event",
			} {
				if got.Labels[key] != want {
					t.Errorf("%s=%q, want %q", key, got.Labels[key], want)
				}
			}
			if got.KernelName != "" || got.DurationNs != 0 {
				t.Fatalf("policy became kernel execution: %+v", got)
			}
		})
	}
}

func TestPolicyLabelsOwnedByRecorder(t *testing.T) {
	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	defer r.Stop()
	labels := map[string]string{"reason": "original", "duration_status": "measured", "hook": "wrong"}
	r.PushPolicyRecord(0, "on_bind", labels)
	labels["reason"] = "changed"
	if labels["duration_status"] != "measured" || labels["hook"] != "wrong" {
		t.Fatal("mutated caller labels")
	}
	var buf bytes.Buffer
	if _, err := r.Snapshot(context.Background(), &buf, "ownership"); err != nil {
		t.Fatal(err)
	}
	b, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatal(err)
	}
	var records []cudaflightrecorder.Record
	if err := json.Unmarshal(b.Files["gpu_records.json"], &records); err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 {
		t.Fatalf("records=%d", len(records))
	}
	got := records[0]
	if got.Labels["reason"] != "original" || got.Labels["hook"] != "on_bind" || got.Labels["duration_status"] != "unavailable" || got.Labels["timestamp_source"] != "collector_receipt" {
		t.Fatalf("labels=%v", got.Labels)
	}
}
