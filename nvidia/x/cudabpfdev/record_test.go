package cudabpfdev

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestExecutionRecord(t *testing.T) {
	e, source := runFixture(t, &fakeRun{})
	if _, err := e.Record(); err == nil {
		t.Fatal("record before execution")
	}
	before := time.Now().UnixNano()
	if err := e.execute(context.Background(), source, 2, 5); err != nil {
		t.Fatal(err)
	}
	after := time.Now().UnixNano()
	rec, err := e.Record()
	if err != nil {
		t.Fatal(err)
	}
	if rec.TimestampNs < before || rec.TimestampNs > after || rec.DurationNs != 0 || rec.CorrelationID != 0 {
		t.Fatalf("bad timing: %+v", rec)
	}
	if rec.KernelName != source.launch.Artifact.EntryPoint {
		t.Fatal("lost entry identity")
	}
	for key, want := range map[string]string{
		"record_kind": "device_bpf_snapshot", "duration_status": "unavailable",
		"timestamp_source": "host_snapshot_publication", "device_bpf.map_slots": "2",
		"device_bpf.map_misses": "29", "device_bpf.active_mask": "5",
		"device_bpf.image_digest": source.launch.Artifact.Digest,
	} {
		if rec.Labels[key] != want {
			t.Errorf("%s=%q want %q", key, rec.Labels[key], want)
		}
	}
	rec.Labels["device_bpf.map_slots"] = "corrupt"
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	rec, err = e.Record()
	if err != nil || rec.Labels["device_bpf.map_slots"] != "2" {
		t.Fatal("record lost or aliased")
	}
	// Two host boundary observations must not manufacture a GPU execution span.
	second, _ := e.Record()
	second.TimestampNs += 1000
	_, gpu, overlap, reliable := cudaflightrecorder.AlignSpans(second.TimestampNs, 10000, before, true, []cudaflightrecorder.Record{rec, second}, 1)
	if gpu != 0 || overlap != 0 || reliable {
		t.Fatalf("fake GPU span: %d %d %v", gpu, overlap, reliable)
	}

	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	defer r.Stop()
	r.Push(rec)
	var buf bytes.Buffer
	if _, err := r.Snapshot(context.Background(), &buf, "device-bpf-boundary"); err != nil {
		t.Fatal(err)
	}
	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatal(err)
	}
	var records []cudaflightrecorder.Record
	if err := json.Unmarshal(bundle.Files["gpu_records.json"], &records); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(records, []cudaflightrecorder.Record{rec}) {
		t.Fatalf("durable record mismatch: %+v", records)
	}
}

func TestFailedExecutionHasNoRecord(t *testing.T) {
	for _, stage := range []string{"load", "launch", "wait", "read2", "restore"} {
		t.Run(stage, func(t *testing.T) {
			f := &fakeRun{fail: stage}
			e, s := runFixture(t, f)
			if err := e.execute(context.Background(), s, 2, 5); err == nil {
				t.Fatal("expected failure")
			}
			f.fail = ""
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err := e.Record(); err == nil {
				t.Fatal("failed execution has completed boundary record")
			}
		})
	}
	var e *Execution
	if _, err := e.Record(); err == nil {
		t.Fatal("nil execution record")
	}
}
