package cudaprobe

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestProbeRecorderRoundTrip(t *testing.T) {
	e, err := Decode(capture("cuLaunchKernel", 1, 99, 2, 3, 4, 32, 1, 1, 0, 55))
	if err != nil {
		t.Fatal(err)
	}
	e.Identity.Provenance = "synthetic"
	e.Identity.Interposed = true
	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	defer r.Stop()
	now := time.Now()
	for _, at := range []time.Time{now.Add(-time.Millisecond), now} {
		record, err := e.Record(at)
		if err != nil {
			t.Fatal(err)
		}
		r.Push(record)
	}
	var b bytes.Buffer
	m, err := r.Snapshot(context.Background(), &b, "probe")
	if err != nil {
		t.Fatal(err)
	}
	if m.GPUSpanNs != 0 || m.OverlapReliable {
		t.Fatalf("host probe produced GPU span: %+v", m)
	}
	bundle, err := cudaflightrecorder.ReadBundle(&b)
	if err != nil {
		t.Fatal(err)
	}
	var records []cudaflightrecorder.Record
	if err := json.Unmarshal(bundle.Files["gpu_records.json"], &records); err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 {
		t.Fatalf("records=%d", len(records))
	}
	got := records[0]
	for k, v := range map[string]string{"probe.build_id": "synthetic-build", "probe.symbol": "cuLaunchKernel", "probe.interposed": "true", "provenance": "synthetic", "probe.grid": "[2 3 4]", "probe.block": "[32 1 1]", "tensor_shapes": "unavailable", "tensor_dtypes": "unavailable", "duration_status": "unavailable", "timestamp_source": "collector_receipt", "probe.clock": "host_monotonic"} {
		if got.Labels[k] != v {
			t.Errorf("%s=%q want %q", k, got.Labels[k], v)
		}
	}
	if got.DurationNs != 0 || got.KernelName != "" || got.TimestampNs == int64(e.Timestamp) {
		t.Fatal("invented GPU execution or clock calibration")
	}
	if _, err := e.Record(time.Time{}); err == nil {
		t.Fatal("accepted absent receipt timestamp")
	}
}

func TestCompareCountsPreservesDisagreement(t *testing.T) {
	a := Count{Process: Process{PID: 7, Generation: 99}, Interval: "fixture-run-1", Source: "uprobe", Value: 3}
	b := a
	b.Source = "application"
	b.Value = 5
	c, err := CompareCounts(a, b)
	if err != nil || c.Missing != 2 || c.Excess != 0 || c.Probe != a || c.Independent != b {
		t.Fatalf("comparison=%+v error=%v", c, err)
	}
	a.Value = ^uint64(0)
	b.Value = 0
	c, err = CompareCounts(a, b)
	if err != nil || c.Excess != ^uint64(0) {
		t.Fatalf("overflow: %+v %v", c, err)
	}
	b.Source = a.Source
	if _, err := CompareCounts(a, b); err == nil {
		t.Fatal("accepted same provenance")
	}
	b.Source = "application"
	b.Process.Generation++
	if _, err := CompareCounts(a, b); err == nil {
		t.Fatal("accepted different process lifetime")
	}
}
